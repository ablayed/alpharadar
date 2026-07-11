from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from src.reflection.historical_backfill import HistoricalPriceFetcher, RepositoryHistoricalPriceFetcher
from src.reflection.tracker import (
    _direction_from_change,
    _fetch_prediction_rows,
    _normalize_prediction_direction,
    _sqlite_timestamp,
    _update_prediction,
    build_outcome_passes,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PassSummary:
    scanned: int = 0
    resolved: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


@dataclass(slots=True)
class BackfillSummary:
    dry_run: bool
    pass_summaries: dict[str, PassSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "passes": {
                label: {
                    "scanned": summary.scanned,
                    "resolved": summary.resolved,
                    "skipped": dict(sorted(summary.skipped.items())),
                }
                for label, summary in self.pass_summaries.items()
            },
        }

    def to_text(self) -> str:
        diagnostic_reasons = {"not_old_enough", "dependency_pass_not_yet_satisfied", "already_resolved"}
        lines = [f"mode: {'dry-run' if self.dry_run else 'live'}"]
        for label, summary in self.pass_summaries.items():
            fetched_skip_items: list[tuple[str, int]] = []
            diagnostic_items: list[tuple[str, int]] = []
            for reason, count in sorted(summary.skipped.items()):
                if reason in diagnostic_reasons:
                    diagnostic_items.append((reason, count))
                else:
                    fetched_skip_items.append((reason, count))

            lines.append(
                f"{label}: fetched_for_processing={summary.scanned}, resolved={summary.resolved}"
            )
            if fetched_skip_items:
                for reason, count in fetched_skip_items:
                    lines.append(f"- {label} skipped_after_fetch {reason}: {count}")
            else:
                lines.append(f"- {label} skipped_after_fetch none")
            if diagnostic_items:
                for reason, count in diagnostic_items:
                    lines.append(f"- {label} diagnostic_excluded {reason}: {count}")
            else:
                lines.append(f"- {label} diagnostic_excluded none")
        return "\n".join(lines)


async def _load_prediction_state(db_path: str) -> list[aiosqlite.Row]:
    async with aiosqlite.connect(db_path) as connection:
        connection.row_factory = aiosqlite.Row
        cursor = await connection.execute(
            """
            SELECT id,
                   created_at,
                   symbol,
                   asset_type,
                   predicted_direction,
                   price_at_prediction,
                   outcome_recorded,
                   actual_price_1d,
                   actual_price_3d,
                   actual_price_5d,
                   actual_price_7d
            FROM predictions
            ORDER BY created_at, id
            """
        )
        return await cursor.fetchall()


def _parse_created_at(value: object) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def _classify_skip_reason(row: aiosqlite.Row, *, label: str, before: datetime) -> str | None:
    created_at = _parse_created_at(row["created_at"])
    if created_at >= before:
        return "not_old_enough"

    if label == "1d":
        if int(row["outcome_recorded"]) != 0 or row["actual_price_1d"] is not None:
            return "already_resolved"
        return None

    if row["actual_price_1d"] is None or int(row["outcome_recorded"]) != 1:
        return "dependency_pass_not_yet_satisfied"

    if label == "3d" and row["actual_price_3d"] is not None:
        return "already_resolved"
    if label == "5d" and row["actual_price_5d"] is not None:
        return "already_resolved"
    if label == "7d" and row["actual_price_7d"] is not None:
        return "already_resolved"
    return None


def _horizon_delta(label: str) -> timedelta:
    mapping = {
        "1d": timedelta(days=1),
        "3d": timedelta(days=3),
        "5d": timedelta(days=5),
        "7d": timedelta(days=7),
    }
    return mapping[label]


async def _run_backfill_passes(
    *,
    db_path: str,
    historical_price_fetcher: HistoricalPriceFetcher,
    now: datetime,
) -> dict[str, PassSummary]:
    pass_summaries = {pass_.label: PassSummary() for pass_ in build_outcome_passes(now)}

    for outcome_pass in build_outcome_passes(now):
        state_rows = await _load_prediction_state(db_path)
        pass_summary = pass_summaries[outcome_pass.label]
        for row in state_rows:
            skip_reason = _classify_skip_reason(
                row,
                label=outcome_pass.label,
                before=outcome_pass.before,
            )
            if skip_reason is not None:
                pass_summary.skipped[skip_reason] += 1

        rows = await _fetch_prediction_rows(
            db_path,
            before=_sqlite_timestamp(outcome_pass.before),
            where_clause=outcome_pass.where_clause,
        )
        pass_summary.scanned = len(rows)

        for row in rows:
            raw_predicted_price = row["price_at_prediction"]
            try:
                predicted_price = float(raw_predicted_price)
            except (TypeError, ValueError):
                predicted_price = None

            if predicted_price is None or not math.isfinite(predicted_price) or predicted_price <= 0:
                pass_summary.skipped["invalid_price_at_prediction"] += 1
                logger.warning(
                    "Skipping historical outcome backfill for prediction id=%s symbol=%s: "
                    "invalid_price_at_prediction (value=%r)",
                    row["id"],
                    row["symbol"],
                    raw_predicted_price,
                )
                continue

            created_at = _parse_created_at(row["created_at"]) if "created_at" in row.keys() else None
            if created_at is None:
                async with aiosqlite.connect(db_path) as connection:
                    connection.row_factory = aiosqlite.Row
                    cursor = await connection.execute(
                        "SELECT created_at FROM predictions WHERE id = ?",
                        (int(row["id"]),),
                    )
                    created_row = await cursor.fetchone()
                created_at = _parse_created_at(created_row["created_at"])

            target_at = created_at + _horizon_delta(outcome_pass.label)
            historical_price = await historical_price_fetcher(
                str(row["symbol"]),
                str(row["asset_type"]),
                target_at,
            )
            if historical_price is None:
                pass_summary.skipped["no_historical_price_found"] += 1
                continue

            predicted_direction = _normalize_prediction_direction(str(row["predicted_direction"]))
            change_pct = round(((historical_price - predicted_price) / predicted_price) * 100, 2)
            actual_direction = _direction_from_change(change_pct, neutral_band=outcome_pass.neutral_band)

            updates: dict[str, object] = {
                outcome_pass.price_column: historical_price,
                outcome_pass.change_column: change_pct,
                outcome_pass.correct_column: 1 if predicted_direction == actual_direction else 0,
            }
            if outcome_pass.update_actual_direction:
                updates["actual_direction"] = actual_direction
            if outcome_pass.mark_outcome_recorded:
                updates["outcome_recorded"] = 1
                updates["outcome_recorded_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            # Backfilled prices are daily-close historical marks, so they intentionally do not feed
            # ensemble performance weighting. Mixing them with the live path's finer-grained marks
            # would bias MLEnsemble._performance with lower-fidelity outcomes.
            await _update_prediction(db_path, int(row["id"]), updates)
            pass_summary.resolved += 1

    return pass_summaries


async def backfill_historical_outcomes(
    *,
    db_path: str,
    historical_price_fetcher: HistoricalPriceFetcher,
    dry_run: bool = True,
    now: datetime | None = None,
) -> BackfillSummary:
    current_time = now or datetime.now(UTC)
    effective_db_path = db_path
    temp_db_path: Path | None = None

    if dry_run:
        with tempfile.NamedTemporaryFile(prefix="alpharadar-backfill-", suffix=".db", delete=False) as handle:
            temp_db_path = Path(handle.name)
        with sqlite3.connect(db_path) as source_connection, sqlite3.connect(temp_db_path) as temp_connection:
            source_connection.backup(temp_connection)
        effective_db_path = str(temp_db_path)

    try:
        pass_summaries = await _run_backfill_passes(
            db_path=effective_db_path,
            historical_price_fetcher=historical_price_fetcher,
            now=current_time,
        )
        return BackfillSummary(dry_run=dry_run, pass_summaries=pass_summaries)
    finally:
        if temp_db_path is not None:
            for attempt in range(5):
                try:
                    temp_db_path.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 4:
                        break
                    await asyncio.sleep(0.1)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical prediction outcomes.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Write backfilled outcomes to the database. Default behavior is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request dry-run mode. This is already the default.",
    )
    return parser


async def _run_cli(db_path: str, *, dry_run: bool) -> BackfillSummary:
    fetcher = RepositoryHistoricalPriceFetcher(db_path=db_path)
    return await backfill_historical_outcomes(
        db_path=db_path,
        historical_price_fetcher=fetcher,
        dry_run=dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = asyncio.run(_run_cli(args.db_path, dry_run=not args.live))
    print(summary.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
