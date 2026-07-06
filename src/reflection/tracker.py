from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from src.ml.ensemble import MLEnsemble

logger = logging.getLogger(__name__)

PriceFetcher = Callable[[str, str], Awaitable[float | None]]
_SQLITE_NOW = object()


@dataclass(frozen=True, slots=True)
class _OutcomePass:
    label: str
    before: datetime
    where_clause: str
    price_column: str
    change_column: str
    correct_column: str
    neutral_band: float
    update_actual_direction: bool = False
    mark_outcome_recorded: bool = False


class PredictionTracker:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def log_prediction(
        self,
        symbol: str,
        asset_type: str,
        predicted_direction: str,
        confidence: float,
        price_at_prediction: float,
        source: str,
        reasoning: str = "",
        event_type: str | None = None,
        predicted_target: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> int:
        normalized_direction = _normalize_prediction_direction(predicted_direction)
        normalized_source = source.strip().lower()
        metadata_json = json.dumps(metadata) if metadata is not None else None
        async with aiosqlite.connect(self._db_path) as connection:
            cursor = await connection.execute(
                """
                INSERT INTO predictions
                (symbol, asset_type, event_type, predicted_direction, predicted_target,
                 confidence, reasoning, metadata, source, price_at_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.upper(),
                    asset_type,
                    event_type,
                    normalized_direction,
                    predicted_target,
                    confidence,
                    reasoning,
                    metadata_json,
                    normalized_source,
                    price_at_prediction,
                ),
            )
            await connection.commit()
            return int(cursor.lastrowid)

    async def record_outcomes(
        self,
        price_fetcher: PriceFetcher,
        ensemble: MLEnsemble | None = None,
    ) -> int:
        now = datetime.now(UTC)
        passes = build_outcome_passes(now)

        total_updates = 0
        for outcome_pass in passes:
            total_updates += await self._record_outcome_pass(price_fetcher, outcome_pass, ensemble)

        logger.info("Recorded %d prediction outcomes (1d + 3d + 5d + 7d)", total_updates)
        return total_updates

    async def _record_outcome_pass(
        self,
        price_fetcher: PriceFetcher,
        outcome_pass: _OutcomePass,
        ensemble: MLEnsemble | None,
    ) -> int:
        rows = await _fetch_prediction_rows(
            self._db_path,
            before=_sqlite_timestamp(outcome_pass.before),
            where_clause=outcome_pass.where_clause,
        )
        recorded_count = 0
        performance_updated = False

        for row in rows:
            current_price = await _safe_fetch_current_price(
                price_fetcher,
                row["symbol"],
                row["asset_type"],
                outcome_label=outcome_pass.label,
            )
            if current_price is None:
                continue

            predicted_price = float(row["price_at_prediction"])
            change_pct = round(((current_price - predicted_price) / predicted_price) * 100, 2)
            actual_direction = _direction_from_change(change_pct, neutral_band=outcome_pass.neutral_band)
            is_correct = 1 if row["predicted_direction"] == actual_direction else 0

            updates: dict[str, object] = {
                outcome_pass.price_column: current_price,
                outcome_pass.change_column: change_pct,
                outcome_pass.correct_column: is_correct,
            }
            if outcome_pass.update_actual_direction:
                updates["actual_direction"] = actual_direction
            if outcome_pass.mark_outcome_recorded:
                updates["outcome_recorded"] = 1
                updates["outcome_recorded_at"] = _SQLITE_NOW

            await _update_prediction(self._db_path, int(row["id"]), updates)
            recorded_count += 1
            if outcome_pass.mark_outcome_recorded:
                performance_updated = _record_ml_performance_outcome(
                    ensemble,
                    row,
                    price_change_pct=change_pct,
                ) or performance_updated

        if performance_updated and ensemble is not None:
            ensemble._performance.save()

        return recorded_count


def build_outcome_passes(now: datetime) -> tuple[_OutcomePass, ...]:
    return (
        _OutcomePass(
            label="1d",
            before=now - timedelta(hours=24),
            where_clause="outcome_recorded = 0 AND actual_price_1d IS NULL",
            price_column="actual_price_1d",
            change_column="actual_change_1d_pct",
            correct_column="is_correct_1d",
            neutral_band=0.5,
            update_actual_direction=True,
            mark_outcome_recorded=True,
        ),
        _OutcomePass(
            label="3d",
            before=now - timedelta(days=3),
            where_clause=(
                "outcome_recorded = 1 "
                "AND actual_price_1d IS NOT NULL "
                "AND actual_price_3d IS NULL"
            ),
            price_column="actual_price_3d",
            change_column="actual_change_3d_pct",
            correct_column="is_correct_3d",
            neutral_band=1.0,
        ),
        _OutcomePass(
            label="5d",
            before=now - timedelta(days=5),
            where_clause=(
                "outcome_recorded = 1 "
                "AND actual_price_1d IS NOT NULL "
                "AND actual_price_5d IS NULL"
            ),
            price_column="actual_price_5d",
            change_column="actual_change_5d_pct",
            correct_column="is_correct_5d",
            neutral_band=2.0,
        ),
        _OutcomePass(
            label="7d",
            before=now - timedelta(days=7),
            where_clause=(
                "outcome_recorded = 1 "
                "AND actual_price_1d IS NOT NULL "
                "AND actual_price_7d IS NULL"
            ),
            price_column="actual_price_7d",
            change_column="actual_change_7d_pct",
            correct_column="is_correct_7d",
            neutral_band=2.0,
        ),
    )


async def _fetch_prediction_rows(
    db_path: str,
    *,
    before: str,
    where_clause: str,
) -> list[aiosqlite.Row]:
    async with aiosqlite.connect(db_path) as connection:
        connection.row_factory = aiosqlite.Row
        cursor = await connection.execute(
            f"""
            SELECT id, symbol, asset_type, predicted_direction, price_at_prediction, metadata
            FROM predictions
            WHERE {where_clause}
              AND created_at < ?
            """,
            (before,),
        )
        rows = await cursor.fetchall()
    return rows


def _record_ml_performance_outcome(
    ensemble: MLEnsemble | None,
    row: aiosqlite.Row,
    *,
    price_change_pct: float,
) -> bool:
    if ensemble is None or not hasattr(ensemble, "_performance"):
        return False

    metadata = _metadata_from_row(row)
    probs = {
        "xgb": metadata.get("xgb_prob"),
        "rf": metadata.get("rf_prob"),
        "lr": metadata.get("lr_prob"),
    }
    if any(value is None for value in probs.values()):
        return False

    parsed_probs: dict[str, float] = {}
    actual_up = price_change_pct > 0
    for model_name, raw_prob in probs.items():
        try:
            parsed_probs[model_name] = float(raw_prob)
        except (TypeError, ValueError):
            return False

    for model_name, prob in parsed_probs.items():
        ensemble._performance.add_outcome(
            symbol=str(row["symbol"]),
            model_name=model_name,
            predicted_up=prob > 0.5,
            actual_up=actual_up,
            prob=prob,
            price_change_pct=price_change_pct,
        )

    return True


def _metadata_from_row(row: aiosqlite.Row) -> dict[str, object]:
    try:
        raw_metadata = row["metadata"]
    except (KeyError, IndexError):
        return {}

    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not isinstance(raw_metadata, str) or not raw_metadata.strip():
        return {}

    try:
        parsed = json.loads(raw_metadata)
    except Exception:
        logger.warning("Ignoring malformed prediction metadata", exc_info=True)
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _update_prediction(db_path: str, prediction_id: int, updates: dict[str, object]) -> None:
    assignments: list[str] = []
    params: list[object] = []
    for column, value in updates.items():
        if value is _SQLITE_NOW:
            assignments.append(f"{column} = datetime('now')")
            continue
        assignments.append(f"{column} = ?")
        params.append(value)

    params.append(prediction_id)

    async with aiosqlite.connect(db_path) as connection:
        await connection.execute(
            f"""
            UPDATE predictions
            SET {", ".join(assignments)}
            WHERE id = ?
            """,
            params,
        )
        await connection.commit()


async def _safe_fetch_current_price(
    price_fetcher: PriceFetcher,
    symbol: str,
    asset_type: str,
    *,
    outcome_label: str,
) -> float | None:
    try:
        current_price = await price_fetcher(symbol, asset_type)
    except Exception:
        logger.warning(
            "Price fetch failed for %s (%s) while recording %s outcome",
            symbol,
            asset_type,
            outcome_label,
            exc_info=True,
        )
        return None

    if current_price is None:
        logger.warning(
            "Could not fetch price for %s (%s); skipping %s outcome",
            symbol,
            asset_type,
            outcome_label,
        )

    return current_price


def _direction_from_change(change_pct: float, *, neutral_band: float) -> str:
    if change_pct > neutral_band:
        return "bullish"
    if change_pct < -neutral_band:
        return "bearish"
    return "neutral"


def _normalize_prediction_direction(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "long": "bullish",
        "buy": "bullish",
        "bullish": "bullish",
        "short": "bearish",
        "sell": "bearish",
        "bearish": "bearish",
        "neutral": "neutral",
    }
    return mapping.get(normalized, "neutral")


def _sqlite_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
