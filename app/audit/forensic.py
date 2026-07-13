from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from statistics import median
from typing import Any

from app.reporting.ev_metrics import DEFAULT_WINDOW_DAYS, load_prediction_event_rows, query_group_metrics

EXPLOSIVE_MOVE_PCT = 5.0
DIRECTION_BULLISH = "BULLISH"
DIRECTION_BEARISH = "BEARISH"
DIRECTION_NEUTRAL = "NEUTRAL"
DIRECTIONAL_DIRECTIONS = frozenset({DIRECTION_BULLISH, DIRECTION_BEARISH})
INVERSION_MIN_SAMPLE_SIZE = 30
INVERSION_ACCURACY_DISTANCE = 0.15


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for parser in (datetime.fromisoformat,):
        try:
            parsed = parser(text)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def check_ordering(rows: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for row in rows:
        prediction_ts = _parse_timestamp(row.get("prediction_ts") or row.get("ts"))
        entry_open_time = _parse_timestamp(row.get("entry_open_time"))
        outcome_open_time = _parse_timestamp(row.get("outcome_open_time"))
        violation = not (
            prediction_ts is not None
            and entry_open_time is not None
            and outcome_open_time is not None
            and prediction_ts < entry_open_time < outcome_open_time
        )
        finding = {
            "id": _optional_int(row.get("id")),
            "violation": violation,
            "prediction_ts": row.get("prediction_ts") or row.get("ts"),
            "entry_open_time": row.get("entry_open_time"),
            "outcome_open_time": row.get("outcome_open_time"),
        }
        findings.append(finding)
        if violation:
            violations.append(finding)

    return {
        "check": "ordering",
        "status": "flagged" if violations else "ok",
        "rows": findings,
        "violations": violations,
    }


def check_inversion_by_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source") or "unknown")
        grouped[source].append(row)

    sources: list[dict[str, Any]] = []
    for source in sorted(grouped):
        source_rows = grouped[source]
        blended_labels = [
            label
            for row in source_rows
            if (label := _optional_int(row.get("outcome_label"))) is not None
        ]
        directional_labels = [
            label
            for row in source_rows
            if str(row.get("direction") or "").upper() in DIRECTIONAL_DIRECTIONS
            and (label := _optional_int(row.get("outcome_label"))) is not None
        ]
        directional_sample_size = len(directional_labels)
        directional_accuracy = (
            sum(directional_labels) / directional_sample_size if directional_sample_size else None
        )
        directional_inverted_accuracy = (
            1.0 - directional_accuracy if directional_accuracy is not None else None
        )
        blended_sample_size = len(blended_labels)
        blended_accuracy = sum(blended_labels) / blended_sample_size if blended_sample_size else None
        likely_bug: bool | str
        if directional_sample_size < INVERSION_MIN_SAMPLE_SIZE:
            likely_bug = "insufficient_sample"
        else:
            likely_bug = bool(
                directional_accuracy is not None
                and abs(directional_accuracy - 0.5) >= INVERSION_ACCURACY_DISTANCE
            )
        sources.append(
            {
                "source": source,
                "directional_sample_size": directional_sample_size,
                "directional_accuracy": directional_accuracy,
                "directional_inverted_accuracy": directional_inverted_accuracy,
                "blended_sample_size": blended_sample_size,
                "blended_accuracy": blended_accuracy,
                "likely_bug": likely_bug,
            }
        )

    return {"check": "inversion", "sources": sources}


def check_price_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    threshold = 0.0001
    violations: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    for row in rows:
        prediction_price = _optional_float(row.get("prediction_price"))
        stored_prediction_price = _optional_float(row.get("stored_prediction_price_from_candle"))
        resolution_price = _optional_float(row.get("resolution_price"))
        candle_close_price = _optional_float(row.get("candle_close_price"))

        delta_a = None
        if prediction_price is not None and stored_prediction_price not in (None, 0.0):
            delta_a = abs(prediction_price - stored_prediction_price) / stored_prediction_price

        delta_b = None
        if resolution_price is not None and candle_close_price not in (None, 0.0):
            delta_b = abs(resolution_price - candle_close_price) / candle_close_price

        row_flags: list[dict[str, Any]] = []
        if delta_a is not None and delta_a > threshold:
            row_flags.append({"check": "A", "delta": delta_a, "delta_pct": delta_a * 100.0})
        if delta_b is not None and delta_b > threshold:
            row_flags.append({"check": "B", "delta": delta_b, "delta_pct": delta_b * 100.0})

        finding = {
            "id": _optional_int(row.get("id")),
            "check_a_flagged": any(flag["check"] == "A" for flag in row_flags),
            "check_b_flagged": any(flag["check"] == "B" for flag in row_flags),
            "violations": row_flags,
        }
        inspected.append(finding)
        if row_flags:
            violations.append({"id": finding["id"], "violations": row_flags})

    return {
        "check": "price_source",
        "status": "flagged" if violations else "ok",
        "rows": inspected,
        "violations": violations,
    }


def check_briefing_isolation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    briefing_rows = [row for row in rows if str(row.get("source") or "") == "briefing"]
    sample_size = len([row for row in briefing_rows if _optional_int(row.get("outcome_label")) is not None])
    win_count = sum(1 for row in briefing_rows if _optional_int(row.get("outcome_label")) == 1)
    net_returns = [
        _optional_float(row.get("net_return_pct"))
        for row in briefing_rows
        if row.get("net_return_pct") is not None
    ]
    if not net_returns:
        from app.reporting.ev_metrics import summarize_rows

        summary = summarize_rows(briefing_rows)
        net_ev = summary["net_ev_pct"]
    else:
        net_ev = sum(value for value in net_returns if value is not None) / len(net_returns)
    accuracy = (win_count / sample_size) if sample_size else None
    return {
        "check": "briefing_isolation",
        "source": "briefing",
        "sample_size": sample_size,
        "accuracy": accuracy,
        "net_ev_pct": net_ev,
    }


def check_neutral_audit(
    rows: list[dict[str, Any]],
    *,
    explosive_move_pct: float = EXPLOSIVE_MOVE_PCT,
) -> dict[str, Any]:
    neutral_rows = [
        row for row in rows if str(row.get("direction") or "").upper() == DIRECTION_NEUTRAL
    ]
    absolute_moves: list[float] = []
    missed_ids: list[int | None] = []
    for row in neutral_rows:
        move = _optional_float(row.get("actual_move_pct_within_horizon"))
        if move is None:
            entry_price = _optional_float(row.get("entry_price"))
            resolution_price = _optional_float(row.get("resolution_price"))
            if entry_price not in (None, 0.0) and resolution_price is not None:
                move = ((resolution_price - entry_price) / entry_price) * 100.0
        if move is None:
            continue
        absolute_move = abs(move)
        absolute_moves.append(absolute_move)
        if absolute_move >= explosive_move_pct:
            missed_ids.append(_optional_int(row.get("id")))

    missed_explosive_count = len(missed_ids)
    neutral_count = len(neutral_rows)
    return {
        "check": "neutral_audit",
        "neutral_count": neutral_count,
        "missed_explosive_count": missed_explosive_count,
        "missed_explosive_rate": missed_explosive_count / neutral_count if neutral_count else None,
        "missed_ids": missed_ids,
        "average_absolute_future_move": (
            sum(absolute_moves) / len(absolute_moves) if absolute_moves else None
        ),
        "median_absolute_future_move": median(absolute_moves) if absolute_moves else None,
        "explosive_move_pct": explosive_move_pct,
    }


def check_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("symbol"),
            row.get("source"),
            row.get("prediction_ts") or row.get("ts"),
            row.get("raw_probability"),
            row.get("horizon"),
        )
        row_id = _optional_int(row.get("id"))
        if row_id is not None:
            grouped[key].append(row_id)

    duplicate_pairs = [
        {
            "key": {
                "symbol": key[0],
                "source": key[1],
                "prediction_ts": key[2],
                "raw_probability": key[3],
                "horizon": key[4],
            },
            "ids": ids,
        }
        for key, ids in grouped.items()
        if len(ids) > 1
    ]

    return {
        "check": "duplicates",
        "duplicate_pair_count": len(duplicate_pairs),
        "duplicate_pairs": duplicate_pairs,
    }


def check_survivorship(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'universe_snapshots'"
    ).fetchone()
    if row is None:
        return {
            "check": "survivorship",
            "status": "degraded",
            "finding": "degraded: universe_snapshots not yet available",
        }

    return {
        "check": "survivorship",
        "status": "ok",
        "finding": "universe_snapshots available",
    }


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _load_rows(connection: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    if not _table_exists(connection, table_name):
        return []
    rows = connection.execute(f"SELECT * FROM {table_name} ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def run_forensic_audit(
    db_path: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict[str, Any]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        prediction_event_rows = load_prediction_event_rows(db_path, window_days=window_days)
        ordering_rows = _load_rows(connection, "forensic_ordering_rows")
        price_source_rows = _load_rows(connection, "forensic_price_source_rows")
        duplicate_rows = _load_rows(connection, "forensic_duplicate_rows")
        neutral_rows = _load_rows(connection, "forensic_neutral_rows")

        if not ordering_rows:
            ordering_result = {
                "check": "ordering",
                "status": "unavailable",
                "rows": [],
                "violations": [],
            }
        else:
            ordering_result = check_ordering(ordering_rows)

        if not price_source_rows:
            price_source_result = {
                "check": "price_source",
                "status": "unavailable",
                "rows": [],
                "violations": [],
            }
        else:
            price_source_result = check_price_source(price_source_rows)

        if not duplicate_rows:
            duplicate_rows = [
                {
                    "id": row.get("id"),
                    "symbol": row.get("symbol"),
                    "source": row.get("source"),
                    "ts": row.get("ts"),
                    "raw_probability": row.get("raw_probability"),
                    "horizon": row.get("horizon"),
                }
                for row in prediction_event_rows
            ]
        duplicate_result = check_duplicates(duplicate_rows)

        if not neutral_rows:
            neutral_result = check_neutral_audit(prediction_event_rows)
            neutral_result["status"] = "derived_from_prediction_events"
        else:
            neutral_result = check_neutral_audit(neutral_rows)
            neutral_result["status"] = "ok"

        return {
            "window_days": window_days,
            "checks": {
                "ordering": ordering_result,
                "inversion": check_inversion_by_source(prediction_event_rows),
                "price_source": price_source_result,
                "briefing_isolation": check_briefing_isolation(prediction_event_rows),
                "neutral_audit": neutral_result,
                "duplicates": duplicate_result,
                "source_ev": {
                    "check": "source_ev",
                    "groups": query_group_metrics(db_path, group_by="source", window_days=window_days),
                },
                "survivorship": check_survivorship(connection),
            },
        }
    finally:
        connection.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Block A forensic audit checks.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Trailing window in days.")
    return parser


def main(argv: list[str] | None = None, *, db_path: str | None = None, window_days: int = DEFAULT_WINDOW_DAYS) -> int:
    if db_path is None:
        args = _build_arg_parser().parse_args(argv)
        db_path = args.db_path
        window_days = args.window_days

    findings = run_forensic_audit(db_path, window_days=window_days)
    print(json.dumps(findings, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
