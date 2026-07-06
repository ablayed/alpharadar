from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from app.reporting.ev_metrics import DEFAULT_WINDOW_DAYS, load_prediction_event_rows, query_group_metrics

EXPLOSIVE_MOVE_PCT = 5.0


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
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source") or "unknown")
        outcome_label = _optional_int(row.get("outcome_label"))
        if outcome_label is not None:
            grouped[source].append(outcome_label)

    sources: list[dict[str, Any]] = []
    for source in sorted(grouped):
        labels = grouped[source]
        sample_size = len(labels)
        accuracy = sum(labels) / sample_size if sample_size else None
        inverted_accuracy = (1.0 - accuracy) if accuracy is not None else None
        likely_bug: bool | str
        if sample_size < 30:
            likely_bug = "insufficient_sample"
        else:
            likely_bug = bool(accuracy is not None and abs(accuracy - 0.5) >= 0.15)
        sources.append(
            {
                "source": source,
                "sample_size": sample_size,
                "accuracy": accuracy,
                "inverted_accuracy": inverted_accuracy,
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
    neutral_rows = [row for row in rows if str(row.get("direction") or "").upper() == "NEUTRAL"]
    missed_ids = [
        _optional_int(row.get("id"))
        for row in neutral_rows
        if (move := _optional_float(row.get("actual_move_pct_within_horizon"))) is not None
        and abs(move) >= explosive_move_pct
    ]
    return {
        "check": "neutral_audit",
        "neutral_count": len(neutral_rows),
        "missed_explosive_count": len(missed_ids),
        "missed_ids": missed_ids,
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
                }
                for row in prediction_event_rows
            ]
        duplicate_result = check_duplicates(duplicate_rows)

        if not neutral_rows:
            neutral_result = {
                "check": "neutral_audit",
                "neutral_count": 0,
                "missed_explosive_count": 0,
                "missed_ids": [],
                "explosive_move_pct": EXPLOSIVE_MOVE_PCT,
                "status": "unavailable",
            }
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
