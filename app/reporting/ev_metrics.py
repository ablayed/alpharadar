from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.audit.cost_model import resolve_round_trip_cost_bps

DEFAULT_WINDOW_DAYS = 90
SUPPORTED_GROUP_BYS = ("source", "confidence_bucket", "symbol", "regime")


@dataclass(frozen=True, slots=True)
class GroupMetric:
    group_by: str
    group_value: str
    sample_size: int
    win_count: int
    accuracy: float | None
    gross_ev_pct: float | None
    net_ev_pct: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_by": self.group_by,
            "group_value": self.group_value,
            "sample_size": self.sample_size,
            "win_count": self.win_count,
            "accuracy": self.accuracy,
            "gross_ev_pct": self.gross_ev_pct,
            "net_ev_pct": self.net_ev_pct,
        }


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


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


def _group_value(row: dict[str, Any], group_by: str) -> str:
    value = row.get(group_by)
    return "unknown" if value in (None, "") else str(value)


def _net_return_pct(
    row: dict[str, Any],
    *,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
) -> float | None:
    explicit_net = _optional_float(row.get("net_return_pct"))
    if explicit_net is not None:
        return explicit_net

    gross_return_pct = _optional_float(row.get("gross_return_pct"))
    if gross_return_pct is None:
        return None

    row_cost_bps = _optional_float(row.get("cost_bps"))
    if row_cost_bps is None:
        row_cost_bps = resolve_round_trip_cost_bps(fee_bps=fee_bps, slippage_bps=slippage_bps)
    return gross_return_pct - (row_cost_bps / 100.0)


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
) -> dict[str, Any]:
    sample_size = 0
    win_count = 0
    gross_returns: list[float] = []
    net_returns: list[float] = []

    for row in rows:
        outcome_label = _optional_int(row.get("outcome_label"))
        if outcome_label is not None:
            sample_size += 1
            if outcome_label == 1:
                win_count += 1

        gross_return_pct = _optional_float(row.get("gross_return_pct"))
        if gross_return_pct is not None:
            gross_returns.append(gross_return_pct)

        net_return_pct = _net_return_pct(row, fee_bps=fee_bps, slippage_bps=slippage_bps)
        if net_return_pct is not None:
            net_returns.append(net_return_pct)

    accuracy = (win_count / sample_size) if sample_size else None
    gross_ev_pct = (sum(gross_returns) / len(gross_returns)) if gross_returns else None
    net_ev_pct = (sum(net_returns) / len(net_returns)) if net_returns else None
    return {
        "sample_size": sample_size,
        "win_count": win_count,
        "accuracy": accuracy,
        "gross_ev_pct": gross_ev_pct,
        "net_ev_pct": net_ev_pct,
    }


def compute_group_metrics(
    rows: list[dict[str, Any]],
    *,
    group_by: str,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
) -> list[dict[str, Any]]:
    if group_by not in SUPPORTED_GROUP_BYS:
        raise ValueError(f"Unsupported group_by value: {group_by}")

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[_group_value(row, group_by)].append(row)

    metrics: list[dict[str, Any]] = []
    for group_value in sorted(grouped_rows):
        summary = summarize_rows(
            grouped_rows[group_value],
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        metrics.append(
            GroupMetric(
                group_by=group_by,
                group_value=group_value,
                sample_size=int(summary["sample_size"]),
                win_count=int(summary["win_count"]),
                accuracy=summary["accuracy"],
                gross_ev_pct=summary["gross_ev_pct"],
                net_ev_pct=summary["net_ev_pct"],
            ).as_dict()
        )
    return metrics


def load_prediction_event_rows(
    db_path: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        if not _table_exists(connection, "prediction_events"):
            return []

        columns = _table_columns(connection, "prediction_events")
        select_parts = [
            "id",
            "ts",
            "symbol",
            "source",
            "direction",
            "raw_probability",
            "confidence_bucket",
            "horizon",
            "entry_price",
            "resolution_price",
            "resolved",
            "outcome_label",
            "gross_return_pct",
            "cost_bps",
            "net_return_pct",
        ]
        if "regime" in columns:
            select_parts.append("regime")
        else:
            select_parts.append("NULL AS regime")

        query = f"SELECT {', '.join(select_parts)} FROM prediction_events"
        params: list[Any] = []
        if "ts" in columns:
            query += " WHERE datetime(ts) >= datetime('now', ?)"
            params.append(f"-{int(window_days)} days")
        query += " ORDER BY id"
        rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def query_group_metrics(
    db_path: str,
    *,
    group_by: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
) -> list[dict[str, Any]]:
    rows = load_prediction_event_rows(db_path, window_days=window_days)
    return compute_group_metrics(
        rows,
        group_by=group_by,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute net EV and directional accuracy for prediction_events.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--group-by", required=True, choices=SUPPORTED_GROUP_BYS, help="Grouping column.")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Trailing window in days.")
    return parser


def main(argv: list[str] | None = None, *, db_path: str | None = None, group_by: str | None = None, window_days: int = DEFAULT_WINDOW_DAYS) -> int:
    if db_path is None or group_by is None:
        args = _build_arg_parser().parse_args(argv)
        db_path = args.db_path
        group_by = args.group_by
        window_days = args.window_days

    metrics = query_group_metrics(
        db_path,
        group_by=group_by,
        window_days=window_days,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
