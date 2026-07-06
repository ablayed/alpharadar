from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.audit.verdict import generate

MIGRATION_SQL_PATH = Path(__file__).resolve().parents[2] / "migrations" / "000_phase0_event_tables.sql"


def _create_db(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(MIGRATION_SQL_PATH.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()


def _insert_prediction_event_rows(db_path: Path, rows: list[dict[str, Any]]) -> None:
    connection = sqlite3.connect(db_path)
    try:
        for row in rows:
            connection.execute(
                """
                INSERT INTO prediction_events (
                    ts,
                    symbol,
                    source,
                    direction,
                    raw_probability,
                    confidence_bucket,
                    horizon,
                    entry_price,
                    resolution_price,
                    resolved,
                    outcome_label,
                    gross_return_pct,
                    cost_bps,
                    net_return_pct,
                    migrated_from_v1_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("ts", "2026-05-01 00:00:00"),
                    row.get("symbol", "BTCUSDT"),
                    row.get("source", "briefing"),
                    row.get("direction", "BULLISH"),
                    row.get("raw_probability", 0.65),
                    row.get("confidence_bucket", "0.65_0.70"),
                    row.get("horizon", "1d"),
                    row.get("entry_price", 100.0),
                    row.get("resolution_price", 101.0),
                    row.get("resolved", 1),
                    row.get("outcome_label"),
                    row.get("gross_return_pct"),
                    row.get("cost_bps"),
                    row.get("net_return_pct"),
                    row["migrated_from_v1_id"],
                ),
            )
        connection.commit()
    finally:
        connection.close()


def test_verdict_returns_insufficient_data_for_unresolved_rows_only(tmp_path: Path) -> None:
    db_path = tmp_path / "unresolved_only.db"
    out_path = tmp_path / "audit_verdict.md"
    _create_db(db_path)
    _insert_prediction_event_rows(
        db_path,
        [
            {"migrated_from_v1_id": index, "resolved": 0, "outcome_label": None}
            for index in range(1, 5)
        ],
    )

    result = generate(db_path=str(db_path), out=str(out_path))
    text = out_path.read_text(encoding="utf-8")

    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert "resolved_row_count: 0" in text
    assert "required_minimum: 100" in text
    assert "BUG_DOMINANT / SIGNAL_WEAK / MIXED cannot be assigned yet." in text


def test_verdict_returns_insufficient_data_below_global_floor(tmp_path: Path) -> None:
    db_path = tmp_path / "below_floor.db"
    out_path = tmp_path / "audit_verdict.md"
    _create_db(db_path)
    rows = []
    for index in range(50):
        rows.append(
            {
                "migrated_from_v1_id": index + 1,
                "source": "auto_trade_scan" if index % 2 == 0 else "briefing",
                "horizon": "1d",
                "resolved": 1,
                "outcome_label": 1 if index % 3 else 0,
                "gross_return_pct": 0.8 if index % 2 == 0 else -0.4,
            }
        )
    _insert_prediction_event_rows(db_path, rows)

    result = generate(db_path=str(db_path), out=str(out_path))
    text = out_path.read_text(encoding="utf-8")

    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert "resolved_row_count: 50" in text
    assert "- auto_trade_scan: resolved_rows=25" in text
    assert "- briefing: resolved_rows=25" in text
    assert "- 1d: resolved_rows=50" in text


def test_verdict_marks_thin_source_and_horizon_slices_as_insufficient_sample(tmp_path: Path) -> None:
    db_path = tmp_path / "thin_slice.db"
    out_path = tmp_path / "audit_verdict.md"
    _create_db(db_path)

    rows: list[dict[str, Any]] = []
    for index in range(88):
        rows.append(
            {
                "migrated_from_v1_id": index + 1,
                "source": "auto_trade_scan",
                "horizon": "1d",
                "resolved": 1,
                "outcome_label": 1 if index % 2 == 0 else 0,
                "gross_return_pct": 0.8 if index % 2 == 0 else -0.5,
            }
        )
    for index in range(12):
        rows.append(
            {
                "migrated_from_v1_id": 1000 + index,
                "source": "thin_source",
                "horizon": "3d",
                "resolved": 1,
                "outcome_label": 1 if index < 2 else 0,
                "gross_return_pct": -0.9,
            }
        )
    _insert_prediction_event_rows(db_path, rows)

    result = generate(db_path=str(db_path), out=str(out_path))
    text = out_path.read_text(encoding="utf-8")

    assert result["verdict"] != "INSUFFICIENT_DATA"
    assert "- thin_source: insufficient_sample (resolved_rows=12)" in text
    assert "- 3d: insufficient_sample (resolved_rows=12)" in text
