from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.audit.verdict import generate

EV_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "prediction_events_ev_fixture.json"
FORENSIC_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "forensic_checks_fixture.json"
MIGRATION_SQL_PATH = Path(__file__).resolve().parents[2] / "migrations" / "000_phase0_event_tables.sql"


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expanded_ev_rows() -> list[dict[str, Any]]:
    base_rows = _load_json(EV_FIXTURE_PATH)
    assert isinstance(base_rows, list)

    # Repeat the exact 6-row EV fixture 17 times total (1 base + 16 filler copies).
    # Each copy preserves the original 3 auto_trade_scan rows and 3 briefing rows unchanged,
    # so per-source gross/net EV means stay invariant while the resolved sample grows past 100.
    expanded_rows: list[dict[str, Any]] = []
    for copy_index in range(17):
        id_offset = copy_index * 1000
        for row in base_rows:
            expanded_rows.append({**row, "id": int(row["id"]) + id_offset})
    return expanded_rows


def _create_support_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE forensic_ordering_rows (
            id INTEGER PRIMARY KEY,
            prediction_ts TEXT,
            entry_open_time TEXT,
            outcome_open_time TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE forensic_price_source_rows (
            id INTEGER PRIMARY KEY,
            prediction_price REAL,
            stored_prediction_price_from_candle REAL,
            resolution_price REAL,
            candle_close_price REAL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE forensic_duplicate_rows (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            source TEXT,
            prediction_ts TEXT,
            raw_probability REAL,
            entry_price REAL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE forensic_neutral_rows (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            direction TEXT,
            horizon TEXT,
            actual_move_pct_within_horizon REAL
        )
        """
    )


def _load_prediction_events(connection: sqlite3.Connection) -> None:
    ev_rows = _expanded_ev_rows()
    forensic_rows = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(forensic_rows, dict)

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    for row in ev_rows:
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
                model_version,
                feature_snapshot_id,
                migrated_from_v1_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                row["symbol"],
                row["source"],
                "BULLISH",
                0.65,
                row["confidence_bucket"],
                "1d",
                100.0,
                101.0,
                1,
                row["outcome_label"],
                row["gross_return_pct"],
                None,
                None,
                None,
                None,
                row["id"],
            ),
        )

    synthetic_id = 100000
    for source_name in ("briefing_test_rows", "new_source_test_rows"):
        for row in forensic_rows[source_name]:
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
                    model_version,
                    feature_snapshot_id,
                    migrated_from_v1_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    "BTCUSDT",
                    row["source"],
                    "BULLISH",
                    0.55,
                    "0.55_0.60",
                    "1d",
                    100.0,
                    101.0,
                    1,
                    row["outcome_label"],
                    None,
                    None,
                    None,
                    None,
                    None,
                    synthetic_id,
                ),
            )
            synthetic_id += 1


def _load_forensic_support_tables(connection: sqlite3.Connection) -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)

    connection.executemany(
        """
        INSERT INTO forensic_ordering_rows (id, prediction_ts, entry_open_time, outcome_open_time)
        VALUES (:id, :prediction_ts, :entry_open_time, :outcome_open_time)
        """,
        fixture["ordering_check_rows"],
    )
    connection.executemany(
        """
        INSERT INTO forensic_price_source_rows (
            id,
            prediction_price,
            stored_prediction_price_from_candle,
            resolution_price,
            candle_close_price
        ) VALUES (
            :id,
            :prediction_price,
            :stored_prediction_price_from_candle,
            :resolution_price,
            :candle_close_price
        )
        """,
        fixture["price_source_rows"],
    )
    connection.executemany(
        """
        INSERT INTO forensic_duplicate_rows (
            id,
            symbol,
            source,
            prediction_ts,
            raw_probability,
            entry_price
        ) VALUES (
            :id,
            :symbol,
            :source,
            :prediction_ts,
            :raw_probability,
            :entry_price
        )
        """,
        fixture["duplicate_rows"],
    )
    connection.executemany(
        """
        INSERT INTO forensic_neutral_rows (
            id,
            symbol,
            direction,
            horizon,
            actual_move_pct_within_horizon
        ) VALUES (
            :id,
            :symbol,
            :direction,
            :horizon,
            :actual_move_pct_within_horizon
        )
        """,
        fixture["neutral_audit_rows"],
    )


def test_verdict_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "verdict.db"
    out_path = tmp_path / "audit_verdict.md"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(MIGRATION_SQL_PATH.read_text(encoding="utf-8"))
        _create_support_tables(connection)
        _load_prediction_events(connection)
        _load_forensic_support_tables(connection)
        connection.commit()
    finally:
        connection.close()

    result = generate(db_path=str(db_path), out=str(out_path))
    text = out_path.read_text(encoding="utf-8")

    assert result["verdict"] == "MIXED"
    assert text.startswith("# Audit Verdict")
    assert re.findall(r"\b(BUG_DOMINANT|SIGNAL_WEAK|MIXED)\b", text) == ["MIXED"]
    assert "Did top-bucket EV turn positive after costs?" in text
    assert "Did high-confidence become calibrated?" in text
    assert "Did source-level results stop being systematically broken?" in text
    assert "Did auto_trade_scan beat no-trade and random baselines?" in text
    assert "Root Cause Findings (informational, does not override sample-size gate)" in text
