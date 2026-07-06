from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.migrate_v1_predictions import migrate_v1_predictions


def _build_legacy_predictions_db(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                event_type TEXT,
                predicted_direction TEXT NOT NULL,
                predicted_target REAL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                source TEXT NOT NULL,
                price_at_prediction REAL NOT NULL,
                outcome_recorded INTEGER NOT NULL DEFAULT 0,
                actual_price_1d REAL,
                actual_price_5d REAL,
                actual_direction TEXT,
                actual_change_1d_pct REAL,
                actual_change_5d_pct REAL,
                is_correct_1d INTEGER,
                is_correct_5d INTEGER,
                outcome_recorded_at TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO predictions (
                created_at,
                symbol,
                asset_type,
                event_type,
                predicted_direction,
                predicted_target,
                confidence,
                reasoning,
                source,
                price_at_prediction,
                outcome_recorded,
                actual_price_1d,
                actual_price_5d,
                actual_direction,
                actual_change_1d_pct,
                actual_change_5d_pct,
                is_correct_1d,
                is_correct_5d,
                outcome_recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "2026-05-01 00:00:00",
                    "BTCUSDT",
                    "crypto",
                    None,
                    "bullish",
                    None,
                    0.68,
                    "Resolved legacy row",
                    "briefing",
                    100.0,
                    1,
                    103.0,
                    104.0,
                    "bullish",
                    3.0,
                    4.0,
                    1,
                    1,
                    "2026-05-02 00:00:00",
                ),
                (
                    "2026-05-02 00:00:00",
                    "ETHUSDT",
                    "crypto",
                    None,
                    "bearish",
                    None,
                    0.57,
                    "Unresolved legacy row",
                    "auto_trade_scan",
                    200.0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_migrate_v1_predictions_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "migration_idempotency.db"
    _build_legacy_predictions_db(db_path)

    first_result = migrate_v1_predictions(str(db_path))
    assert first_result.inserted_rows == 2

    connection = sqlite3.connect(db_path)
    try:
        first_count = connection.execute("SELECT COUNT(*) FROM prediction_events").fetchone()[0]
    finally:
        connection.close()

    second_result = migrate_v1_predictions(str(db_path))
    assert second_result.inserted_rows == 0

    connection = sqlite3.connect(db_path)
    try:
        second_count = connection.execute("SELECT COUNT(*) FROM prediction_events").fetchone()[0]
    finally:
        connection.close()

    assert first_count == second_count == 2
