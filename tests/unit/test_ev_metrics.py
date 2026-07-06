from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.reporting.ev_metrics import query_group_metrics

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "prediction_events_ev_fixture.json"


def _load_fixture_rows() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_prediction_events_db(tmp_path: Path, rows: list[dict[str, Any]]) -> str:
    db_path = tmp_path / "ev_metrics.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE prediction_events (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                direction TEXT NOT NULL,
                raw_probability REAL,
                confidence_bucket TEXT,
                horizon TEXT NOT NULL,
                entry_price REAL,
                resolution_price REAL,
                resolved INTEGER DEFAULT 0,
                outcome_label INTEGER,
                gross_return_pct REAL,
                cost_bps REAL,
                net_return_pct REAL,
                regime TEXT
            )
            """
        )
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        for row in rows:
            connection.execute(
                """
                INSERT INTO prediction_events (
                    id,
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
                    regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
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
                    row.get("gross_return_pct"),
                    row.get("cost_bps"),
                    row.get("net_return_pct"),
                    row["regime"],
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return str(db_path)


def _metric_by_group(metrics: list[dict[str, Any]], group_value: str) -> dict[str, Any]:
    for metric in metrics:
        if metric["group_value"] == group_value:
            return metric
    raise AssertionError(f"Missing group {group_value!r} in {metrics!r}")


def test_ev_by_source(tmp_path: Path) -> None:
    db_path = _build_prediction_events_db(tmp_path, _load_fixture_rows())
    metrics = query_group_metrics(db_path, group_by="source")

    assert sum(metric["sample_size"] for metric in metrics) == 6

    auto_trade_scan = _metric_by_group(metrics, "auto_trade_scan")
    briefing = _metric_by_group(metrics, "briefing")
    assert auto_trade_scan["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert auto_trade_scan["net_ev_pct"] == pytest.approx(0.13333333333333333, abs=1e-6)
    assert briefing["accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert briefing["net_ev_pct"] == pytest.approx(-0.9, abs=1e-6)


def test_ev_by_bucket(tmp_path: Path) -> None:
    db_path = _build_prediction_events_db(tmp_path, _load_fixture_rows())
    metrics = query_group_metrics(db_path, group_by="confidence_bucket")
    assert sum(metric["sample_size"] for metric in metrics) == 6
    assert {metric["group_value"] for metric in metrics} == {
        "0.50_0.55",
        "0.55_0.60",
        "0.60_0.65",
        "0.65_0.70",
    }


def test_ev_by_symbol(tmp_path: Path) -> None:
    db_path = _build_prediction_events_db(tmp_path, _load_fixture_rows())
    metrics = query_group_metrics(db_path, group_by="symbol")
    assert sum(metric["sample_size"] for metric in metrics) == 6
    assert {metric["group_value"] for metric in metrics} == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def test_ev_by_regime(tmp_path: Path) -> None:
    db_path = _build_prediction_events_db(tmp_path, _load_fixture_rows())
    metrics = query_group_metrics(db_path, group_by="regime")
    assert sum(metric["sample_size"] for metric in metrics) == 6
    assert {metric["group_value"] for metric in metrics} == {"BULL_TREND", "SIDEWAYS"}


def test_ev_metrics_no_double_cost(tmp_path: Path) -> None:
    rows = _load_fixture_rows()
    rows[0]["cost_bps"] = 30
    rows[0]["net_return_pct"] = 0.7
    db_path = _build_prediction_events_db(tmp_path, rows)
    metrics = query_group_metrics(db_path, group_by="source")
    auto_trade_scan = _metric_by_group(metrics, "auto_trade_scan")
    assert auto_trade_scan["net_ev_pct"] == pytest.approx(0.13333333333333333, abs=1e-6)
