from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.audit.forensic import (
    check_briefing_isolation,
    check_duplicates,
    check_inversion_by_source,
    check_neutral_audit,
    check_ordering,
    check_price_source,
    check_survivorship,
)

FORENSIC_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "forensic_checks_fixture.json"
EV_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "prediction_events_ev_fixture.json"


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ordering_check_flags_bad_row() -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)
    result = check_ordering(fixture["ordering_check_rows"])
    rows_by_id = {row["id"]: row for row in result["rows"]}
    assert rows_by_id[9001]["violation"] is True
    assert rows_by_id[9000]["violation"] is False
    assert [row["id"] for row in result["violations"]] == [9001]


def test_inversion_check_flags_likely_bug() -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)
    result = check_inversion_by_source(fixture["briefing_test_rows"])
    source_result = result["sources"][0]
    assert set(source_result) == {"source", "sample_size", "accuracy", "inverted_accuracy", "likely_bug"}
    assert source_result["sample_size"] == 40
    assert source_result["accuracy"] == pytest.approx(0.25, abs=1e-6)
    assert source_result["likely_bug"] is True


def test_inversion_check_insufficient_sample() -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)
    result = check_inversion_by_source(fixture["new_source_test_rows"])
    source_result = result["sources"][0]
    assert source_result["sample_size"] == 12
    assert source_result["accuracy"] == pytest.approx(2 / 12, abs=1e-6)
    assert source_result["likely_bug"] == "insufficient_sample"


def test_price_source_check_a() -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)
    result = check_price_source(fixture["price_source_rows"])
    assert len(result["violations"]) == 1
    violation = result["violations"][0]
    assert violation["id"] == 9002
    assert violation["violations"][0]["check"] == "A"
    assert violation["violations"][0]["delta_pct"] == pytest.approx(0.019996000799840034, abs=1e-12)


def test_price_source_check_b_passes() -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)
    result = check_price_source(fixture["price_source_rows"])
    row = result["rows"][0]
    assert row["check_b_flagged"] is False


def test_briefing_isolation() -> None:
    fixture = _load_json(EV_FIXTURE_PATH)
    assert isinstance(fixture, list)
    result = check_briefing_isolation(fixture)
    assert result["sample_size"] == 3
    assert result["accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["net_ev_pct"] == pytest.approx(-0.9, abs=1e-6)


def test_neutral_audit_counts_missed_moves() -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)
    result = check_neutral_audit(fixture["neutral_audit_rows"])
    assert result["neutral_count"] == 4
    assert result["missed_explosive_count"] == 2


def test_duplicate_check_finds_exact_pair() -> None:
    fixture = _load_json(FORENSIC_FIXTURE_PATH)
    assert isinstance(fixture, dict)
    result = check_duplicates(fixture["duplicate_rows"])
    assert result["duplicate_pair_count"] == 1
    assert result["duplicate_pairs"][0]["ids"] == [9003, 9004]


def test_survivorship_degraded_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "survivorship.db"
    connection = sqlite3.connect(db_path)
    try:
        result = check_survivorship(connection)
    finally:
        connection.close()
    assert result["finding"] == "degraded: universe_snapshots not yet available"
    assert result["status"] == "degraded"
