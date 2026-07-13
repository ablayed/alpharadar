from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.audit.forensic import (
    INVERSION_ACCURACY_DISTANCE,
    INVERSION_MIN_SAMPLE_SIZE,
    check_ordering,
)
from scripts.investigate_auto_trade_scan import (
    HORIZONS,
    LIKELY_BUG_DISTANCE,
    MIN_SLICE_SAMPLE_SIZE,
    NEUTRAL_BAND_BY_HORIZON,
    _answer_questions,
    actual_change_pct_from_gross_return,
    check_ordering_for_source,
    compute_inversion_diagnostics,
    compute_sign_label_consistency,
    expected_outcome_label,
    reconstructed_change_pct,
)
from src.reflection.tracker import _direction_from_change

MIGRATION_SQL_PATH = Path(__file__).resolve().parents[2] / "migrations" / "000_phase0_event_tables.sql"


def _row(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": 1,
        "symbol": "BTCUSDT",
        "source": "auto_trade_scan",
        "direction": "BULLISH",
        "horizon": "1d",
        "entry_price": 100.0,
        "resolution_price": 101.0,
        "gross_return_pct": 1.0,
        "outcome_label": 1,
        "resolved": 1,
        "ts": "2026-01-01 00:00:00",
    }
    base.update(overrides)
    return base


def test_reconstructed_change_pct_basic() -> None:
    row = _row(entry_price=100.0, resolution_price=105.0)
    assert reconstructed_change_pct(row) == pytest.approx(5.0, abs=1e-9)


def test_reconstructed_change_pct_negative_move() -> None:
    row = _row(entry_price=200.0, resolution_price=190.0)
    assert reconstructed_change_pct(row) == pytest.approx(-5.0, abs=1e-9)


def test_reconstructed_change_pct_none_when_entry_missing() -> None:
    row = _row(entry_price=None, resolution_price=105.0)
    assert reconstructed_change_pct(row) is None


def test_actual_change_pct_from_gross_return_undoes_bearish_sign_flip() -> None:
    bullish_row = _row(direction="BULLISH", gross_return_pct=2.5)
    bearish_row = _row(direction="BEARISH", gross_return_pct=2.5)
    assert actual_change_pct_from_gross_return(bullish_row) == pytest.approx(2.5)
    # BEARISH gross_return_pct is stored as -actual_change_pct (see migrate_v1_predictions),
    # so recovering actual_change_pct requires negating it back.
    assert actual_change_pct_from_gross_return(bearish_row) == pytest.approx(-2.5)


def test_expected_outcome_label_bullish_correct_1d() -> None:
    # 1d neutral_band is 0.5; a +1.0% move clears it, so BULLISH should be a win.
    row = _row(direction="BULLISH", horizon="1d")
    assert NEUTRAL_BAND_BY_HORIZON["1d"] == 0.5
    assert expected_outcome_label(row, change_pct=1.0) == 1


def test_expected_outcome_label_bullish_inside_neutral_band_is_not_a_mismatch() -> None:
    # A +0.3% move at the 1d horizon (neutral_band=0.5) falls inside the neutral band,
    # so _direction_from_change correctly returns "neutral" -- a BULLISH prediction
    # against that is correctly a loss (label 0), not a labeling mismatch.
    row = _row(direction="BULLISH", horizon="1d")
    actual_direction = _direction_from_change(0.3, neutral_band=0.5)
    assert actual_direction == "neutral"
    assert expected_outcome_label(row, change_pct=0.3) == 0


def test_expected_outcome_label_bearish_correct_5d_wide_band() -> None:
    # 5d neutral_band is 2.0; a -3.0% move clears it as bearish.
    row = _row(direction="BEARISH", horizon="5d")
    assert NEUTRAL_BAND_BY_HORIZON["5d"] == 2.0
    assert expected_outcome_label(row, change_pct=-3.0) == 1


def test_expected_outcome_label_bearish_inside_wide_neutral_band() -> None:
    # A -1.0% move at 5d (neutral_band=2.0) is inside the band -> neutral, not bearish.
    row = _row(direction="BEARISH", horizon="5d")
    assert _direction_from_change(-1.0, neutral_band=2.0) == "neutral"
    assert expected_outcome_label(row, change_pct=-1.0) == 0


def test_expected_outcome_label_none_for_unknown_horizon() -> None:
    row = _row(direction="BULLISH", horizon="9d")
    assert expected_outcome_label(row, change_pct=5.0) is None


def test_mismatch_detection_flags_disagreeing_stored_label() -> None:
    # Stored outcome_label says win (1), but the reconstructed change_pct (0.3%) falls
    # inside the 1d neutral band, so the real label should be 0 -- a genuine mismatch.
    row = _row(direction="BULLISH", horizon="1d", gross_return_pct=0.3, outcome_label=1)
    result = compute_sign_label_consistency([row])
    assert result["checked_row_count"] == 1
    assert result["mismatch_count"] == 1
    assert result["examples"][0]["id"] == row["id"]


def test_mismatch_detection_does_not_flag_agreeing_stored_label() -> None:
    row = _row(direction="BULLISH", horizon="1d", gross_return_pct=1.5, outcome_label=1)
    result = compute_sign_label_consistency([row])
    assert result["mismatch_count"] == 0


def test_inversion_diagnostic_output_has_no_should_invert_field() -> None:
    rows = [
        _row(id=index, direction="BULLISH", horizon="1d", gross_return_pct=1.0, outcome_label=int(index % 2 == 0))
        for index in range(40)
    ]
    result = compute_inversion_diagnostics(rows)
    for item in result["slices"]:
        assert "should_invert" not in item
    assert set(result["slices"][0]) == {
        "horizon",
        "direction",
        "sample_size",
        "original_accuracy",
        "inverted_accuracy",
        "original_average_return_pct",
        "inverted_average_return_pct",
        "pattern_note",
    }


def test_inversion_diagnostic_covers_all_horizons_and_directions() -> None:
    result = compute_inversion_diagnostics([])
    pairs = {(item["horizon"], item["direction"]) for item in result["slices"]}
    expected_pairs = {(horizon, direction) for horizon in HORIZONS for direction in ("BULLISH", "BEARISH")}
    assert pairs == expected_pairs


def test_neutral_rows_excluded_from_directional_diagnostics() -> None:
    from scripts.investigate_auto_trade_scan import compute_directional_diagnostics

    directional_rows = [
        _row(id=index, direction="BULLISH", horizon="1d", outcome_label=1) for index in range(10)
    ]
    neutral_rows = [
        _row(id=100 + index, direction="NEUTRAL", horizon="1d", outcome_label=0) for index in range(50)
    ]
    diagnostics = compute_directional_diagnostics(directional_rows + neutral_rows)
    assert diagnostics["overall"]["sample_size"] == 10
    assert diagnostics["overall"]["accuracy"] == pytest.approx(1.0)


def _create_ordering_table_with_source(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE forensic_ordering_rows (
            id INTEGER PRIMARY KEY,
            source TEXT,
            prediction_ts TEXT,
            entry_open_time TEXT,
            outcome_open_time TEXT
        )
        """
    )
    rows = [
        (1, "auto_trade_scan", "2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-02 00:00:00"),
        (2, "auto_trade_scan", "2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-02 00:00:00"),
        (3, "briefing", "2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-02 00:00:00"),
    ]
    connection.executemany(
        "INSERT INTO forensic_ordering_rows (id, source, prediction_ts, entry_open_time, outcome_open_time) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()


def test_check_ordering_for_source_matches_full_dataset_zero_violations(tmp_path: Path) -> None:
    db_path = tmp_path / "ordering.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(MIGRATION_SQL_PATH.read_text(encoding="utf-8"))
        _create_ordering_table_with_source(connection)
    finally:
        connection.close()

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        all_rows = [dict(row) for row in connection.execute("SELECT * FROM forensic_ordering_rows").fetchall()]
    finally:
        connection.close()

    full_result = check_ordering(all_rows)
    assert len(full_result["violations"]) == 0

    result = check_ordering_for_source(str(db_path), "auto_trade_scan")
    assert result["status"] == "ok"
    assert result["full_dataset_violation_count"] == 0
    assert result["source_violation_count"] == 0
    assert result["source_violation_count"] == len(full_result["violations"])


def test_check_ordering_for_source_unavailable_when_table_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "no_ordering.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(MIGRATION_SQL_PATH.read_text(encoding="utf-8"))
    finally:
        connection.close()

    result = check_ordering_for_source(str(db_path), "auto_trade_scan")
    assert result["status"] == "unavailable_no_ordering_table"
    assert result["result"] is None


def test_thresholds_are_reused_from_forensic_module_not_reinvented() -> None:
    assert LIKELY_BUG_DISTANCE == INVERSION_ACCURACY_DISTANCE == pytest.approx(0.15)
    assert MIN_SLICE_SAMPLE_SIZE == INVERSION_MIN_SAMPLE_SIZE == 30


def test_inversion_pattern_note_uses_large_deviation_only_label() -> None:
    rows = [
        _row(id=index, direction="BULLISH", horizon="1d", gross_return_pct=1.0, outcome_label=int(index < 5))
        for index in range(40)
    ]
    result = compute_inversion_diagnostics(rows)
    slice_1d_bullish = next(
        item for item in result["slices"] if item["horizon"] == "1d" and item["direction"] == "BULLISH"
    )
    # 5/40 = 12.5% accuracy -> |0.125 - 0.5| = 0.375 >= 0.15 -> flagged, using the new label.
    assert slice_1d_bullish["pattern_note"] == "large_deviation_only"
    assert "possible_wiring_or_label_bug" not in {item["pattern_note"] for item in result["slices"]}


def _clean_ordering() -> dict[str, Any]:
    return {"status": "unavailable_no_ordering_table", "result": None, "note": "n/a"}


def _sign_label(*, checked: int, mismatches: int) -> dict[str, Any]:
    return {
        "checked_row_count": checked,
        "mismatch_count": mismatches,
        "mismatch_rate": (mismatches / checked) if checked else None,
        "examples": [],
    }


def _price_consistency(*, checked: int, violations: int) -> dict[str, Any]:
    return {
        "checked_row_count": checked,
        "violation_count": violations,
        "violation_rate": (violations / checked) if checked else None,
        "violations": [],
    }


def _overall(*, sample_size: int, accuracy: float = 0.5) -> dict[str, Any]:
    return {"sample_size": sample_size, "accuracy": accuracy}


def test_answer_questions_flags_bug_when_rate_above_threshold_and_sample_large_enough() -> None:
    # 16% mismatch rate, n=100 (>= MIN_SLICE_SAMPLE_SIZE) -> above the 15% materiality bar -> YES.
    answers = _answer_questions(
        overall=_overall(sample_size=100),
        sign_label_consistency=_sign_label(checked=100, mismatches=16),
        price_consistency=_price_consistency(checked=100, violations=0),
        ordering=_clean_ordering(),
    )
    assert answers["q1"] == "YES"
    assert answers["q2"] == "label inversion"


def test_answer_questions_does_not_flag_bug_when_sample_too_small() -> None:
    # Same 16%-ish rate, but checked_row_count (10) is below MIN_SLICE_SAMPLE_SIZE (30) -- too small
    # a sample for the rate to be treated as material, regardless of how extreme it looks.
    answers = _answer_questions(
        overall=_overall(sample_size=100),
        sign_label_consistency=_sign_label(checked=10, mismatches=2),
        price_consistency=_price_consistency(checked=100, violations=0),
        ordering=_clean_ordering(),
    )
    assert answers["q1"] == "NO"


def test_answer_questions_does_not_flag_bug_when_rate_at_or_below_threshold() -> None:
    # Exactly 15% (not strictly greater than LIKELY_BUG_DISTANCE) with a large sample (1000) --
    # even at scale, a rate at the threshold boundary is not treated as bug evidence.
    answers = _answer_questions(
        overall=_overall(sample_size=1000),
        sign_label_consistency=_sign_label(checked=1000, mismatches=150),
        price_consistency=_price_consistency(checked=1000, violations=0),
        ordering=_clean_ordering(),
    )
    assert answers["q1"] == "NO"


def test_answer_questions_price_consistency_flags_bug_when_rate_above_threshold() -> None:
    answers = _answer_questions(
        overall=_overall(sample_size=100),
        sign_label_consistency=_sign_label(checked=100, mismatches=0),
        price_consistency=_price_consistency(checked=100, violations=20),
        ordering=_clean_ordering(),
    )
    assert answers["q1"] == "YES"
    assert answers["q2"] == "price reconstruction mismatch"


def test_answer_questions_price_consistency_low_rate_is_footnote_not_bug() -> None:
    # 48/3669 ~= 1.3% -- well below the 15% materiality bar, so it should NOT produce YES.
    answers = _answer_questions(
        overall=_overall(sample_size=3669),
        sign_label_consistency=_sign_label(checked=3669, mismatches=0),
        price_consistency=_price_consistency(checked=3669, violations=48),
        ordering=_clean_ordering(),
    )
    assert answers["q1"] == "NO"
    assert "1.3%" in answers["q3"] or "1.3" in answers["q3"]
