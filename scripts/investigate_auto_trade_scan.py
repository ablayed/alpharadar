from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

from app.audit.forensic import check_ordering
from src.reflection.tracker import _direction_from_change

DEFAULT_SOURCE = "auto_trade_scan"
DEFAULT_LIMIT_SAMPLES = 100
DEFAULT_OUT = "reports/auto_trade_scan_forensic.md"
SAMPLE_CSV_NAME = "auto_trade_scan_sample_rows.csv"

HORIZONS: tuple[str, ...] = ("1d", "3d", "5d", "7d")
NEUTRAL_BAND_BY_HORIZON: dict[str, float] = {"1d": 0.5, "3d": 2.0, "5d": 2.0, "7d": 2.0}
DIRECTIONAL_DIRECTIONS = ("BULLISH", "BEARISH")
LIKELY_BUG_DISTANCE = 0.15
MIN_SLICE_SAMPLE_SIZE = 30
MATERIAL_CHANGE_DIFF_PCT = 0.05
SAMPLE_SEED = 20260713


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


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def load_source_rows(db_path: str, source: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM prediction_events WHERE source = ? ORDER BY id",
            (source,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def is_resolved_row(row: dict[str, Any]) -> bool:
    resolved = row.get("resolved")
    resolved_int = _optional_int(resolved)
    resolved_flag = bool(resolved_int) if resolved_int is not None else bool(resolved)
    return (
        resolved_flag
        and _optional_int(row.get("outcome_label")) is not None
        and _optional_float(row.get("entry_price")) is not None
        and _optional_float(row.get("resolution_price")) is not None
    )


def is_directional(row: dict[str, Any]) -> bool:
    return str(row.get("direction") or "").upper() in DIRECTIONAL_DIRECTIONS


def reconstructed_change_pct(row: dict[str, Any]) -> float | None:
    entry_price = _optional_float(row.get("entry_price"))
    resolution_price = _optional_float(row.get("resolution_price"))
    if entry_price in (None, 0.0) or resolution_price is None:
        return None
    return ((resolution_price - entry_price) / entry_price) * 100.0


def actual_change_pct_from_gross_return(row: dict[str, Any]) -> float | None:
    """Recovers the direction-agnostic price move from the stored (direction-signed)
    gross_return_pct. prediction_events does not persist a raw change_pct column; the
    sign convention applied at migration time (see scripts/migrate_v1_predictions.py
    _gross_return_pct) flips gross_return_pct for BEARISH rows, so this undoes that flip.
    """
    gross_return_pct = _optional_float(row.get("gross_return_pct"))
    direction = str(row.get("direction") or "").upper()
    if gross_return_pct is None:
        return None
    if direction == "BULLISH":
        return gross_return_pct
    if direction == "BEARISH":
        return -gross_return_pct
    return None


def expected_outcome_label(row: dict[str, Any], *, change_pct: float | None) -> int | None:
    direction = str(row.get("direction") or "").upper()
    horizon = str(row.get("horizon") or "")
    neutral_band = NEUTRAL_BAND_BY_HORIZON.get(horizon)
    if neutral_band is None or change_pct is None or direction not in DIRECTIONAL_DIRECTIONS:
        return None
    actual_direction = _direction_from_change(change_pct, neutral_band=neutral_band)
    return 1 if direction.lower() == actual_direction else 0


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [_optional_int(row.get("outcome_label")) for row in rows]
    labels = [label for label in labels if label is not None]
    sample_size = len(labels)
    accuracy = (sum(labels) / sample_size) if sample_size else None
    return {"sample_size": sample_size, "accuracy": accuracy}


def _returns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_optional_float(row.get("gross_return_pct")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {"sample_size": 0, "average_gross_return_pct": None, "median_gross_return_pct": None}
    return {
        "sample_size": len(values),
        "average_gross_return_pct": sum(values) / len(values),
        "median_gross_return_pct": median(values),
    }


def compute_directional_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved_directional = [row for row in rows if is_resolved_row(row) and is_directional(row)]
    overall = _accuracy(resolved_directional)

    by_horizon: dict[str, Any] = {}
    bullish_by_horizon: dict[str, Any] = {}
    bearish_by_horizon: dict[str, Any] = {}
    returns_by_direction_horizon: dict[str, Any] = {}

    for horizon in HORIZONS:
        horizon_rows = [row for row in resolved_directional if str(row.get("horizon")) == horizon]
        by_horizon[horizon] = _accuracy(horizon_rows)

        bullish_rows = [row for row in horizon_rows if str(row.get("direction")).upper() == "BULLISH"]
        bearish_rows = [row for row in horizon_rows if str(row.get("direction")).upper() == "BEARISH"]
        bullish_by_horizon[horizon] = _accuracy(bullish_rows)
        bearish_by_horizon[horizon] = _accuracy(bearish_rows)

        returns_by_direction_horizon[horizon] = {
            "BULLISH": _returns(bullish_rows),
            "BEARISH": _returns(bearish_rows),
        }

    return {
        "overall": overall,
        "by_horizon": by_horizon,
        "bullish_by_horizon": bullish_by_horizon,
        "bearish_by_horizon": bearish_by_horizon,
        "returns_by_direction_horizon": returns_by_direction_horizon,
        "resolved_directional_rows": resolved_directional,
    }


def compute_inversion_diagnostics(resolved_directional_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnostic only: reports what accuracy/return would look like if the direction
    were inverted. Does NOT recommend trading the inverse and does NOT emit a
    should_invert field.
    """
    slices: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for direction in DIRECTIONAL_DIRECTIONS:
            slice_rows = [
                row
                for row in resolved_directional_rows
                if str(row.get("horizon")) == horizon and str(row.get("direction")).upper() == direction
            ]
            accuracy_summary = _accuracy(slice_rows)
            returns_summary = _returns(slice_rows)
            sample_size = accuracy_summary["sample_size"]
            original_accuracy = accuracy_summary["accuracy"]
            original_avg_return = returns_summary["average_gross_return_pct"]
            inverted_accuracy = (1.0 - original_accuracy) if original_accuracy is not None else None
            inverted_avg_return = (-original_avg_return) if original_avg_return is not None else None

            pattern_note = "insufficient_sample"
            if sample_size >= MIN_SLICE_SAMPLE_SIZE and original_accuracy is not None:
                distance = abs(original_accuracy - 0.5)
                pattern_note = (
                    "possible_wiring_or_label_bug"
                    if distance >= LIKELY_BUG_DISTANCE
                    else "no_strong_inversion_signal"
                )

            slices.append(
                {
                    "horizon": horizon,
                    "direction": direction,
                    "sample_size": sample_size,
                    "original_accuracy": original_accuracy,
                    "inverted_accuracy": inverted_accuracy,
                    "original_average_return_pct": original_avg_return,
                    "inverted_average_return_pct": inverted_avg_return,
                    "pattern_note": pattern_note,
                }
            )
    return {"slices": slices}


def check_ordering_for_source(db_path: str, source: str) -> dict[str, Any]:
    """Reuses app.audit.forensic.check_ordering rather than reimplementing ordering
    logic. forensic_ordering_rows is a support table that, in production, is not
    populated with a source/symbol linkage back to prediction_events -- so a
    source-filtered ordering check can only run when that linkage is present.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        if not _table_exists(connection, "forensic_ordering_rows"):
            return {
                "status": "unavailable_no_ordering_table",
                "result": None,
                "note": (
                    "forensic_ordering_rows does not exist in this database. The full-dataset "
                    "audit's '0 ordering violations' figure for this table reflects the same "
                    "absence (status=unavailable), not a verified check against real timestamps. "
                    "prediction_events itself does not store entry_open_time/outcome_open_time, "
                    "so no genuine ordering check can be run against production data today."
                ),
            }

        columns = _table_columns(connection, "forensic_ordering_rows")
        if "source" not in columns:
            return {
                "status": "unavailable_no_source_link",
                "result": None,
                "note": (
                    "forensic_ordering_rows exists but has no source column, so rows cannot be "
                    "filtered to auto_trade_scan specifically."
                ),
            }

        all_rows = [dict(row) for row in connection.execute("SELECT * FROM forensic_ordering_rows").fetchall()]
        source_rows = [row for row in all_rows if str(row.get("source")) == source]
        full_result = check_ordering(all_rows)
        source_result = check_ordering(source_rows)
        return {
            "status": "ok",
            "result": source_result,
            "full_dataset_violation_count": len(full_result["violations"]),
            "source_violation_count": len(source_result["violations"]),
            "note": None,
        }
    finally:
        connection.close()


def compute_sign_label_consistency(resolved_directional_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for row in resolved_directional_rows:
        change_pct = actual_change_pct_from_gross_return(row)
        if change_pct is None:
            continue
        checked += 1
        expected = expected_outcome_label(row, change_pct=change_pct)
        stored = _optional_int(row.get("outcome_label"))
        if expected is not None and stored is not None and expected != stored:
            mismatches.append(
                {
                    "id": row.get("id"),
                    "symbol": row.get("symbol"),
                    "horizon": row.get("horizon"),
                    "direction": row.get("direction"),
                    "actual_change_pct": change_pct,
                    "expected_outcome_label": expected,
                    "stored_outcome_label": stored,
                }
            )
    mismatch_count = len(mismatches)
    return {
        "checked_row_count": checked,
        "mismatch_count": mismatch_count,
        "mismatch_rate": (mismatch_count / checked) if checked else None,
        "examples": mismatches[:20],
    }


def compute_price_consistency(resolved_directional_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked = 0
    for row in resolved_directional_rows:
        actual_change_pct = actual_change_pct_from_gross_return(row)
        reconstructed = reconstructed_change_pct(row)
        if actual_change_pct is None or reconstructed is None:
            continue
        checked += 1
        diff = abs(actual_change_pct - reconstructed)
        expected_from_reconstructed = expected_outcome_label(row, change_pct=reconstructed)
        stored_label = _optional_int(row.get("outcome_label"))
        label_disagrees = (
            expected_from_reconstructed is not None
            and stored_label is not None
            and expected_from_reconstructed != stored_label
        )
        if diff > MATERIAL_CHANGE_DIFF_PCT or label_disagrees:
            violations.append(
                {
                    "id": row.get("id"),
                    "symbol": row.get("symbol"),
                    "entry_price": row.get("entry_price"),
                    "resolution_price": row.get("resolution_price"),
                    "actual_change_pct": actual_change_pct,
                    "reconstructed_change_pct": reconstructed,
                    "diff": diff,
                    "material_diff": diff > MATERIAL_CHANGE_DIFF_PCT,
                    "outcome_label_disagrees_with_reconstructed": label_disagrees,
                }
            )
    return {
        "checked_row_count": checked,
        "violation_count": len(violations),
        "violations": violations[:20],
    }


def _build_sample_row(row: dict[str, Any]) -> dict[str, Any]:
    reconstructed = reconstructed_change_pct(row)
    actual_change_pct = actual_change_pct_from_gross_return(row)
    expected_label = expected_outcome_label(row, change_pct=actual_change_pct)
    stored_label = _optional_int(row.get("outcome_label"))
    horizon = str(row.get("horizon") or "")
    neutral_band = NEUTRAL_BAND_BY_HORIZON.get(horizon)
    mismatch_flag = bool(
        expected_label is not None and stored_label is not None and expected_label != stored_label
    )
    return {
        "id": row.get("id"),
        "symbol": row.get("symbol"),
        "source": row.get("source"),
        "direction": row.get("direction"),
        "horizon": horizon,
        "prediction_ts": row.get("ts"),
        "entry_price": row.get("entry_price"),
        "resolution_price": row.get("resolution_price"),
        "actual_change_pct": actual_change_pct,
        "outcome_label": stored_label,
        "raw_probability": row.get("raw_probability"),
        "confidence_bucket": row.get("confidence_bucket"),
        "reconstructed_change_pct": reconstructed,
        "expected_outcome_label": expected_label,
        "neutral_band_used": neutral_band,
        "mismatch_flag": mismatch_flag,
    }


def build_sample_export(
    resolved_directional_rows: list[dict[str, Any]],
    *,
    limit_samples: int,
) -> list[dict[str, Any]]:
    pool = resolved_directional_rows[:limit_samples]
    rng = random.Random(SAMPLE_SEED)

    wins = [row for row in pool if _optional_int(row.get("outcome_label")) == 1]
    losses = [row for row in pool if _optional_int(row.get("outcome_label")) == 0]
    mismatched_ids: set[int] = set()
    mismatches: list[dict[str, Any]] = []
    for row in pool:
        change_pct = actual_change_pct_from_gross_return(row)
        expected = expected_outcome_label(row, change_pct=change_pct)
        stored = _optional_int(row.get("outcome_label"))
        if expected is not None and stored is not None and expected != stored:
            mismatches.append(row)
            row_id = _optional_int(row.get("id"))
            if row_id is not None:
                mismatched_ids.add(row_id)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def _take(candidates: list[dict[str, Any]], count: int) -> None:
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        remaining = count
        for row in shuffled:
            if remaining <= 0:
                break
            row_id = _optional_int(row.get("id"))
            if row_id in selected_ids:
                continue
            selected.append(row)
            if row_id is not None:
                selected_ids.add(row_id)
            remaining -= 1

    _take(wins, 15)
    _take(losses, 15)
    _take(mismatches, 10)
    remaining = [row for row in pool if _optional_int(row.get("id")) not in selected_ids]
    _take(remaining, 10)

    return [_build_sample_row(row) for row in selected[:50]]


def write_sample_csv(sample_rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "symbol",
        "source",
        "direction",
        "horizon",
        "prediction_ts",
        "entry_price",
        "resolution_price",
        "actual_change_pct",
        "outcome_label",
        "raw_probability",
        "confidence_bucket",
        "reconstructed_change_pct",
        "expected_outcome_label",
        "mismatch_flag",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.1f}%"


def _format_return(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def _answer_questions(
    *,
    overall: dict[str, Any],
    sign_label_consistency: dict[str, Any],
    price_consistency: dict[str, Any],
    ordering: dict[str, Any],
    inversion: dict[str, Any],
) -> dict[str, str]:
    sample_size = overall["sample_size"]
    accuracy = overall["accuracy"]

    bug_signals: list[str] = []
    if sign_label_consistency["mismatch_count"] > 0 and (sign_label_consistency["mismatch_rate"] or 0) > 0.01:
        bug_signals.append("label mismatch rate above 1%")
    if price_consistency["violation_count"] > 0:
        bug_signals.append("price/return reconstruction violations found")
    if ordering.get("status") == "ok" and ordering.get("source_violation_count", 0) > 0:
        bug_signals.append("ordering violations found on auto_trade_scan-filtered rows")
    strong_inversion_slices = [
        item
        for item in inversion["slices"]
        if item["pattern_note"] == "possible_wiring_or_label_bug"
    ]

    if sample_size == 0:
        q1 = "INCONCLUSIVE"
        q2 = "unknown"
        q3 = (
            "Cannot answer: 0 resolved auto_trade_scan directional rows are present in this "
            "database, so no accuracy/return distribution can be computed here."
        )
    elif bug_signals:
        q1 = "YES"
        if sign_label_consistency["mismatch_count"] > 0 and price_consistency["violation_count"] == 0:
            q2 = "label inversion"
        elif price_consistency["violation_count"] > 0:
            q2 = "price reconstruction mismatch"
        elif ordering.get("source_violation_count", 0) > 0:
            q2 = "timestamp/candle alignment"
        else:
            q2 = "unknown"
        q3 = "N/A -- see Q2 bug class."
    else:
        q1 = "NO"
        q2 = "N/A"
        q3 = (
            f"Directional-only accuracy is {_format_pct(accuracy)} on n={sample_size} resolved "
            "BULLISH/BEARISH rows, with no label/sign/price-reconstruction mismatches detected "
            f"({sign_label_consistency['mismatch_count']} of {sign_label_consistency['checked_row_count']} "
            "checked rows disagreed) and no inversion slice crossing the "
            f"{LIKELY_BUG_DISTANCE:.2f} likely_bug distance threshold with n>={MIN_SLICE_SAMPLE_SIZE}. "
            "This is consistent with a genuinely weak (near/below-random) directional signal at scale "
            "rather than a wiring bug."
        )

    q4 = "No live execution change in this block. Any future change must go through the roadmap and tests."

    if sample_size < MIN_SLICE_SAMPLE_SIZE:
        q5 = "INCONCLUSIVE -- insufficient resolved sample size in this database to judge source authority."
    elif q1 == "YES":
        q5 = (
            "INCONCLUSIVE -- a concrete bug was flagged (see Q2); auto_trade_scan's authority status "
            "should not be decided until the bug is root-caused and fixed, then re-measured."
        )
    elif accuracy is not None and accuracy < 0.5 - LIKELY_BUG_DISTANCE:
        q5 = (
            f"NO -- directional-only accuracy {_format_pct(accuracy)} (n={sample_size}) is well below "
            "random with no bug evidence, indicating genuinely weak signal; it should not remain "
            "highest-authority source without further investigation (e.g. the v2 EV-gated rebuild)."
        )
    else:
        q5 = f"YES -- directional-only accuracy {_format_pct(accuracy)} (n={sample_size}) shows no red flags."

    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}


def _inversion_table_lines(inversion: dict[str, Any]) -> list[str]:
    lines = [
        "| Horizon | Direction | n | Original Acc | Inverted Acc | Original Avg Return | Inverted Avg Return | Pattern |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in inversion["slices"]:
        lines.append(
            "| {horizon} | {direction} | {n} | {oa} | {ia} | {oar} | {iar} | {pattern} |".format(
                horizon=item["horizon"],
                direction=item["direction"],
                n=item["sample_size"],
                oa=_format_pct(item["original_accuracy"]),
                ia=_format_pct(item["inverted_accuracy"]),
                oar=_format_return(item["original_average_return_pct"]),
                iar=_format_return(item["inverted_average_return_pct"]),
                pattern=item["pattern_note"],
            )
        )
    return lines


def _horizon_table_lines(diagnostics: dict[str, Any]) -> list[str]:
    lines = [
        "| Horizon | Overall n | Overall Acc | BULLISH n | BULLISH Acc | BEARISH n | BEARISH Acc | Avg Return (BULL/BEAR) | Median Return (BULL/BEAR) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        overall = diagnostics["by_horizon"][horizon]
        bull = diagnostics["bullish_by_horizon"][horizon]
        bear = diagnostics["bearish_by_horizon"][horizon]
        returns = diagnostics["returns_by_direction_horizon"][horizon]
        lines.append(
            "| {h} | {on} | {oa} | {bn} | {ba} | {brn} | {bra} | {avg_b}/{avg_r} | {med_b}/{med_r} |".format(
                h=horizon,
                on=overall["sample_size"],
                oa=_format_pct(overall["accuracy"]),
                bn=bull["sample_size"],
                ba=_format_pct(bull["accuracy"]),
                brn=bear["sample_size"],
                bra=_format_pct(bear["accuracy"]),
                avg_b=_format_return(returns["BULLISH"]["average_gross_return_pct"]),
                avg_r=_format_return(returns["BEARISH"]["average_gross_return_pct"]),
                med_b=_format_return(returns["BULLISH"]["median_gross_return_pct"]),
                med_r=_format_return(returns["BEARISH"]["median_gross_return_pct"]),
            )
        )
    return lines


def run_investigation(
    *,
    db_path: str,
    source: str = DEFAULT_SOURCE,
    limit_samples: int = DEFAULT_LIMIT_SAMPLES,
    out: str = DEFAULT_OUT,
) -> dict[str, Any]:
    rows = load_source_rows(db_path, source)
    diagnostics = compute_directional_diagnostics(rows)
    resolved_directional_rows = diagnostics["resolved_directional_rows"]

    inversion = compute_inversion_diagnostics(resolved_directional_rows)
    ordering = check_ordering_for_source(db_path, source)
    sign_label_consistency = compute_sign_label_consistency(resolved_directional_rows)
    price_consistency = compute_price_consistency(resolved_directional_rows)
    answers = _answer_questions(
        overall=diagnostics["overall"],
        sign_label_consistency=sign_label_consistency,
        price_consistency=price_consistency,
        ordering=ordering,
        inversion=inversion,
    )

    sample_rows = build_sample_export(resolved_directional_rows, limit_samples=limit_samples)
    out_path = Path(out)
    csv_path = out_path.parent / SAMPLE_CSV_NAME
    write_sample_csv(sample_rows, csv_path)

    ordering_note = ordering.get("note") or "check_ordering ran against source-filtered rows; see counts above."
    ordering_disagreement = (
        ordering.get("status") == "ok" and ordering.get("source_violation_count", 0) > 0
    )
    ordering_anomaly_lines = (
        [
            "- ANOMALY: source-filtered ordering check found violations that the full-dataset "
            "'0 ordering violations' claim did not surface. Reported without reconciliation."
        ]
        if ordering_disagreement
        else []
    )

    markdown_lines = [
        "# auto_trade_scan Forensic Investigation (Block A.3)",
        "",
        f"Source: `{source}`",
        f"Total rows loaded for source: {len(rows)}",
        f"Resolved directional (BULLISH/BEARISH) rows: {diagnostics['overall']['sample_size']}",
        "",
        "## Data Availability Note",
        (
            "prediction_events does not persist a raw, direction-agnostic `actual_change_pct` "
            "column -- only `gross_return_pct` (already sign-adjusted for direction at migration "
            "time) plus `entry_price`/`resolution_price`. This report reconstructs a raw "
            "`actual_change_pct` by undoing that sign adjustment "
            "(`actual_change_pct_from_gross_return`), and independently reconstructs a change "
            "percentage from `entry_price`/`resolution_price` (`reconstructed_change_pct`). "
            "Materially disagreeing values between these two independent reconstructions is a "
            "price/sign consistency signal (see Task 6 section below)."
        ),
        "",
        "## Q1. Is auto_trade_scan likely affected by a concrete label/sign/timestamp bug?",
        f"Answer: {answers['q1']}",
        "",
        "## Q2. If YES, what is the most likely bug class?",
        f"Answer: {answers['q2']}",
        "",
        "## Q3. If NO, does the evidence suggest genuinely weak directional signal?",
        answers["q3"],
        "",
        "## Q4. Should we change live execution based on this investigation?",
        answers["q4"],
        "",
        "## Q5. Should auto_trade_scan remain highest-authority source?",
        answers["q5"],
        "",
        "## Directional-Only Diagnostics",
        (
            f"- Overall directional-only accuracy: {_format_pct(diagnostics['overall']['accuracy'])} "
            f"(n={diagnostics['overall']['sample_size']})."
        ),
        "",
        *_horizon_table_lines(diagnostics),
        "",
        "## Inversion Hypothesis Test (diagnostic only -- no should_invert field, no trading recommendation)",
        *_inversion_table_lines(inversion),
        "",
        "## Ordering Check (reused from app.audit.forensic.check_ordering)",
        f"Status: {ordering['status']}",
        (
            f"- Full-dataset violation count (this run): {ordering.get('full_dataset_violation_count')}, "
            f"source-filtered violation count: {ordering.get('source_violation_count')}."
            if ordering["status"] == "ok"
            else f"- {ordering_note}"
        ),
        *ordering_anomaly_lines,
        "",
        "## Sign/Label Consistency (Task 5, using real tracker._direction_from_change)",
        (
            f"- Checked rows: {sign_label_consistency['checked_row_count']}; "
            f"mismatches: {sign_label_consistency['mismatch_count']}; "
            f"mismatch_rate: {_format_pct(sign_label_consistency['mismatch_rate'])}."
        ),
        "",
        "## Price Consistency (Task 6)",
        (
            f"- Checked rows: {price_consistency['checked_row_count']}; "
            f"violations (material diff > {MATERIAL_CHANGE_DIFF_PCT}pp or label disagreement): "
            f"{price_consistency['violation_count']}."
        ),
        "",
        f"## Sample Export\nSee `{csv_path.as_posix()}` for {len(sample_rows)} representative rows.",
        "",
    ]
    markdown = "\n".join(markdown_lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    return {
        "content": markdown,
        "output_path": str(out_path),
        "sample_csv_path": str(csv_path),
        "diagnostics": diagnostics,
        "inversion": inversion,
        "ordering": ordering,
        "sign_label_consistency": sign_label_consistency,
        "price_consistency": price_consistency,
        "answers": answers,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block A.3 forensic investigation of auto_trade_scan.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="prediction_events.source to investigate.")
    parser.add_argument("--limit-samples", type=int, default=DEFAULT_LIMIT_SAMPLES, help="Row pool cap before sampling.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output markdown report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_investigation(
        db_path=args.db_path,
        source=args.source,
        limit_samples=args.limit_samples,
        out=args.out,
    )
    print(result["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
