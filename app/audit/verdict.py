from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.audit.forensic import check_inversion_by_source, run_forensic_audit
from app.reporting.ev_metrics import DEFAULT_WINDOW_DAYS, compute_group_metrics, load_prediction_event_rows, summarize_rows

MIN_RESOLVED_ROWS = 100
MIN_SLICE_RESOLVED_ROWS = 30


def _bucket_floor(bucket: str) -> float:
    try:
        return float(str(bucket).split("_", 1)[0])
    except (TypeError, ValueError):
        return -1.0


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _source_metric(metrics: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    for metric in metrics:
        if metric.get("group_value") == source:
            return metric
    return None


def _status_text(value: bool | str) -> str:
    if value == "insufficient_sample":
        return "insufficient_sample"
    return "Yes" if value else "No"


def _is_resolved(row: dict[str, Any]) -> bool:
    resolved_value = row.get("resolved")
    resolved_int = _optional_int(resolved_value)
    if resolved_int is not None:
        return resolved_int == 1
    return bool(resolved_value)


def _resolved_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _is_resolved(row) and _optional_int(row.get("outcome_label")) is not None
    ]


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] += 1
    return dict(sorted(counts.items()))


def _slice_status(count: int) -> str:
    return "eligible" if count >= MIN_SLICE_RESOLVED_ROWS else "insufficient_sample"


def _breakdown_lines(
    counts: dict[str, int],
    *,
    include_status: bool,
) -> list[str]:
    if not counts:
        return ["- No resolved prediction_events found."]

    lines: list[str] = []
    for key, count in counts.items():
        if include_status:
            lines.append(f"- {key}: {_slice_status(count)} (resolved_rows={count})")
        else:
            lines.append(f"- {key}: resolved_rows={count}")
    return lines


def _select_verdict(*, bug_indicators: int, positive_signal_indicators: int) -> str:
    if bug_indicators >= 2 and positive_signal_indicators <= 1:
        return "BUG_DOMINANT"
    if bug_indicators == 0 and positive_signal_indicators <= 1:
        return "SIGNAL_WEAK"
    return "MIXED"


def _root_cause_lines() -> list[str]:
    return [
        "## Root Cause Findings (informational, does not override sample-size gate)",
        (
            "- Confirmed scheduling bug: legacy outcome resolution used a fixed BETWEEN-style absolute time band, "
            "which permanently missed predictions created outside that band. Block A.3 replaces that behavior "
            "with an open-ended age cutoff for ongoing resolution."
        ),
        (
            "- Historical reference figures `37.3%`, `33.3%`, `29.6%`, and `99 resolved` could not be verified "
            "from this repository's current database, self-report tables, or discovery hits. They should be "
            "treated as unverifiable rather than as audit ground truth."
        ),
        "- This section is documentation only. It does not bypass the INSUFFICIENT_DATA floor or alter verdict assignment logic.",
    ]


def generate(
    *,
    db_path: str,
    out: str = "audit_verdict.md",
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict[str, Any]:
    findings = run_forensic_audit(db_path, window_days=window_days)
    prediction_rows = load_prediction_event_rows(db_path, window_days=window_days)
    resolved_rows = _resolved_prediction_rows(prediction_rows)
    resolved_row_count = len(resolved_rows)
    source_breakdown = _count_by(resolved_rows, "source")
    horizon_breakdown = _count_by(resolved_rows, "horizon")

    if resolved_row_count < MIN_RESOLVED_ROWS:
        markdown = "\n".join(
            [
                "# Audit Verdict",
                "",
                "Verdict: INSUFFICIENT_DATA",
                "",
                f"resolved_row_count: {resolved_row_count}",
                f"required_minimum: {MIN_RESOLVED_ROWS}",
                "",
                "BUG_DOMINANT / SIGNAL_WEAK / MIXED cannot be assigned yet.",
                "",
                "## Source Breakdown",
                *_breakdown_lines(source_breakdown, include_status=False),
                "",
                "## Horizon Breakdown",
                *_breakdown_lines(horizon_breakdown, include_status=False),
                "",
                *_root_cause_lines(),
            ]
        )
        output_path = Path(out)
        output_path.write_text(markdown, encoding="utf-8")
        return {
            "verdict": "INSUFFICIENT_DATA",
            "resolved_row_count": resolved_row_count,
            "required_minimum": MIN_RESOLVED_ROWS,
            "source_breakdown": source_breakdown,
            "horizon_breakdown": horizon_breakdown,
            "output_path": str(output_path),
            "content": markdown,
            "findings": findings,
        }

    source_metrics = compute_group_metrics(resolved_rows, group_by="source")
    bucket_metrics = compute_group_metrics(resolved_rows, group_by="confidence_bucket")
    top_bucket_metric = max(bucket_metrics, key=lambda item: _bucket_floor(item["group_value"]), default=None)
    high_confidence_rows = [
        row
        for row in resolved_rows
        if _bucket_floor(str(row.get("confidence_bucket") or "")) >= 0.65
    ]
    high_confidence_summary = summarize_rows(high_confidence_rows)
    inversion_sources = check_inversion_by_source(resolved_rows)["sources"]
    eligible_inversion_sources = [
        source for source in inversion_sources if int(source["sample_size"]) >= MIN_SLICE_RESOLVED_ROWS
    ]

    top_bucket_status: bool | str
    if (
        top_bucket_metric is None
        or int(top_bucket_metric["sample_size"]) < MIN_SLICE_RESOLVED_ROWS
        or top_bucket_metric.get("net_ev_pct") is None
    ):
        top_bucket_status = "insufficient_sample"
    else:
        top_bucket_status = bool(float(top_bucket_metric["net_ev_pct"]) > 0)

    high_confidence_status: bool | str
    if (
        int(high_confidence_summary["sample_size"]) < MIN_SLICE_RESOLVED_ROWS
        or high_confidence_summary["accuracy"] is None
        or high_confidence_summary["net_ev_pct"] is None
    ):
        high_confidence_status = "insufficient_sample"
    else:
        high_confidence_status = bool(
            float(high_confidence_summary["accuracy"]) >= 0.60
            and float(high_confidence_summary["net_ev_pct"]) > 0
        )

    source_level_status: bool | str
    if not eligible_inversion_sources:
        source_level_status = "insufficient_sample"
    else:
        source_level_status = all(source["likely_bug"] is not True for source in eligible_inversion_sources)

    auto_trade_scan_metric = _source_metric(source_metrics, "auto_trade_scan")
    auto_trade_scan_status: bool | str
    if (
        auto_trade_scan_metric is None
        or int(auto_trade_scan_metric["sample_size"]) < MIN_SLICE_RESOLVED_ROWS
        or auto_trade_scan_metric.get("accuracy") is None
        or auto_trade_scan_metric.get("net_ev_pct") is None
    ):
        auto_trade_scan_status = "insufficient_sample"
    else:
        auto_trade_scan_status = bool(
            float(auto_trade_scan_metric["accuracy"]) > 0.5
            and float(auto_trade_scan_metric["net_ev_pct"]) > 0.0
        )

    top_bucket_positive = top_bucket_status is True and top_bucket_metric is not None
    high_confidence_calibrated = bool(high_confidence_status is True)
    auto_trade_scan_beats_baselines = bool(auto_trade_scan_status is True)

    bug_indicators = 0
    if findings["checks"]["ordering"]["violations"]:
        bug_indicators += 1
    if findings["checks"]["price_source"]["violations"]:
        bug_indicators += 1
    if findings["checks"]["duplicates"]["duplicate_pair_count"] > 0:
        bug_indicators += 1
    if source_level_status is False:
        bug_indicators += 1

    positive_signal_indicators = sum(
        int(value)
        for value in (
            top_bucket_positive,
            high_confidence_calibrated,
            auto_trade_scan_beats_baselines,
        )
    )
    verdict = _select_verdict(
        bug_indicators=bug_indicators,
        positive_signal_indicators=positive_signal_indicators,
    )

    auto_trade_scan_accuracy = None if auto_trade_scan_metric is None else auto_trade_scan_metric["accuracy"]
    auto_trade_scan_net_ev = None if auto_trade_scan_metric is None else auto_trade_scan_metric["net_ev_pct"]
    top_bucket_label = "n/a" if top_bucket_metric is None else str(top_bucket_metric["group_value"])
    top_bucket_net_ev = None if top_bucket_metric is None else top_bucket_metric["net_ev_pct"]
    top_bucket_sample_size = 0 if top_bucket_metric is None else int(top_bucket_metric["sample_size"])
    auto_trade_scan_sample_size = 0 if auto_trade_scan_metric is None else int(auto_trade_scan_metric["sample_size"])

    markdown = "\n".join(
        [
            "# Audit Verdict",
            "",
            f"Verdict: {verdict}",
            "",
            f"resolved_row_count: {resolved_row_count}",
            f"required_minimum: {MIN_RESOLVED_ROWS}",
            "",
            "## Gate Questions",
            (
                f"- Did top-bucket EV turn positive after costs? {_status_text(top_bucket_status)}. "
                f"Evidence: bucket `{top_bucket_label}` resolved_rows = {top_bucket_sample_size}, "
                f"net EV = {top_bucket_net_ev if top_bucket_net_ev is not None else 'n/a'}."
            ),
            (
                f"- Did high-confidence become calibrated? {_status_text(high_confidence_status)}. "
                f"Evidence: high-confidence sample = {high_confidence_summary['sample_size']}, "
                f"accuracy = {high_confidence_summary['accuracy']}, "
                f"net EV = {high_confidence_summary['net_ev_pct']}."
            ),
            (
                f"- Did source-level results stop being systematically broken? {_status_text(source_level_status)}. "
                f"Evidence: inversion findings = {inversion_sources}."
            ),
            (
                f"- Did auto_trade_scan beat no-trade and random baselines? {_status_text(auto_trade_scan_status)}. "
                f"Evidence: auto_trade_scan resolved_rows = {auto_trade_scan_sample_size}, accuracy = {auto_trade_scan_accuracy}, "
                f"net EV = {auto_trade_scan_net_ev}; baselines = accuracy 0.5, net EV 0.0."
            ),
            "",
            "## Source Slice Readiness",
            *_breakdown_lines(source_breakdown, include_status=True),
            "",
            "## Horizon Slice Readiness",
            *_breakdown_lines(horizon_breakdown, include_status=True),
            "",
            "## Supporting Evidence",
            (
                f"- Ordering violations: {len(findings['checks']['ordering']['violations'])}. "
                f"Price-source violations: {len(findings['checks']['price_source']['violations'])}. "
                f"Duplicate pairs: {findings['checks']['duplicates']['duplicate_pair_count']}."
            ),
            (
                f"- Neutral audit: {findings['checks']['neutral_audit'].get('missed_explosive_count', 0)} "
                f"missed explosive moves out of {findings['checks']['neutral_audit'].get('neutral_count', 0)} neutral calls."
            ),
            (
                f"- Survivorship check: {findings['checks']['survivorship']['finding']}."
            ),
            "",
            *_root_cause_lines(),
        ]
    )

    output_path = Path(out)
    output_path.write_text(markdown, encoding="utf-8")
    return {
        "verdict": verdict,
        "resolved_row_count": resolved_row_count,
        "required_minimum": MIN_RESOLVED_ROWS,
        "source_breakdown": source_breakdown,
        "horizon_breakdown": horizon_breakdown,
        "output_path": str(output_path),
        "content": markdown,
        "findings": findings,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the Block A audit verdict markdown file.")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--out", default="audit_verdict.md", help="Output markdown path.")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Trailing window in days.")
    return parser


def main(argv: list[str] | None = None, *, db_path: str | None = None, out: str = "audit_verdict.md", window_days: int = DEFAULT_WINDOW_DAYS) -> int:
    if db_path is None:
        args = _build_arg_parser().parse_args(argv)
        db_path = args.db_path
        out = args.out
        window_days = args.window_days

    result = generate(db_path=db_path, out=out, window_days=window_days)
    print(result["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
