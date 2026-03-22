from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.alerts import AlertPipeline, AlertSeverity


def test_alert_pipeline_classifies_warning_and_respects_cooldown() -> None:
    pipeline = AlertPipeline(
        time_func=lambda: 1_000.0,
        london_now_func=lambda: datetime(2026, 3, 23, 14, 0, tzinfo=ZoneInfo("Europe/London")),
    )
    pipeline.set_reference_price("NVDA", 100.0)

    alert = pipeline.classify_severity("NVDA", 103.5, "stock")

    assert alert is not None
    assert alert.severity is AlertSeverity.WARNING
    assert round(alert.change_pct, 2) == 3.5

    pipeline.record_sent(alert)
    repeat = pipeline.classify_severity("NVDA", 104.0, "stock")
    assert repeat is None


def test_alert_pipeline_blocks_during_quiet_hours_unless_critical() -> None:
    pipeline = AlertPipeline(
        time_func=lambda: 2_000.0,
        london_now_func=lambda: datetime(2026, 3, 23, 23, 30, tzinfo=ZoneInfo("Europe/London")),
    )
    pipeline.set_reference_price("BTC", 100.0)

    warning = pipeline.classify_severity("BTC", 109.0, "crypto")
    critical = pipeline.classify_severity("BTC", 113.0, "crypto")

    assert warning is None
    assert critical is not None
    assert critical.severity is AlertSeverity.CRITICAL


def test_alert_pipeline_enforces_global_hourly_limit_for_non_critical_alerts() -> None:
    current_time = {"value": 10_000.0}
    pipeline = AlertPipeline(
        time_func=lambda: current_time["value"],
        london_now_func=lambda: datetime(2026, 3, 23, 15, 0, tzinfo=ZoneInfo("Europe/London")),
    )
    pipeline.set_reference_price("AAPL", 100.0)

    for index in range(pipeline.GLOBAL_MAX_PER_HOUR):
        pipeline.set_reference_price(f"SYM{index}", 100.0)
        alert = pipeline.classify_severity(f"SYM{index}", 102.5, "stock")
        assert alert is not None
        pipeline.record_sent(alert)
        current_time["value"] += 120.0

    blocked = pipeline.classify_severity("AAPL", 102.5, "stock")
    assert blocked is None

    critical = pipeline.classify_severity("AAPL", 106.0, "stock")
    assert critical is not None
    assert critical.severity is AlertSeverity.CRITICAL
