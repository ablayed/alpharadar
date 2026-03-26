from __future__ import annotations

import time

from src.trade.portfolio import OpenPosition, PortfolioTracker


def _position(
    *,
    symbol: str,
    risk_pct: float,
    opened_at: float | None = None,
    tp1_hit: bool = False,
) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        asset_type="crypto",
        direction="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        tp1=105.0,
        risk_pct=risk_pct,
        opened_at=opened_at if opened_at is not None else time.time(),
        tp1_hit=tp1_hit,
    )


def test_check_risk_budget_enforces_monthly_cap() -> None:
    tracker = PortfolioTracker()
    tracker.add_position(_position(symbol="BTC", risk_pct=5.0))

    allowed, reason = tracker.check_risk_budget(2.0)
    replacing_allowed, _ = tracker.check_risk_budget(2.0, replacing_symbol="BTC")

    assert allowed is False
    assert "6% monthly risk cap" in reason
    assert replacing_allowed is True


def test_check_correlation_overlap_warns_on_highly_correlated_positions() -> None:
    tracker = PortfolioTracker()
    tracker.add_position(_position(symbol="ETH", risk_pct=2.0))

    warning = tracker.check_correlation_overlap("BTC", {"ETH": 0.82})

    assert warning is not None
    assert "BTC is highly correlated (0.82)" in warning
    assert "ETH position" in warning


def test_dead_money_detection_and_status_output() -> None:
    tracker = PortfolioTracker()
    stale_opened_at = time.time() - (8 * 86400)
    tracker.add_position(_position(symbol="BTC", risk_pct=2.0, opened_at=stale_opened_at))
    tracker.add_position(_position(symbol="ETH", risk_pct=1.0, opened_at=stale_opened_at, tp1_hit=True))

    stale = tracker.get_dead_money_trades()
    status = tracker.format_status()

    assert stale == ["BTC"]
    assert "Dead money (7+ days, no TP1):</b> BTC" in status
    assert "BTC LONG | crypto | risk: 2.0%" in status
