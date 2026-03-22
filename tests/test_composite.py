from __future__ import annotations

import time

from src.signals import SignalInput, compute_composite


def test_compute_composite_balances_signal_decay_and_direction() -> None:
    now = time.time()
    signals = [
        SignalInput("sentiment", 0.8, 0.9, now - 3600, 12),
        SignalInput("ta", 0.6, 0.8, now - 2 * 3600, 72),
        SignalInput("event", -0.4, 0.7, now - 10 * 3600, 48),
        SignalInput("correlation", 0.3, 0.6, now - 24 * 3600, 168),
    ]

    composite = compute_composite("NVDA", signals, regime="risk_on")

    assert composite.symbol == "NVDA"
    assert -1.0 <= composite.score <= 1.0
    assert composite.direction in {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
    assert 0.0 <= composite.confidence <= 0.95
    assert set(composite.signal_breakdown) == {"sentiment", "ta", "event", "correlation"}
    assert "Regime: risk_on." in composite.summary


def test_compute_composite_handles_signal_disagreement() -> None:
    now = time.time()
    signals = [
        SignalInput("sentiment", 1.0, 1.0, now, 12),
        SignalInput("ta", -1.0, 1.0, now, 72),
        SignalInput("event", 0.0, 1.0, now, 48),
    ]

    composite = compute_composite("BTC", signals)

    assert composite.direction in {"neutral", "sell", "buy"}
    assert composite.agreement < 1.0
