from __future__ import annotations

import numpy as np

from src.ta.indicators import atr, bollinger_bands, ema, rsi, sma
from src.ta.signals import compute_signals


def test_sma_and_ema_return_expected_values() -> None:
    close = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    sma_values = sma(close, period=3)
    ema_values = ema(close, period=2)

    assert np.isnan(sma_values[0])
    assert np.isnan(sma_values[1])
    np.testing.assert_allclose(sma_values[2:], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(
        ema_values,
        np.array([1.0, 1.66666667, 2.55555556, 3.51851852, 4.50617284]),
        rtol=1e-6,
    )


def test_rsi_reaches_extremes_on_monotonic_series() -> None:
    up_only = np.arange(1, 21, dtype=np.float64)
    down_only = np.arange(20, 0, -1, dtype=np.float64)

    up_rsi = rsi(up_only, period=14)
    down_rsi = rsi(down_only, period=14)

    assert up_rsi[-1] == 100.0
    assert down_rsi[-1] == 0.0


def test_bollinger_bands_and_atr_align_to_input_length() -> None:
    close = np.linspace(100.0, 129.0, 30)
    high = close + 2.0
    low = close - 2.0

    upper, middle, lower = bollinger_bands(close, period=20)
    atr_values = atr(high, low, close, period=14)

    assert len(upper) == len(close)
    assert len(middle) == len(close)
    assert len(lower) == len(close)
    assert len(atr_values) == len(close)
    assert upper[-1] > middle[-1] > lower[-1]
    assert atr_values[-1] > 0


def test_compute_signals_returns_structured_ta_summary() -> None:
    close = np.linspace(100.0, 150.0, 80)
    high = close + 1.5
    low = close - 1.5

    signal = compute_signals("NVDA", close, high, low)

    assert signal.symbol == "NVDA"
    assert signal.timeframe == "daily"
    assert signal.current_price == close[-1]
    assert signal.overall_signal in {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
    assert 0.0 <= signal.confidence <= 1.0
    assert "NVDA:" in signal.summary
