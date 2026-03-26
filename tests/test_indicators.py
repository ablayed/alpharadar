from __future__ import annotations

import numpy as np

from src.ta.indicators import (
    atr,
    bollinger_bands,
    ema,
    obv,
    obv_divergence,
    rsi,
    sma,
    volume_confirmation,
)
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


def test_obv_and_divergence_helpers_detect_flow_shifts() -> None:
    close = np.array([1.0, 2.0, 1.0, 1.0, 3.0], dtype=np.float64)
    volume = np.array([10.0, 5.0, 7.0, 2.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(obv(close, volume), np.array([0.0, 5.0, -2.0, -2.0, 1.0]))

    bullish_close = np.array([12.0, 11.0, 10.0, 11.0, 10.0, 9.0], dtype=np.float64)
    bullish_volume = np.array([100.0, 100.0, 100.0, 100.0, 1.0, 1.0], dtype=np.float64)
    bearish_close = np.array([10.0, 11.0, 12.0, 11.0, 12.0, 13.0], dtype=np.float64)
    bearish_volume = np.array([100.0, 100.0, 100.0, 100.0, 1.0, 1.0], dtype=np.float64)

    assert obv_divergence(bullish_close, bullish_volume, lookback=3) == "bullish_divergence"
    assert obv_divergence(bearish_close, bearish_volume, lookback=3) == "bearish_divergence"


def test_volume_confirmation_flags_breakout_volume() -> None:
    volume = np.array([100.0] * 20 + [140.0], dtype=np.float64)

    confirmed, ratio = volume_confirmation(volume, lookback=20, threshold=1.25)

    assert confirmed is True
    assert ratio == 1.4


def test_compute_signals_returns_structured_ta_summary() -> None:
    close = np.linspace(100.0, 150.0, 80)
    high = close + 1.5
    low = close - 1.5
    volume = np.linspace(1_000.0, 1_500.0, 80)
    open_ = close - 0.5

    signal = compute_signals("NVDA", close, high, low, volume=volume, open_=open_)

    assert signal.symbol == "NVDA"
    assert signal.timeframe == "daily"
    assert signal.current_price == close[-1]
    assert signal.overall_signal in {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
    assert 0.0 <= signal.confidence <= 1.0
    assert "NVDA:" in signal.summary
    np.testing.assert_allclose(signal._close, close)
    np.testing.assert_allclose(signal._high, high)
    np.testing.assert_allclose(signal._low, low)
    np.testing.assert_allclose(signal._volume, volume)
    np.testing.assert_allclose(signal._open, open_)
