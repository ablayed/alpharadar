from __future__ import annotations

import numpy as np

from src.data.breakout_scorer import score_breakout


def _long_setup_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.array(
        [
            100.0,
            101.5,
            99.0,
            102.0,
            98.5,
            103.0,
            99.5,
            102.5,
            100.0,
            101.8,
            100.2,
            101.4,
            100.8,
            101.2,
            100.9,
            101.6,
            101.7,
            101.8,
            101.9,
            102.0,
            102.1,
            102.2,
            103.8,
            103.85,
            103.9,
            104.0,
            104.1,
            104.2,
            104.25,
            104.3,
        ],
        dtype=np.float64,
    )
    high = close + 0.35
    high[15] = 105.0
    low = close - 0.35
    volume = np.array(
        ([100.0] * 22) + [120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 320.0],
        dtype=np.float64,
    )
    open_ = close - 0.1
    return close, high, low, volume, open_


def _short_setup_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.array(
        [
            110.0,
            109.2,
            108.5,
            107.8,
            107.0,
            106.2,
            105.5,
            104.8,
            104.0,
            103.5,
            103.0,
            102.5,
            101.8,
            101.2,
            100.8,
            100.2,
            99.8,
            99.2,
            98.8,
            98.2,
            97.8,
            97.2,
            96.4,
            96.35,
            96.3,
            96.25,
            96.2,
            96.15,
            96.1,
            96.05,
        ],
        dtype=np.float64,
    )
    high = close + 0.15
    low = close - 0.15
    low[-1] = 95.9
    volume = np.array(
        ([110.0] * 22) + [140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 320.0],
        dtype=np.float64,
    )
    open_ = close - 0.02
    return close, high, low, volume, open_


def test_score_breakout_detects_long_setup() -> None:
    close, high, low, volume, open_ = _long_setup_arrays()

    result = score_breakout(
        symbol="ENJ",
        close=close,
        high=high,
        low=low,
        volume=volume,
        open_=open_,
        taker_buy_ratio=0.58,
        is_trending=True,
        price_change_24h=2.4,
    )

    assert result.direction == "long_setup"
    assert result.total_score >= 60
    assert result.signals_firing >= 4
    assert result.volume_accumulation >= 20
    assert result.price_position >= 8
    assert result.taker_pressure == 10
    assert result.social_velocity == 10
    assert result.key_level is not None


def test_score_breakout_detects_short_setup() -> None:
    close, high, low, volume, open_ = _short_setup_arrays()

    result = score_breakout(
        symbol="SHIB",
        close=close,
        high=high,
        low=low,
        volume=volume,
        open_=open_,
        taker_buy_ratio=0.42,
        is_trending=False,
        price_change_24h=-8.0,
    )

    assert result.direction == "short_setup"
    assert result.total_score >= 40
    assert result.signals_firing >= 3
    assert result.volume_accumulation >= 15
    assert result.price_position >= 8
    assert result.taker_pressure == 10
    assert result.key_level is not None


def test_score_breakout_returns_neutral_when_signals_are_mixed() -> None:
    close = np.array(
        [
            100.0,
            103.0,
            98.0,
            104.0,
            97.0,
            105.0,
            96.0,
            104.0,
            97.0,
            103.0,
            98.0,
            102.0,
            99.0,
            101.0,
            100.0,
            99.0,
            101.0,
            100.0,
            102.0,
            98.0,
            101.0,
            99.0,
            100.0,
            101.0,
            99.0,
            100.0,
            101.0,
            99.0,
            100.0,
            100.0,
        ],
        dtype=np.float64,
    )
    high = close + 0.5
    low = close - 0.5
    volume = np.full_like(close, 100.0)

    result = score_breakout(
        symbol="PENGU",
        close=close,
        high=high,
        low=low,
        volume=volume,
        taker_buy_ratio=None,
        is_trending=False,
        price_change_24h=0.2,
    )

    assert result.direction == "neutral"
    assert result.total_score < 30
    assert result.signals_firing == 0


def test_score_breakout_rejects_insufficient_data() -> None:
    close = np.arange(10, dtype=np.float64)

    result = score_breakout(
        symbol="NEW",
        close=close,
        high=close + 1,
        low=close - 1,
        volume=np.full_like(close, 100.0),
    )

    assert result.direction == "neutral"
    assert result.total_score == 0
    assert result.description == "Insufficient data"
