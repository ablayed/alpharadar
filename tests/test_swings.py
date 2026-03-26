from __future__ import annotations

import pytest
import numpy as np

from src.ta.swings import compute_fibonacci, detect_stop_hunt, detect_swing_points, get_last_swing


def test_detect_swing_points_and_last_swing() -> None:
    high = np.array([10.0, 12.0, 15.0, 12.0, 11.0, 13.0, 12.0], dtype=np.float64)
    low = np.array([8.0, 6.0, 7.0, 7.5, 5.0, 6.5, 7.0], dtype=np.float64)

    swing_highs, swing_lows = detect_swing_points(high, low, order=1)
    last_high, last_low = get_last_swing(high, low, order=1)

    assert [point.index for point in swing_highs] == [2, 5]
    assert [point.index for point in swing_lows] == [1, 4]
    assert last_high is not None and last_high.index == 5 and last_high.price == 13.0
    assert last_low is not None and last_low.index == 4 and last_low.price == 5.0


def test_compute_fibonacci_returns_expected_levels() -> None:
    fib = compute_fibonacci(120.0, 100.0, "up")

    assert fib.fib_382 == pytest.approx(112.36, rel=1e-4)
    assert fib.fib_618 == pytest.approx(107.64, rel=1e-4)
    assert fib.ext_1272 == pytest.approx(125.44, rel=1e-4)
    assert fib.ext_1618 == pytest.approx(132.36, rel=1e-4)
    assert fib.ext_2618 == pytest.approx(152.36, rel=1e-4)


def test_detect_stop_hunt_finds_recent_wick_sweep() -> None:
    open_ = np.array([10.0, 10.0, 10.0, 10.0, 10.1], dtype=np.float64)
    close = np.array([10.1, 10.0, 9.9, 10.2, 10.2], dtype=np.float64)
    high = np.array([10.3, 10.2, 10.1, 10.4, 10.4], dtype=np.float64)
    low = np.array([9.8, 9.7, 9.6, 8.8, 9.9], dtype=np.float64)

    hunts = detect_stop_hunt(high, low, close, open_, lookback=3)

    assert len(hunts) == 1
    assert hunts[0]["type"] == "bearish_stop_hunt"
    assert hunts[0]["direction_after"] == "bullish"
