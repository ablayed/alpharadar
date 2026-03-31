from __future__ import annotations

import numpy as np

from src.ta.swings import (
    detect_break_of_structure,
    detect_fair_value_gaps,
    detect_order_blocks,
)


def test_detect_fair_value_gaps_finds_bullish_void() -> None:
    high = np.array([102.0, 109.0, 111.0], dtype=np.float64)
    low = np.array([99.0, 100.0, 108.0], dtype=np.float64)
    close = np.array([101.0, 108.0, 110.0], dtype=np.float64)
    open_ = np.array([100.0, 101.0, 109.0], dtype=np.float64)

    gaps = detect_fair_value_gaps(high, low, close, open_)

    assert len(gaps) == 1
    assert gaps[0]["type"] == "bullish_fvg"
    assert gaps[0]["bottom"] == 102.0
    assert gaps[0]["top"] == 108.0


def test_detect_break_of_structure_requires_close_beyond_wick() -> None:
    high = np.array([10.0, 12.0, 11.0, 12.4, 12.6], dtype=np.float64)
    low = np.array([9.0, 10.0, 9.8, 10.5, 11.0], dtype=np.float64)
    close = np.array([9.5, 11.0, 10.5, 11.9, 12.2], dtype=np.float64)

    bos_events = detect_break_of_structure(high, low, close, order=1)

    assert len(bos_events) == 1
    assert bos_events[0]["type"] == "bullish_bos"
    assert bos_events[0]["index"] == 4
    assert bos_events[0]["broken_level"] == 12.0


def test_detect_order_blocks_finds_pre_break_candle_zone() -> None:
    high = np.array([100.0, 105.0, 103.0, 104.0, 103.0, 102.0, 106.0], dtype=np.float64)
    low = np.array([98.0, 99.0, 97.0, 98.0, 97.0, 96.0, 101.0], dtype=np.float64)
    close = np.array([99.0, 104.0, 100.0, 103.0, 98.0, 97.0, 106.0], dtype=np.float64)
    open_ = np.array([99.5, 100.0, 101.0, 102.0, 99.0, 100.0, 101.0], dtype=np.float64)

    order_blocks = detect_order_blocks(high, low, close, open_, order=1)

    assert len(order_blocks) == 1
    assert order_blocks[0]["type"] == "bullish_ob"
    assert order_blocks[0]["index"] == 5
    assert order_blocks[0]["bottom"] == 96.0
    assert order_blocks[0]["top"] == 102.0
