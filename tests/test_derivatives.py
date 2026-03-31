from __future__ import annotations

import pandas as pd

from src.data.derivatives import (
    _classify_funding,
    _classify_open_interest,
    _detect_cme_gaps_from_history,
)


def test_classify_funding_marks_crowded_longs_as_bearish() -> None:
    data = _classify_funding("BTC", 0.0012, 2.7)

    assert data.signal == "bearish"
    assert data.rate_annualized > 100
    assert "market heavily long" in data.description
    assert "z-score +2.7" in data.description


def test_classify_open_interest_flags_fast_cascade_risk() -> None:
    data = _classify_open_interest("ETH", 1_000_000.0, 12.4, 8.1)

    assert data.signal == "warning"
    assert "cascade risk" in data.description
    assert data.oi_change_4h_pct == 12.4


def test_detect_cme_gaps_returns_nearest_unfilled_gap() -> None:
    history = pd.DataFrame(
        {
            "Open": [100.0, 112.0, 118.0, 120.0, 132.0],
            "High": [101.0, 119.0, 121.0, 122.0, 142.0],
            "Low": [99.0, 111.0, 100.0, 119.0, 131.0],
            "Close": [100.0, 118.0, 119.0, 120.0, 140.0],
        },
        index=pd.to_datetime(
            ["2026-03-06", "2026-03-09", "2026-03-10", "2026-03-13", "2026-03-16"]
        ),
    )

    gap = _detect_cme_gaps_from_history(history, current_btc_price=126.0)

    assert gap is not None
    assert gap.gap_low == 120.0
    assert gap.gap_high == 132.0
    assert gap.gap_direction == "up"
    assert gap.filled is False
