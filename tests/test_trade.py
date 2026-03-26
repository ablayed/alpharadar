from __future__ import annotations

from src.agents.state import DebateVerdict
from src.signals.models import CompositeScore
from src.ta.signals import TASignal
from src.trade.generator import (
    BTC_TP_CAPS,
    SPOT_SL_CAPS,
    SPOT_TP_CAPS,
    STOCK_SL_CAPS,
    STOCK_TP_CAPS,
    _cap_spot_targets,
    generate_crypto_trade_signal,
    generate_trade_signal,
)


def _ta_signal(
    *,
    symbol: str = "BTC",
    current_price: float = 100.0,
    atr_value: float = 4.0,
    atr_pct: float = 4.0,
    bb_upper: float = 118.0,
    bb_lower: float = 94.0,
    bb_position: str = "middle",
    bb_squeeze: bool = False,
    trend: str = "up",
    overall_signal: str = "buy",
    confidence: float = 0.7,
    rsi_signal: str = "neutral",
    rsi_value: float = 52.0,
    macd_signal: str = "bullish",
    macd_histogram: float = 0.2,
) -> TASignal:
    return TASignal(
        symbol=symbol,
        timeframe="daily",
        current_price=current_price,
        rsi_value=rsi_value,
        rsi_signal=rsi_signal,
        macd_line=1.0,
        macd_signal_line=0.8,
        macd_histogram=macd_histogram,
        macd_signal=macd_signal,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_position=bb_position,
        bb_squeeze=bb_squeeze,
        atr_value=atr_value,
        atr_pct=atr_pct,
        volatility="high" if atr_pct > 4 else "normal",
        ema_20=current_price - 1.0,
        ema_50=current_price - 4.0,
        trend=trend,
        overall_signal=overall_signal,
        confidence=confidence,
        summary="test",
    )


def _composite(
    *,
    symbol: str = "BTC",
    score: float = 0.22,
    agreement: float = 0.75,
    direction: str = "buy",
    breakdown: dict[str, float] | None = None,
) -> CompositeScore:
    return CompositeScore(
        symbol=symbol,
        score=score,
        confidence=0.75,
        direction=direction,
        signal_breakdown=breakdown or {"sentiment": 0.08, "ta": 0.06, "event": 0.05},
        agreement=agreement,
        regime_context="risk_on",
        summary="test",
    )


def _verdict(
    *,
    symbol: str = "BTC",
    prevailing_view: str = "bullish",
    confidence: float = 0.82,
) -> DebateVerdict:
    return DebateVerdict(
        symbol=symbol,
        display_name=symbol,
        prevailing_view=prevailing_view,
        confidence=confidence,
        summary="test",
        bull_thesis="test",
        bear_thesis="test",
        key_risk="test",
        action_context="test",
    )


def test_generate_trade_signal_builds_stock_setup() -> None:
    signal = generate_trade_signal(
        symbol="NVDA",
        current_price=100.0,
        ta_signal=_ta_signal(
            symbol="NVDA",
            current_price=100.0,
            atr_value=2.0,
            atr_pct=2.0,
            bb_upper=108.0,
            bb_lower=97.0,
        ),
        composite=_composite(symbol="NVDA"),
        verdict=_verdict(symbol="NVDA"),
        event_data={"event_type": "analyst_upgrade", "remaining_drift_pct": 6.0},
        regime="risk_on",
    )

    assert signal is not None
    assert signal.direction == "LONG"
    assert signal.tp1 < signal.tp2 < signal.tp3
    assert signal.risk_reward >= 1.5
    assert signal.tp3_pct <= STOCK_TP_CAPS["tp3_max_pct"] + 0.1
    assert signal.stop_loss_pct <= STOCK_SL_CAPS["sl_max_pct"] + 0.1


def test_generate_crypto_trade_signal_requires_three_confluence_factors() -> None:
    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=2_500.0,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=2_500.0,
            atr_value=90.0,
            atr_pct=3.6,
            trend="neutral",
            overall_signal="neutral",
            confidence=0.4,
            rsi_value=48.0,
            macd_signal="bullish",
            macd_histogram=0.1,
        ),
        composite=_composite(
            symbol="ETH",
            score=0.06,
            agreement=0.45,
            direction="neutral",
            breakdown={"sentiment": 0.03},
        ),
        verdict=None,
        event_data=None,
        regime="default",
        btc_change_pct=0.5,
    )

    assert signal is None


def test_generate_crypto_trade_signal_builds_high_confluence_long_setup() -> None:
    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=2_500.0,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=2_500.0,
            atr_value=85.0,
            atr_pct=3.4,
            bb_upper=2_950.0,
            bb_lower=2_340.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.72,
            rsi_signal="oversold",
            rsi_value=28.0,
            macd_signal="bullish_cross",
            macd_histogram=0.35,
        ),
        composite=_composite(
            symbol="ETH",
            score=0.24,
            agreement=0.78,
            direction="buy",
            breakdown={"sentiment": 0.11, "ta": 0.08, "event": 0.04},
        ),
        verdict=_verdict(symbol="ETH", prevailing_view="bullish", confidence=0.81),
        event_data={"event_type": "protocol_upgrade", "remaining_drift_pct": 4.5},
        regime="risk_on",
        btc_change_pct=3.1,
    )

    assert signal is not None
    assert signal.direction == "LONG"
    assert signal.confluence_count >= 3
    assert signal.suggested_risk_pct == 2.0
    assert signal.tp1 < signal.tp2 < signal.tp3
    assert signal.risk_reward >= 1.5
    assert len(signal.confluence_factors) == signal.confluence_count


def test_tp_caps_prevent_unrealistic_targets() -> None:
    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=2_100.0,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=2_100.0,
            atr_value=60.0,
            atr_pct=2.9,
            bb_upper=2_700.0,
            bb_lower=1_980.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.74,
            rsi_signal="oversold",
            rsi_value=29.0,
            macd_signal="bullish_cross",
            macd_histogram=0.32,
        ),
        composite=_composite(
            symbol="ETH",
            score=0.23,
            agreement=0.79,
            direction="buy",
            breakdown={"sentiment": 0.09, "ta": 0.08, "event": 0.04},
        ),
        verdict=_verdict(symbol="ETH", prevailing_view="bullish", confidence=0.82),
        event_data={"event_type": "protocol_upgrade", "remaining_drift_pct": 7.5},
        regime="risk_on",
        btc_change_pct=2.7,
    )

    assert signal is not None
    assert signal.tp1_pct <= SPOT_TP_CAPS["tp1_max_pct"] + 0.1
    assert signal.tp2_pct <= SPOT_TP_CAPS["tp2_max_pct"] + 0.1
    assert signal.tp3_pct <= SPOT_TP_CAPS["tp3_max_pct"] + 0.1


def test_btc_has_tighter_caps() -> None:
    btc_tp1, btc_tp2, btc_tp3, btc_stop = _cap_spot_targets(
        symbol="BTC",
        current_price=100.0,
        direction="LONG",
        tp1=112.0,
        tp2=125.0,
        tp3=140.0,
        stop_loss=86.0,
    )
    alt_tp1, alt_tp2, alt_tp3, alt_stop = _cap_spot_targets(
        symbol="ETH",
        current_price=100.0,
        direction="LONG",
        tp1=112.0,
        tp2=125.0,
        tp3=140.0,
        stop_loss=86.0,
    )

    assert btc_tp1 == 100.0 * (1 + BTC_TP_CAPS["tp1_max_pct"] / 100)
    assert btc_tp2 == 100.0 * (1 + BTC_TP_CAPS["tp2_max_pct"] / 100)
    assert btc_tp3 == 100.0 * (1 + BTC_TP_CAPS["tp3_max_pct"] / 100)
    assert btc_tp1 < alt_tp1
    assert btc_tp2 < alt_tp2
    assert btc_tp3 < alt_tp3
    assert btc_stop == alt_stop == 100.0 * (1 - SPOT_SL_CAPS["sl_max_pct"] / 100)


def test_sl_never_exceeds_8_percent() -> None:
    _tp1, _tp2, _tp3, stop_loss = _cap_spot_targets(
        symbol="SOL",
        current_price=100.0,
        direction="LONG",
        tp1=104.0,
        tp2=109.0,
        tp3=117.0,
        stop_loss=72.0,
    )

    assert stop_loss == 100.0 * (1 - SPOT_SL_CAPS["sl_max_pct"] / 100)


def test_rr_still_enforced_after_capping() -> None:
    signal = generate_crypto_trade_signal(
        symbol="BTC",
        current_price=100.0,
        ta_signal=_ta_signal(
            symbol="BTC",
            current_price=100.0,
            atr_value=4.0,
            atr_pct=4.0,
            bb_upper=135.0,
            bb_lower=92.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.75,
            rsi_signal="oversold",
            rsi_value=28.0,
            macd_signal="bullish_cross",
            macd_histogram=0.4,
        ),
        composite=_composite(
            symbol="BTC",
            score=0.24,
            agreement=0.8,
            direction="buy",
            breakdown={"sentiment": 0.1, "ta": 0.08, "event": 0.04},
        ),
        verdict=_verdict(symbol="BTC", prevailing_view="bullish", confidence=0.82),
        event_data={"event_type": "protocol_upgrade", "remaining_drift_pct": 6.0},
        regime="risk_on",
        btc_change_pct=3.0,
    )

    assert signal is None


def test_generate_crypto_trade_signal_blocks_risk_off_long_without_five_factors() -> None:
    signal = generate_crypto_trade_signal(
        symbol="SOL",
        current_price=180.0,
        ta_signal=_ta_signal(
            symbol="SOL",
            current_price=180.0,
            atr_value=8.0,
            atr_pct=4.4,
            trend="up",
            overall_signal="buy",
            confidence=0.68,
            rsi_value=44.0,
            macd_signal="bullish_cross",
            macd_histogram=0.3,
        ),
        composite=_composite(
            symbol="SOL",
            score=0.18,
            agreement=0.7,
            direction="buy",
            breakdown={"sentiment": 0.09, "ta": 0.06},
        ),
        verdict=_verdict(symbol="SOL", prevailing_view="bullish", confidence=0.75),
        event_data=None,
        regime="risk_off",
        btc_change_pct=0.0,
    )

    assert signal is None


def test_generate_crypto_trade_signal_skips_chaos_zone() -> None:
    signal = generate_crypto_trade_signal(
        symbol="PEPE",
        current_price=0.000012,
        ta_signal=_ta_signal(
            symbol="PEPE",
            current_price=0.000012,
            atr_value=0.0000025,
            atr_pct=20.5,
            bb_upper=0.000016,
            bb_lower=0.000009,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.74,
            rsi_value=24.0,
            macd_signal="bullish_cross",
            macd_histogram=0.4,
        ),
        composite=_composite(
            symbol="PEPE",
            score=0.3,
            agreement=0.82,
            direction="buy",
            breakdown={"sentiment": 0.1, "ta": 0.09},
        ),
        verdict=_verdict(symbol="PEPE", prevailing_view="bullish", confidence=0.8),
        event_data={"event_type": "exchange_listing", "remaining_drift_pct": 7.0},
        regime="risk_on",
        btc_change_pct=4.0,
    )

    assert signal is None


def test_generate_crypto_trade_signal_sets_high_vol_risk_and_flags_manipulation() -> None:
    signal = generate_crypto_trade_signal(
        symbol="LINK",
        current_price=18.0,
        ta_signal=_ta_signal(
            symbol="LINK",
            current_price=18.0,
            atr_value=1.4,
            atr_pct=12.0,
            bb_upper=21.0,
            bb_lower=15.0,
            bb_position="above_upper",
            bb_squeeze=True,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.7,
            rsi_signal="overbought",
            rsi_value=84.0,
            macd_signal="bullish_cross",
            macd_histogram=0.5,
        ),
        composite=_composite(
            symbol="LINK",
            score=0.26,
            agreement=0.76,
            direction="buy",
            breakdown={"sentiment": 0.1, "ta": -0.08, "event": 0.03},
        ),
        verdict=_verdict(symbol="LINK", prevailing_view="bullish", confidence=0.79),
        event_data={"event_type": "partnership", "remaining_drift_pct": 3.0},
        regime="volatile",
        btc_change_pct=2.8,
    )

    assert signal is not None
    assert signal.suggested_risk_pct == 1.0
    assert signal.confluence_count >= 3
    assert "possible distribution by whales" in signal.manipulation_check
    assert "Extreme volatility" in signal.manipulation_check
    assert "Breakout from squeeze" in signal.manipulation_check
    assert "Strong signals conflicting" in signal.manipulation_check
    assert "Volatile regime" in signal.manipulation_check
