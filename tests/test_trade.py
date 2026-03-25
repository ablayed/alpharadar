from __future__ import annotations

from src.agents.state import DebateVerdict
from src.signals.models import CompositeScore
from src.ta.signals import TASignal
from src.trade.generator import generate_trade_signal


def _ta_signal(
    *,
    symbol: str = "NVDA",
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
) -> TASignal:
    return TASignal(
        symbol=symbol,
        timeframe="daily",
        current_price=current_price,
        rsi_value=rsi_value,
        rsi_signal=rsi_signal,
        macd_line=1.0,
        macd_signal_line=0.8,
        macd_histogram=0.2,
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
    symbol: str = "NVDA",
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
    symbol: str = "NVDA",
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


def test_generate_trade_signal_builds_long_setup() -> None:
    signal = generate_trade_signal(
        symbol="NVDA",
        current_price=100.0,
        ta_signal=_ta_signal(),
        composite=_composite(),
        verdict=_verdict(),
        event_data={"event_type": "analyst_upgrade", "remaining_drift_pct": 6.0},
        regime="risk_on",
    )

    assert signal is not None
    assert signal.direction == "LONG"
    assert signal.confidence == 0.82
    assert signal.entry_low < signal.entry_high
    assert signal.stop_loss < signal.entry_low
    assert signal.tp1 < signal.tp2 < signal.tp3
    assert signal.risk_reward >= 1.5
    assert "AI verdict is bullish" in signal.reasoning
    assert signal.manipulation_check == "No manipulation flags detected."


def test_generate_trade_signal_returns_none_for_weak_setup() -> None:
    signal = generate_trade_signal(
        symbol="NVDA",
        current_price=100.0,
        ta_signal=_ta_signal(confidence=0.4, overall_signal="neutral", trend="neutral"),
        composite=_composite(score=0.05, agreement=0.4, direction="neutral"),
        verdict=None,
        event_data=None,
        regime="default",
    )

    assert signal is None


def test_generate_trade_signal_flags_manipulation_patterns() -> None:
    signal = generate_trade_signal(
        symbol="NVDA",
        current_price=100.0,
        ta_signal=_ta_signal(
            atr_pct=9.2,
            bb_position="above_upper",
            bb_squeeze=True,
            overall_signal="neutral",
            confidence=0.4,
            trend="down",
            macd_signal="bullish_cross",
        ),
        composite=_composite(
            score=0.18,
            agreement=0.65,
            direction="buy",
            breakdown={"sentiment": 0.10, "ta": -0.08, "correlation": -0.14},
        ),
        verdict=_verdict(confidence=0.78),
        event_data=None,
        regime="volatile",
    )

    assert signal is not None
    assert "Sentiment and price action disagree" in signal.manipulation_check
    assert "Volatility is unusually high" in signal.manipulation_check
    assert "Price just broke out of a tight range" in signal.manipulation_check
    assert "broader market is volatile" in signal.manipulation_check


def test_generate_trade_signal_builds_short_setup() -> None:
    signal = generate_trade_signal(
        symbol="BTC",
        current_price=50_000.0,
        ta_signal=_ta_signal(
            symbol="BTC",
            current_price=50_000.0,
            atr_value=1_200.0,
            atr_pct=5.0,
            bb_upper=52_500.0,
            bb_lower=45_000.0,
            trend="strong_down",
            overall_signal="sell",
            confidence=0.7,
            macd_signal="bearish",
        ),
        composite=_composite(
            symbol="BTC",
            score=-0.24,
            agreement=0.72,
            direction="sell",
            breakdown={"sentiment": -0.06, "ta": -0.12, "event": -0.04},
        ),
        verdict=_verdict(symbol="BTC", prevailing_view="bearish", confidence=0.8),
        event_data={"event_type": "regulatory_crypto", "remaining_drift_pct": -5.0},
        regime="risk_off",
    )

    assert signal is not None
    assert signal.direction == "SHORT"
    assert signal.entry_low < signal.entry_high
    assert signal.stop_loss > signal.entry_high
    assert signal.tp1 > signal.tp2 > signal.tp3
    assert signal.risk_reward >= 1.5
