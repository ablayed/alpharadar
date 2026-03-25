from __future__ import annotations

import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "123")
os.environ.setdefault("FINNHUB_API_KEY", "test-key")
os.environ.setdefault("COINGECKO_API_KEY", "test-key")

from src.bot.handlers import (  # noqa: E402
    _format_correlation_pair_line,
    _format_lead_lag_line,
    _format_regime_text,
    _format_signals_text,
    _format_ta_text,
)
from src.correlation.models import LeadLagResult, MarketRegime, PairCorrelation, RegimeState
from src.signals.models import CompositeScore
from src.ta.signals import TASignal


def test_format_signals_text_uses_plain_english_summary() -> None:
    composite = CompositeScore(
        symbol="NVDA",
        score=0.09,
        confidence=0.6,
        direction="neutral",
        signal_breakdown={
            "sentiment": 0.08,
            "event": 0.068,
            "ta": -0.049,
            "correlation": -0.024,
        },
        agreement=0.5,
        regime_context="risk_off",
        summary="technical summary that should not be shown",
    )

    text = _format_signals_text("NVDA", composite)

    assert "Overall: Neutral" in text
    assert "News and sentiment lean positive." in text
    assert "Technical trend is still bearish with weak momentum." in text
    assert "Market mood:" in text
    assert "Signals are split 50/50" in text
    assert "Wait for clearer direction before acting." in text
    assert "+0.080" not in text
    assert "technical summary" not in text


def test_format_correlation_output_translates_relationships() -> None:
    pair = PairCorrelation(
        symbol_a="BTC",
        symbol_b="QQQ",
        corr_20d=-0.03,
        corr_60d=-0.04,
        corr_90d=-0.03,
        divergence=0.0,
        z_score=0.7,
        regime_signal="stable",
    )
    lead_lag = LeadLagResult(
        symbol_a="BTC",
        symbol_b="QQQ",
        peak_lag=4,
        peak_correlation=0.28,
        interpretation="BTC leads QQQ by 4 day(s) (corr: 0.28)",
    )

    line = _format_correlation_pair_line("BTC & tech stocks", pair)
    lead_lag_line = _format_lead_lag_line(lead_lag)

    assert "BTC &amp; tech stocks: moving independently right now" == line
    assert "BTC has tended to move before QQQ by about 4 day(s) recently." == lead_lag_line
    assert "corr:" not in lead_lag_line


def test_format_regime_text_shows_market_mood_without_metrics() -> None:
    regime = RegimeState(
        regime=MarketRegime.RISK_OFF,
        confidence=0.85,
        description="Risk-off: VIX elevated, broad selloff across stocks and crypto",
        btc_qqq_corr=-0.03,
        vix_level="high",
        divergence_speed=0.01,
        implication_stocks="Defensive positioning; expect continued selling pressure",
        implication_crypto="High contagion risk; crypto often amplifies equity drawdowns in risk-off",
    )

    text = _format_regime_text(regime)

    assert "MARKET MOOD: Fearful" in text
    assert "Investors are selling and moving to safety." in text
    assert "<b>Stocks:</b> Expect continued selling pressure." in text
    assert "<b>Crypto:</b> Crypto is at risk of following stocks lower" in text
    assert "Confidence:" not in text
    assert "VIX:" not in text
    assert "corr" not in text.lower()


def test_format_ta_text_hides_raw_indicators() -> None:
    signal = TASignal(
        symbol="BTC",
        timeframe="daily",
        current_price=69_871.0,
        rsi_value=37.8,
        rsi_signal="neutral",
        macd_line=12.0,
        macd_signal_line=10.0,
        macd_histogram=2.0,
        macd_signal="bullish_cross",
        bb_upper=100_135.45,
        bb_lower=55_167.95,
        bb_position="middle",
        bb_squeeze=False,
        atr_value=6_523.38,
        atr_pct=9.3,
        volatility="high",
        ema_20=76_052.8,
        ema_50=86_862.08,
        trend="strong_down",
        overall_signal="neutral",
        confidence=0.4,
        summary="BTC: MACD bullish, trend strong down. Signal: neutral (40%).",
    )

    text = _format_ta_text(signal)

    assert "<b>BTC</b>  Neutral" in text
    assert "Trend: Downtrend - price is below both key moving averages." in text
    assert "Momentum: Starting to recover, but the turn is not fully confirmed yet." in text
    assert "Volatility: High - expect larger swings in either direction." in text
    assert "Key levels: Support ~$55,168 | Resistance ~$76,053" in text
    assert "Overall the technical picture is mixed." in text
    assert "RSI" not in text
    assert "MACD" not in text
    assert "37.8" not in text
