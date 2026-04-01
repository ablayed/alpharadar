from __future__ import annotations

import numpy as np

from src.agents.state import DebateVerdict
from src.data.derivatives import CMEGap, FundingData, OpenInterestData
from src.data.macro import DXYData, FearGreedData
from src.ml import EnsemblePrediction
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
    close: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    open_: np.ndarray | None = None,
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
        _close=close,
        _high=high,
        _low=low,
        _volume=volume,
        _open=open_,
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


def _funding(
    *,
    symbol: str = "BTC",
    current_rate: float = 0.0012,
    z_score: float = 2.8,
    signal: str = "bearish",
    description: str = "Funding stretched",
) -> FundingData:
    return FundingData(
        symbol=symbol,
        current_rate=current_rate,
        rate_annualized=current_rate * 3 * 365 * 100,
        z_score=z_score,
        signal=signal,
        description=description,
    )


def _oi(
    *,
    symbol: str = "BTC",
    oi_change_4h_pct: float = 4.0,
    oi_change_24h_pct: float = 8.0,
    signal: str = "rising",
    description: str = "OI rising",
) -> OpenInterestData:
    return OpenInterestData(
        symbol=symbol,
        current_oi=1_000_000.0,
        oi_change_4h_pct=oi_change_4h_pct,
        oi_change_24h_pct=oi_change_24h_pct,
        signal=signal,
        description=description,
    )


def _fear_greed(
    *,
    value: int = 18,
    classification: str = "Fear",
    signal: str = "bullish",
    description: str = "Contrarian fear setup",
) -> FearGreedData:
    return FearGreedData(
        value=value,
        classification=classification,
        signal=signal,
        description=description,
    )


def _dxy(
    *,
    current: float = 102.4,
    change_7d_pct: float = -1.4,
    signal: str = "bullish",
    description: str = "Dollar falling",
) -> DXYData:
    return DXYData(
        current=current,
        change_7d_pct=change_7d_pct,
        btc_correlation_30d=-0.6,
        signal=signal,
        description=description,
    )


def _cme_gap(
    *,
    gap_low: float = 103.0,
    gap_high: float = 105.0,
    gap_direction: str = "up",
    distance_pct: float = 4.0,
    filled: bool = False,
    description: str = "Gap above",
) -> CMEGap:
    return CMEGap(
        gap_low=gap_low,
        gap_high=gap_high,
        gap_direction=gap_direction,
        distance_pct=distance_pct,
        filled=filled,
        description=description,
    )


def _ml_prediction(
    *,
    symbol: str = "BTC",
    direction: str = "bullish",
    confidence: float = 0.72,
    agreement: int = 2,
    description: str = "ML ensemble: 2/3 models predict upside (avg prob: 72%)",
) -> EnsemblePrediction:
    return EnsemblePrediction(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        xgb_prob=0.71,
        rf_prob=0.69,
        lr_prob=0.54,
        agreement=agreement,
        description=description,
        last_trained=1_700_000_000.0,
    )


def _fib_raw_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.array(
        [
            98.0,
            97.0,
            96.0,
            95.0,
            96.0,
            97.0,
            96.0,
            95.0,
            96.0,
            97.0,
            98.0,
            99.0,
            100.0,
            99.0,
            98.0,
            97.0,
            96.0,
            95.0,
            96.0,
            97.0,
            98.0,
            99.0,
            100.0,
            99.0,
            98.0,
            97.0,
            98.0,
            99.0,
            99.5,
            100.0,
        ],
        dtype=np.float64,
    )
    high = close + 1.0
    low = close - 1.0
    volume = np.full_like(close, 10.0)
    open_ = close - 0.5
    return close, high, low, volume, open_


def _factor_raw_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.array(
        [
            120.0,
            118.0,
            116.0,
            114.0,
            112.0,
            110.0,
            108.0,
            106.0,
            104.0,
            102.0,
            100.0,
            98.0,
            96.0,
            94.0,
            95.0,
            96.0,
            97.0,
            98.0,
            99.0,
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            104.0,
            103.0,
            93.0,
            94.0,
            94.5,
        ],
        dtype=np.float64,
    )
    high = close + 1.0
    low = close - 1.0
    low[-1] = 91.0
    volume = np.array(([100.0] * 9) + ([10.0] * 7) + ([1.0] * 13) + [6.0], dtype=np.float64)
    open_ = close - 0.5
    open_[-1] = 94.0
    high[-1] = 95.0
    return close, high, low, volume, open_


def _smc_raw_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.array(
        [
            96.0,
            97.0,
            98.0,
            99.0,
            100.0,
            109.0,
            104.0,
            103.0,
            102.0,
            101.0,
            104.0,
            100.0,
            111.0,
            107.0,
            106.0,
            105.0,
        ],
        dtype=np.float64,
    )
    open_ = np.array(
        [
            95.0,
            96.0,
            97.0,
            98.0,
            99.0,
            108.0,
            105.0,
            104.0,
            103.0,
            102.0,
            103.0,
            106.0,
            109.0,
            108.0,
            107.0,
            106.0,
        ],
        dtype=np.float64,
    )
    high = np.array(
        [
            97.0,
            98.0,
            99.0,
            100.0,
            101.0,
            110.0,
            105.0,
            104.0,
            103.0,
            102.0,
            105.0,
            106.0,
            112.0,
            108.0,
            107.0,
            106.0,
        ],
        dtype=np.float64,
    )
    low = np.array(
        [
            94.0,
            95.0,
            96.0,
            97.0,
            98.0,
            107.0,
            103.0,
            102.0,
            101.0,
            100.0,
            103.0,
            100.0,
            108.0,
            106.0,
            105.0,
            104.0,
        ],
        dtype=np.float64,
    )
    volume = np.full_like(close, 12.0)
    return close, high, low, volume, open_


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


def test_generate_crypto_trade_signal_uses_fibonacci_targets_and_hybrid_stop() -> None:
    close, high, low, volume, open_ = _fib_raw_series()

    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=100.0,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=100.0,
            atr_value=1.0,
            atr_pct=1.0,
            bb_upper=110.0,
            bb_lower=95.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.75,
            rsi_signal="oversold",
            rsi_value=29.0,
            macd_signal="bullish_cross",
            macd_histogram=0.3,
            close=close,
            high=high,
            low=low,
            volume=volume,
            open_=open_,
        ),
        composite=_composite(
            symbol="ETH",
            score=0.24,
            agreement=0.8,
            direction="buy",
            breakdown={"sentiment": 0.1, "ta": 0.08, "event": 0.04},
        ),
        verdict=_verdict(symbol="ETH", prevailing_view="bullish", confidence=0.82),
        event_data={"event_type": "protocol_upgrade", "remaining_drift_pct": 4.0},
        regime="risk_on",
        btc_change_pct=2.5,
    )

    assert signal is not None
    assert signal.stop_loss == 98.0
    assert signal.tp1 < 103.0
    assert signal.tp2 <= 105.0


def test_generate_crypto_trade_signal_adds_obv_volume_and_stop_hunt_factors() -> None:
    close, high, low, volume, open_ = _factor_raw_series()

    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=94.5,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=94.5,
            atr_value=1.0,
            atr_pct=1.1,
            bb_upper=100.0,
            bb_lower=91.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.78,
            rsi_signal="oversold",
            rsi_value=28.0,
            macd_signal="bullish_cross",
            macd_histogram=0.35,
            close=close,
            high=high,
            low=low,
            volume=volume,
            open_=open_,
        ),
        composite=_composite(
            symbol="ETH",
            score=0.25,
            agreement=0.82,
            direction="buy",
            breakdown={"sentiment": 0.1, "ta": 0.08, "event": 0.05},
        ),
        verdict=_verdict(symbol="ETH", prevailing_view="bullish", confidence=0.83),
        event_data={"event_type": "protocol_upgrade", "remaining_drift_pct": 4.5},
        regime="risk_on",
        btc_change_pct=2.8,
    )

    assert signal is not None
    factor_names = {factor.name for factor in signal.confluence_factors}
    assert "obv_divergence" in factor_names
    assert "volume" in factor_names
    assert "stop_hunt" in factor_names


def test_generate_crypto_trade_signal_adds_smc_factors() -> None:
    close, high, low, volume, open_ = _smc_raw_series()

    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=106.0,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=106.0,
            atr_value=2.5,
            atr_pct=2.4,
            bb_upper=118.0,
            bb_lower=99.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.79,
            rsi_signal="oversold",
            rsi_value=29.0,
            macd_signal="bullish_cross",
            macd_histogram=0.4,
            close=close,
            high=high,
            low=low,
            volume=volume,
            open_=open_,
        ),
        composite=_composite(
            symbol="ETH",
            score=0.24,
            agreement=0.82,
            direction="buy",
            breakdown={"sentiment": 0.09, "ta": 0.08, "event": 0.05},
        ),
        verdict=_verdict(symbol="ETH", prevailing_view="bullish", confidence=0.84),
        event_data={"event_type": "protocol_upgrade", "remaining_drift_pct": 4.0},
        regime="risk_on",
        btc_change_pct=2.0,
    )

    assert signal is not None
    factor_names = {factor.name for factor in signal.confluence_factors}
    assert "fair_value_gap" in factor_names
    assert "break_of_structure" in factor_names
    assert "order_block" in factor_names


def test_generate_crypto_trade_signal_adds_funding_and_oi_bearish_factors() -> None:
    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=2_500.0,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=2_500.0,
            atr_value=90.0,
            atr_pct=3.6,
            bb_upper=2_620.0,
            bb_lower=2_420.0,
            trend="strong_down",
            overall_signal="sell",
            confidence=0.74,
            rsi_value=72.0,
            rsi_signal="overbought",
            macd_signal="bearish_cross",
            macd_histogram=-0.3,
        ),
        composite=_composite(
            symbol="ETH",
            score=-0.24,
            agreement=0.8,
            direction="sell",
            breakdown={"sentiment": -0.08, "ta": -0.09, "event": -0.04},
        ),
        verdict=_verdict(symbol="ETH", prevailing_view="bearish", confidence=0.82),
        event_data={"event_type": "exploit", "remaining_drift_pct": -4.0},
        regime="risk_off",
        btc_change_pct=-3.0,
        funding=_funding(symbol="ETH", current_rate=0.0013, z_score=3.1, signal="bearish"),
        oi=_oi(symbol="ETH", oi_change_4h_pct=12.5, oi_change_24h_pct=9.2),
    )

    assert signal is not None
    assert signal.direction == "SHORT"
    factor_names = {factor.name for factor in signal.confluence_factors}
    assert "funding_rate" in factor_names
    assert "oi_divergence" in factor_names
    assert "Funding rate z-score at +3.1" in signal.manipulation_check
    assert "OI changed +12.5% in 4h" in signal.manipulation_check


def test_generate_crypto_trade_signal_adds_macro_and_cme_factors() -> None:
    signal = generate_crypto_trade_signal(
        symbol="BTC",
        current_price=100.0,
        ta_signal=_ta_signal(
            symbol="BTC",
            current_price=100.0,
            atr_value=2.0,
            atr_pct=2.0,
            bb_upper=112.0,
            bb_lower=96.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.78,
            rsi_signal="oversold",
            rsi_value=29.0,
            macd_signal="bullish_cross",
            macd_histogram=0.3,
        ),
        composite=_composite(
            symbol="BTC",
            score=0.24,
            agreement=0.82,
            direction="buy",
            breakdown={"sentiment": 0.09, "ta": 0.08, "event": 0.04},
        ),
        verdict=_verdict(symbol="BTC", prevailing_view="bullish", confidence=0.84),
        event_data={"event_type": "etf_inflow", "remaining_drift_pct": 4.0},
        regime="risk_on",
        btc_change_pct=0.0,
        cme_gap=_cme_gap(gap_low=103.0, gap_high=105.0, gap_direction="up"),
        fear_greed=_fear_greed(),
        dxy=_dxy(),
    )

    assert signal is not None
    assert signal.direction == "LONG"
    factor_names = {factor.name for factor in signal.confluence_factors}
    assert "cme_gap" in factor_names
    assert "fear_greed" in factor_names
    assert "dxy" in factor_names


def test_generate_crypto_trade_signal_adds_ml_ensemble_factor() -> None:
    signal = generate_crypto_trade_signal(
        symbol="ETH",
        current_price=2_500.0,
        ta_signal=_ta_signal(
            symbol="ETH",
            current_price=2_500.0,
            atr_value=90.0,
            atr_pct=3.6,
            bb_upper=2_620.0,
            bb_lower=2_420.0,
            trend="strong_up",
            overall_signal="buy",
            confidence=0.74,
            rsi_value=28.0,
            rsi_signal="oversold",
            macd_signal="bullish_cross",
            macd_histogram=0.3,
        ),
        composite=_composite(
            symbol="ETH",
            score=0.24,
            agreement=0.8,
            direction="buy",
            breakdown={"sentiment": 0.08, "ta": 0.07, "event": 0.04},
        ),
        verdict=_verdict(symbol="ETH", prevailing_view="bullish", confidence=0.82),
        event_data={"event_type": "protocol_upgrade", "remaining_drift_pct": 3.5},
        regime="risk_on",
        btc_change_pct=2.2,
        ml_prediction=_ml_prediction(symbol="ETH"),
    )

    assert signal is not None
    ml_factor = next(factor for factor in signal.confluence_factors if factor.name == "ml_ensemble")
    assert ml_factor.direction == "bullish"
    assert ml_factor.strength == 0.72
    assert "2/3 models predict upside" in ml_factor.description
