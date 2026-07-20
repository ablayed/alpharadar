from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from src.agents.state import DebateVerdict
from src.data.derivatives import CMEGap, FundingData, OpenInterestData
from src.data.macro import DXYData, FearGreedData
from src.signals.models import CompositeScore
from src.ta.indicators import obv_divergence, volume_confirmation
from src.ta.signals import TASignal
from src.ta.swings import (
    compute_fibonacci,
    detect_break_of_structure,
    detect_fair_value_gaps,
    detect_order_blocks,
    detect_stop_hunt,
    get_last_swing,
)

if TYPE_CHECKING:
    from src.ml.ensemble import EnsemblePrediction

logger = logging.getLogger(__name__)

SPOT_TP_CAPS = {
    "tp1_max_pct": 5.0,
    "tp2_max_pct": 10.0,
    "tp3_max_pct": 18.0,
}

BTC_TP_CAPS = {
    "tp1_max_pct": 3.0,
    "tp2_max_pct": 6.0,
    "tp3_max_pct": 10.0,
}

SPOT_SL_CAPS = {
    "sl_max_pct": 8.0,
}

STOCK_TP_CAPS = {
    "tp1_max_pct": 3.0,
    "tp2_max_pct": 6.0,
    "tp3_max_pct": 10.0,
}

STOCK_SL_CAPS = {
    "sl_max_pct": 5.0,
}


@dataclass(slots=True)
class ConfluenceFactor:
    name: str
    direction: str
    strength: float
    description: str


@dataclass(slots=True)
class TradeSignal:
    symbol: str
    direction: str
    confidence: float
    entry_low: float
    entry_high: float
    stop_loss: float
    stop_loss_pct: float
    risk_per_unit: float
    tp1: float
    tp1_pct: float
    tp2: float
    tp2_pct: float
    tp3: float
    tp3_pct: float
    risk_reward: float
    timeframe: str
    reasoning: str
    manipulation_check: str
    confluence_count: int = 0
    confluence_factors: list[ConfluenceFactor] = field(default_factory=list)
    suggested_risk_pct: float = 2.0
    timestamp: float = field(default_factory=time.time)


def generate_trade_signal(
    *,
    symbol: str,
    current_price: float,
    ta_signal: TASignal | None,
    composite: CompositeScore | None,
    verdict: DebateVerdict | None,
    event_data: Mapping[str, object] | None = None,
    regime: str = "default",
) -> TradeSignal | None:
    score = composite.score if composite is not None else 0.0
    agreement = composite.agreement if composite is not None else 0.0
    strong_verdict = (
        verdict is not None
        and verdict.prevailing_view in {"bullish", "bearish"}
        and verdict.confidence > 0.7
    )

    if not strong_verdict and abs(score) < 0.10 and agreement < 0.5:
        return None

    direction: str | None = None
    confidence = 0.0
    if strong_verdict and verdict is not None:
        direction = "LONG" if verdict.prevailing_view == "bullish" else "SHORT"
        confidence = float(verdict.confidence)
    elif score > 0.10:
        direction = "LONG"
        confidence = min(0.85, 0.5 + abs(score))
    elif score < -0.10:
        direction = "SHORT"
        confidence = min(0.85, 0.5 + abs(score))
    else:
        return None

    if _has_strong_ta_conflict(direction, ta_signal):
        return None

    atr_value = (
        ta_signal.atr_value
        if ta_signal is not None and ta_signal.atr_value > 0
        else current_price * 0.03
    )
    sl_distance = max(atr_value * 1.5, current_price * 0.015)
    sl_distance = min(sl_distance, current_price * (STOCK_SL_CAPS["sl_max_pct"] / 100))

    if direction == "LONG":
        entry_high = current_price
        entry_low = max(current_price - (atr_value * 0.3), 0.0001)
        if ta_signal is not None:
            support_candidates = [entry_low]
            for support in (ta_signal.ema_20, ta_signal.bb_lower):
                if current_price - atr_value <= support <= current_price:
                    support_candidates.append(support)
            entry_low = max(min(support_candidates), 0.0001)
        stop_loss = _hybrid_stop_loss(
            current_price=current_price,
            direction=direction,
            sl_distance=sl_distance,
            ta_signal=ta_signal,
            sl_max_pct=STOCK_SL_CAPS["sl_max_pct"],
            min_price=0.0001,
        )
        tp1 = entry_high + (sl_distance * 1.5)
        tp2 = entry_high + (sl_distance * 2.5)
        tp3 = entry_high + (sl_distance * 4.0)
    else:
        entry_low = current_price
        entry_high = current_price + (atr_value * 0.3)
        if ta_signal is not None:
            resistance_candidates = [entry_high]
            for resistance in (ta_signal.ema_20, ta_signal.bb_upper):
                if current_price <= resistance <= current_price + atr_value:
                    resistance_candidates.append(resistance)
            entry_high = max(resistance_candidates)
        stop_loss = _hybrid_stop_loss(
            current_price=current_price,
            direction=direction,
            sl_distance=sl_distance,
            ta_signal=ta_signal,
            sl_max_pct=STOCK_SL_CAPS["sl_max_pct"],
            min_price=0.0001,
        )
        tp1 = entry_low - (sl_distance * 1.5)
        tp2 = entry_low - (sl_distance * 2.5)
        tp3 = entry_low - (sl_distance * 4.0)

    drift_pct = _event_float(event_data, "remaining_drift_pct")
    if drift_pct is not None:
        drift_target = current_price * (1 + (drift_pct / 100))
        if direction == "LONG" and drift_pct > 0 and tp1 < drift_target < tp3:
            tp2 = drift_target
        if direction == "SHORT" and drift_pct < 0 and tp3 < drift_target < tp1:
            tp2 = drift_target

    if ta_signal is not None:
        if direction == "LONG":
            tp3 = min(tp3, max(ta_signal.bb_upper, tp2))
        else:
            tp3 = max(tp3, min(ta_signal.bb_lower, tp2))

    tp1, tp2, tp3 = _apply_fibonacci_targets(
        current_price=current_price,
        direction=direction,
        ta_signal=ta_signal,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
    )

    entry_reference = entry_high if direction == "LONG" else entry_low
    tp1, tp2, tp3 = _normalize_targets(direction, entry_reference, sl_distance, tp1, tp2, tp3)
    tp1, tp2, tp3, stop_loss = _cap_stock_targets(
        symbol=symbol,
        current_price=current_price,
        direction=direction,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        stop_loss=stop_loss,
    )
    entry_low, entry_high = _rebuild_entry_zone(
        current_price=current_price,
        stop_loss=stop_loss,
        direction=direction,
    )
    entry_reference = _entry_reference_from_zone(
        direction=direction,
        entry_low=entry_low,
        entry_high=entry_high,
    )

    risk = abs(entry_reference - stop_loss)
    reward = abs(tp2 - entry_reference)
    if risk <= 0:
        return None
    risk_reward = reward / risk
    if risk_reward < 1.5:
        return None

    days_to_tp2 = reward / max(atr_value, 0.0001)
    if days_to_tp2 < 2:
        timeframe = "1-3 days"
    elif days_to_tp2 < 5:
        timeframe = "3-7 days"
    else:
        timeframe = "1-2 weeks"

    stop_loss_pct = (abs(current_price - stop_loss) / current_price) * 100 if current_price > 0 else 0.0
    tp1_pct = (abs(tp1 - current_price) / current_price) * 100 if current_price > 0 else 0.0
    tp2_pct = (abs(tp2 - current_price) / current_price) * 100 if current_price > 0 else 0.0
    tp3_pct = (abs(tp3 - current_price) / current_price) * 100 if current_price > 0 else 0.0

    return TradeSignal(
        symbol=symbol.upper(),
        direction=direction,
        confidence=round(confidence, 4),
        entry_low=round(entry_low, 4),
        entry_high=round(entry_high, 4),
        stop_loss=round(stop_loss, 4),
        stop_loss_pct=round(stop_loss_pct, 1),
        risk_per_unit=round(risk, 4),
        tp1=round(tp1, 4),
        tp1_pct=round(tp1_pct, 1),
        tp2=round(tp2, 4),
        tp2_pct=round(tp2_pct, 1),
        tp3=round(tp3, 4),
        tp3_pct=round(tp3_pct, 1),
        risk_reward=round(risk_reward, 1),
        timeframe=timeframe,
        reasoning=_build_reasoning(direction, regime, ta_signal, composite, verdict, event_data),
        manipulation_check=_check_manipulation(ta_signal, composite, regime),
    )


def generate_crypto_trade_signal(
    *,
    symbol: str,
    current_price: float,
    ta_signal: TASignal | None,
    composite: CompositeScore | None,
    verdict: DebateVerdict | None,
    event_data: Mapping[str, object] | None,
    regime: str,
    btc_change_pct: float = 0.0,
    funding: FundingData | None = None,
    oi: OpenInterestData | None = None,
    cme_gap: CMEGap | None = None,
    fear_greed: FearGreedData | None = None,
    dxy: DXYData | None = None,
    ml_prediction: EnsemblePrediction | None = None,
    breakout_score: int | None = None,
    breakout_direction: str | None = None,
    breakout_description: str | None = None,
) -> TradeSignal | None:
    if verdict is not None and verdict.prevailing_view == "neutral":
        logger.info("NEUTRAL_EXCLUDED_FROM_EXECUTION: %s", symbol.upper())
        return None

    factors: list[ConfluenceFactor] = []

    if ta_signal is not None:
        if ta_signal.trend in {"strong_up", "up"}:
            factors.append(
                ConfluenceFactor(
                    name="trend",
                    direction="bullish",
                    strength=0.8 if ta_signal.trend == "strong_up" else 0.6,
                    description="Price above key moving averages - uptrend intact",
                )
            )
        elif ta_signal.trend in {"strong_down", "down"}:
            factors.append(
                ConfluenceFactor(
                    name="trend",
                    direction="bearish",
                    strength=0.8 if ta_signal.trend == "strong_down" else 0.6,
                    description="Price below key moving averages - downtrend in place",
                )
            )

        rsi = ta_signal.rsi_value
        if rsi < 20:
            factors.append(
                ConfluenceFactor(
                    name="rsi_extreme",
                    direction="bullish",
                    strength=0.9,
                    description=f"RSI at {rsi:.0f} - deeply oversold, bounce likely",
                )
            )
        elif rsi < 35 and ta_signal.trend != "strong_down":
            factors.append(
                ConfluenceFactor(
                    name="rsi_oversold",
                    direction="bullish",
                    strength=0.6,
                    description=f"RSI at {rsi:.0f} - approaching oversold territory",
                )
            )
        elif rsi > 80:
            factors.append(
                ConfluenceFactor(
                    name="rsi_extreme",
                    direction="bearish",
                    strength=0.9,
                    description=f"RSI at {rsi:.0f} - deeply overbought, pullback likely",
                )
            )
        elif rsi > 65 and ta_signal.trend != "strong_up":
            factors.append(
                ConfluenceFactor(
                    name="rsi_overbought",
                    direction="bearish",
                    strength=0.6,
                    description=f"RSI at {rsi:.0f} - approaching overbought territory",
                )
            )

        if ta_signal.macd_signal == "bullish_cross":
            factors.append(
                ConfluenceFactor(
                    name="macd_cross",
                    direction="bullish",
                    strength=0.8,
                    description="MACD bullish crossover - momentum shifting positive",
                )
            )
        elif ta_signal.macd_signal == "bearish_cross":
            factors.append(
                ConfluenceFactor(
                    name="macd_cross",
                    direction="bearish",
                    strength=0.8,
                    description="MACD bearish crossover - momentum shifting negative",
                )
            )
        elif ta_signal.macd_signal == "bullish" and ta_signal.macd_histogram > 0:
            factors.append(
                ConfluenceFactor(
                    name="macd_momentum",
                    direction="bullish",
                    strength=0.5,
                    description="MACD positive - bullish momentum continues",
                )
            )
        elif ta_signal.macd_signal == "bearish" and ta_signal.macd_histogram < 0:
            factors.append(
                ConfluenceFactor(
                    name="macd_momentum",
                    direction="bearish",
                    strength=0.5,
                    description="MACD negative - bearish momentum continues",
                )
            )

        if ta_signal.bb_position == "below_lower":
            factors.append(
                ConfluenceFactor(
                    name="bollinger",
                    direction="bullish",
                    strength=0.7,
                    description="Price below lower Bollinger Band - mean reversion likely",
                )
            )
        elif ta_signal.bb_position == "above_upper":
            factors.append(
                ConfluenceFactor(
                    name="bollinger",
                    direction="bearish",
                    strength=0.7,
                    description="Price above upper Bollinger Band - overextended",
                )
            )

        if ta_signal.bb_squeeze:
            squeeze_direction = "bullish" if ta_signal.macd_histogram >= 0 else "bearish"
            factors.append(
                ConfluenceFactor(
                    name="squeeze",
                    direction=squeeze_direction,
                    strength=0.6,
                    description="Bollinger squeeze - big move imminent, direction from MACD",
                )
            )

        if ta_signal._close is not None and ta_signal._volume is not None and len(ta_signal._volume) > 0:
            divergence = obv_divergence(ta_signal._close, ta_signal._volume)
            if divergence == "bullish_divergence":
                factors.append(
                    ConfluenceFactor(
                        name="obv_divergence",
                        direction="bullish",
                        strength=0.8,
                        description="Volume accumulation despite price decline - smart money buying",
                    )
                )
            elif divergence == "bearish_divergence":
                factors.append(
                    ConfluenceFactor(
                        name="obv_divergence",
                        direction="bearish",
                        strength=0.8,
                        description="Volume distribution despite price rise - smart money selling",
                    )
                )

            volume_confirmed, volume_ratio = volume_confirmation(ta_signal._volume)
            if volume_confirmed:
                factors.append(
                    ConfluenceFactor(
                        name="volume",
                        direction=_trend_bias_direction(ta_signal),
                        strength=min(0.7, volume_ratio * 0.3),
                        description=f"Volume {volume_ratio:.1f}x above average - confirms the move",
                    )
                )

        latest_hunt = _latest_stop_hunt(ta_signal)
        if latest_hunt is not None:
            factors.append(
                ConfluenceFactor(
                    name="stop_hunt",
                    direction=str(latest_hunt["direction_after"]),
                    strength=0.7,
                    description=str(latest_hunt["description"]),
                )
            )

        if (
            ta_signal._high is not None
            and ta_signal._low is not None
            and ta_signal._close is not None
            and ta_signal._open is not None
        ):
            fvgs = detect_fair_value_gaps(
                ta_signal._high,
                ta_signal._low,
                ta_signal._close,
                ta_signal._open,
            )
            recent_fvgs = [
                gap for gap in fvgs if int(gap["index"]) > len(ta_signal._close) - 20
            ]
            for fvg in sorted(recent_fvgs, key=lambda gap: int(gap["index"]), reverse=True):
                if not (float(fvg["bottom"]) <= current_price <= float(fvg["top"])):
                    continue
                factors.append(
                    ConfluenceFactor(
                        name="fair_value_gap",
                        direction="bullish" if fvg["type"] == "bullish_fvg" else "bearish",
                        strength=0.7,
                        description=(
                            "Price is trading inside a recent bullish FVG - liquidity void can "
                            "support a push higher"
                            if fvg["type"] == "bullish_fvg"
                            else "Price is trading inside a recent bearish FVG - liquidity void "
                            "can pull price lower"
                        ),
                    )
                )
                break

            bos_events = detect_break_of_structure(
                ta_signal._high,
                ta_signal._low,
                ta_signal._close,
            )
            recent_bos = [
                bos for bos in bos_events if int(bos["index"]) > len(ta_signal._close) - 5
            ]
            if recent_bos:
                latest_bos = recent_bos[-1]
                factors.append(
                    ConfluenceFactor(
                        name="break_of_structure",
                        direction="bullish" if latest_bos["type"] == "bullish_bos" else "bearish",
                        strength=0.85,
                        description=str(latest_bos["description"]),
                    )
                )

            order_blocks = detect_order_blocks(
                ta_signal._high,
                ta_signal._low,
                ta_signal._close,
                ta_signal._open,
            )
            for block in sorted(order_blocks, key=lambda ob: int(ob["index"]), reverse=True):
                if not (float(block["bottom"]) <= current_price <= float(block["top"])):
                    continue
                factors.append(
                    ConfluenceFactor(
                        name="order_block",
                        direction="bullish" if block["type"] == "bullish_ob" else "bearish",
                        strength=0.75,
                        description=str(block["description"]),
                    )
                )
                break

    event_type = str(event_data.get("event_type", "")) if event_data is not None else ""
    remaining = _event_float(event_data, "remaining_drift_pct") or 0.0
    if event_type and event_type not in {"general_news", "unknown", "none"}:
        if remaining > 1.0:
            factors.append(
                ConfluenceFactor(
                    name="event_car",
                    direction="bullish",
                    strength=min(0.8, remaining / 5),
                    description=(
                        f"Event ({event_type}) with {remaining:+.1f}% remaining upside based on historical data"
                    ),
                )
            )
        elif remaining < -1.0:
            factors.append(
                ConfluenceFactor(
                    name="event_car",
                    direction="bearish",
                    strength=min(0.8, abs(remaining) / 5),
                    description=(
                        f"Event ({event_type}) with {remaining:+.1f}% remaining downside based on historical data"
                    ),
                )
            )

    if composite is not None:
        sentiment = composite.signal_breakdown.get("sentiment", 0.0)
        if sentiment > 0.05:
            factors.append(
                ConfluenceFactor(
                    name="sentiment",
                    direction="bullish",
                    strength=min(0.7, sentiment * 5),
                    description="Positive news sentiment supporting the move",
                )
            )
        elif sentiment < -0.05:
            factors.append(
                ConfluenceFactor(
                    name="sentiment",
                    direction="bearish",
                    strength=min(0.7, abs(sentiment) * 5),
                    description="Negative news sentiment adding selling pressure",
                )
            )

    if verdict is not None and verdict.confidence > 0.6:
        if verdict.prevailing_view == "bullish":
            factors.append(
                ConfluenceFactor(
                    name="ai_verdict",
                    direction="bullish",
                    strength=verdict.confidence,
                    description=f"AI analysis concludes bullish ({verdict.confidence:.0%} confidence)",
                )
            )
        elif verdict.prevailing_view == "bearish":
            factors.append(
                ConfluenceFactor(
                    name="ai_verdict",
                    direction="bearish",
                    strength=verdict.confidence,
                    description=f"AI analysis concludes bearish ({verdict.confidence:.0%} confidence)",
                )
            )

    if regime == "risk_on":
        factors.append(
            ConfluenceFactor(
                name="regime",
                direction="bullish",
                strength=0.5,
                description="Risk-on market environment - favorable for crypto longs",
            )
        )
    elif regime == "risk_off":
        factors.append(
            ConfluenceFactor(
                name="regime",
                direction="bearish",
                strength=0.6,
                description="Risk-off environment - headwinds for crypto",
            )
        )

    if symbol.upper() != "BTC" and abs(btc_change_pct) > 1:
        if btc_change_pct > 2:
            factors.append(
                ConfluenceFactor(
                    name="btc_tailwind",
                    direction="bullish",
                    strength=0.5,
                    description=f"BTC up {btc_change_pct:+.1f}% - broad crypto tailwind",
                )
            )
        elif btc_change_pct < -2:
            factors.append(
                ConfluenceFactor(
                    name="btc_headwind",
                    direction="bearish",
                    strength=0.5,
                    description=f"BTC down {btc_change_pct:+.1f}% - broad crypto headwind",
                )
            )

    if funding is not None and funding.signal != "neutral":
        factors.append(
            ConfluenceFactor(
                name="funding_rate",
                direction=funding.signal,
                strength=min(0.85, 0.5 + (abs(funding.z_score) * 0.1)),
                description=funding.description,
            )
        )

    if oi is not None and ta_signal is not None:
        if ta_signal.trend in {"strong_up", "up"}:
            price_dir = "up"
        elif ta_signal.trend in {"strong_down", "down"}:
            price_dir = "down"
        else:
            price_dir = "flat"
        oi_dir = "rising" if oi.oi_change_24h_pct > 3 else "falling" if oi.oi_change_24h_pct < -3 else "flat"

        if oi_dir == "rising" and price_dir == "down":
            factors.append(
                ConfluenceFactor(
                    name="oi_divergence",
                    direction="bearish",
                    strength=0.7,
                    description=(
                        f"OI rising ({oi.oi_change_24h_pct:+.1f}%) while price trends down - "
                        "fragile longs can become liquidation fuel"
                    ),
                )
            )
        elif oi_dir == "falling" and price_dir == "down":
            factors.append(
                ConfluenceFactor(
                    name="oi_divergence",
                    direction="bullish",
                    strength=0.7,
                    description=(
                        f"OI falling ({oi.oi_change_24h_pct:+.1f}%) while price trends down - "
                        "capitulation may be forming"
                    ),
                )
            )
        elif oi_dir == "rising" and price_dir == "up":
            factors.append(
                ConfluenceFactor(
                    name="oi_confirmation",
                    direction="bullish",
                    strength=0.5,
                    description=(
                        f"OI rising ({oi.oi_change_24h_pct:+.1f}%) with price - "
                        "fresh money is confirming the move"
                    ),
                )
            )

    if symbol.upper() == "BTC" and cme_gap is not None and not cme_gap.filled:
        gap_mid = (cme_gap.gap_low + cme_gap.gap_high) / 2
        if current_price > gap_mid:
            factors.append(
                ConfluenceFactor(
                    name="cme_gap",
                    direction="bearish",
                    strength=0.5,
                    description=(
                        f"Unfilled CME gap below at ${cme_gap.gap_low:,.0f}-${cme_gap.gap_high:,.0f} "
                        "can pull price lower"
                    ),
                )
            )
        elif current_price < gap_mid:
            factors.append(
                ConfluenceFactor(
                    name="cme_gap",
                    direction="bullish",
                    strength=0.5,
                    description=(
                        f"Unfilled CME gap above at ${cme_gap.gap_low:,.0f}-${cme_gap.gap_high:,.0f} "
                        "can act as a bullish magnet"
                    ),
                )
            )

    if fear_greed is not None and fear_greed.signal != "neutral":
        factors.append(
            ConfluenceFactor(
                name="fear_greed",
                direction=fear_greed.signal,
                strength=0.6 if fear_greed.value <= 10 or fear_greed.value >= 90 else 0.4,
                description=fear_greed.description,
            )
        )

    if dxy is not None and dxy.signal != "neutral":
        factors.append(
            ConfluenceFactor(
                name="dxy",
                direction=dxy.signal,
                strength=0.4,
                description=dxy.description,
            )
        )

    if ml_prediction is not None and ml_prediction.direction != "neutral":
        factors.append(
            ConfluenceFactor(
                name="ml_ensemble",
                direction=ml_prediction.direction,
                strength=min(0.8, max(0.0, ml_prediction.confidence)),
                description=ml_prediction.description,
            )
        )

    if breakout_score is not None and breakout_score >= 50:
        if breakout_direction == "long_setup":
            factors.append(
                ConfluenceFactor(
                    name="breakout_score",
                    direction="bullish",
                    strength=min(0.9, breakout_score / 100),
                    description=(
                        f"Breakout score: {breakout_score}/100 - "
                        f"{breakout_description or 'multiple bullish signals converging'}"
                    ),
                )
            )
        elif breakout_direction == "short_setup":
            factors.append(
                ConfluenceFactor(
                    name="breakout_score",
                    direction="bearish",
                    strength=min(0.9, breakout_score / 100),
                    description=(
                        f"Breakdown score: {breakout_score}/100 - "
                        f"{breakout_description or 'distribution signals detected'}"
                    ),
                )
            )

    bullish_factors = [factor for factor in factors if factor.direction == "bullish"]
    bearish_factors = [factor for factor in factors if factor.direction == "bearish"]
    bullish_score = sum(factor.strength for factor in bullish_factors)
    bearish_score = sum(factor.strength for factor in bearish_factors)

    direction: str
    active_factors: list[ConfluenceFactor]
    opposing_factors: list[ConfluenceFactor]
    if len(bullish_factors) >= 3 and bullish_score > (bearish_score * 1.3):
        direction = "LONG"
        active_factors = bullish_factors
        opposing_factors = bearish_factors
    elif len(bearish_factors) >= 3 and bearish_score > (bullish_score * 1.3):
        direction = "SHORT"
        active_factors = bearish_factors
        opposing_factors = bullish_factors
    else:
        logger.debug(
            "Skipping %s: insufficient confluence (%d bullish, %d bearish)",
            symbol.upper(),
            len(bullish_factors),
            len(bearish_factors),
        )
        return None

    if direction == "LONG" and regime == "risk_off" and len(active_factors) < 5:
        return None

    atr = ta_signal.atr_value if ta_signal is not None and ta_signal.atr_value > 0 else current_price * 0.05
    atr_pct = ta_signal.atr_pct if ta_signal is not None else 5.0
    if atr_pct > 15:
        return None

    atr_multiplier = 2.5 if atr_pct > 6 else 2.0
    sl_distance = atr * atr_multiplier
    sl_distance = min(sl_distance, current_price * (SPOT_SL_CAPS["sl_max_pct"] / 100))

    if direction == "LONG":
        entry_high = current_price
        entry_low = max(current_price - (atr * 0.3), 0.0000001)
        if ta_signal is not None and current_price - atr <= ta_signal.ema_20 <= current_price:
            entry_low = min(entry_low, ta_signal.ema_20)
        stop_loss = _hybrid_stop_loss(
            current_price=current_price,
            direction=direction,
            sl_distance=sl_distance,
            ta_signal=ta_signal,
            sl_max_pct=SPOT_SL_CAPS["sl_max_pct"],
            min_price=0.0000001,
        )
        tp1 = entry_high + (sl_distance * 1.5)
        tp2 = entry_high + (sl_distance * 2.5)
        tp3 = entry_high + (sl_distance * 4.0)

        if remaining > 1:
            drift_target = current_price * (1 + remaining / 100)
            if tp1 < drift_target < tp3:
                tp2 = max(tp2, drift_target)

        if (
            ta_signal is not None
            and ta_signal.bb_upper > tp2
            and ta_signal.bb_upper < tp3
        ):
            tp3 = min(tp3, ta_signal.bb_upper * 1.02)
    else:
        entry_low = current_price
        entry_high = current_price + (atr * 0.3)
        if ta_signal is not None and current_price <= ta_signal.ema_20 <= current_price + atr:
            entry_high = max(entry_high, ta_signal.ema_20)
        stop_loss = _hybrid_stop_loss(
            current_price=current_price,
            direction=direction,
            sl_distance=sl_distance,
            ta_signal=ta_signal,
            sl_max_pct=SPOT_SL_CAPS["sl_max_pct"],
            min_price=0.0000001,
        )
        tp1 = entry_low - (sl_distance * 1.5)
        tp2 = entry_low - (sl_distance * 2.5)
        tp3 = entry_low - (sl_distance * 4.0)

        if remaining < -1:
            drift_target = current_price * (1 + remaining / 100)
            if tp3 < drift_target < tp1:
                tp2 = min(tp2, drift_target)

        if ta_signal is not None and ta_signal.bb_lower < tp2 and ta_signal.bb_lower > tp3:
            tp3 = max(tp3, ta_signal.bb_lower * 0.98)

    tp1, tp2, tp3 = _apply_fibonacci_targets(
        current_price=current_price,
        direction=direction,
        ta_signal=ta_signal,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
    )

    entry_reference = current_price
    tp1, tp2, tp3 = _normalize_targets(direction, entry_reference, sl_distance, tp1, tp2, tp3)
    tp1, tp2, tp3, stop_loss = _cap_spot_targets(
        symbol=symbol,
        current_price=current_price,
        direction=direction,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        stop_loss=stop_loss,
    )
    entry_low, entry_high = _rebuild_entry_zone(
        current_price=current_price,
        stop_loss=stop_loss,
        direction=direction,
    )
    entry_reference = _entry_reference_from_zone(
        direction=direction,
        entry_low=entry_low,
        entry_high=entry_high,
    )
    risk = abs(entry_reference - stop_loss)
    reward = abs(tp2 - entry_reference)
    if risk <= 0:
        return None
    risk_reward = reward / risk
    if risk_reward < 1.5:
        return None

    days_to_tp2 = reward / max(atr, 0.0001)
    if days_to_tp2 < 2:
        timeframe = "1-3 days"
    elif days_to_tp2 < 5:
        timeframe = "3-7 days"
    else:
        timeframe = "1-2 weeks"

    suggested_risk_pct = 1.0 if atr_pct > 6 else 2.0
    confidence = min(
        0.90,
        0.40 + (len(active_factors) * 0.08) + (sum(f.strength for f in active_factors) * 0.05),
    )
    stop_loss_pct = abs(current_price - stop_loss) / current_price * 100
    tp1_pct = abs(tp1 - current_price) / current_price * 100
    tp2_pct = abs(tp2 - current_price) / current_price * 100
    tp3_pct = abs(tp3 - current_price) / current_price * 100
    manipulation = _check_manipulation_v2(
        ta_signal,
        composite,
        regime,
        factors,
        oi=oi,
        funding=funding,
    )

    top_reasons = sorted(active_factors, key=lambda factor: factor.strength, reverse=True)[:3]
    reasoning = ". ".join(factor.description for factor in top_reasons) + "."
    if opposing_factors:
        top_risk = max(opposing_factors, key=lambda factor: factor.strength)
        reasoning += f" Main risk: {top_risk.description}."

    precision = _decimals(current_price)
    return TradeSignal(
        symbol=symbol.upper(),
        direction=direction,
        confidence=round(confidence, 4),
        entry_low=round(entry_low, precision),
        entry_high=round(entry_high, precision),
        stop_loss=round(stop_loss, precision),
        stop_loss_pct=round(stop_loss_pct, 1),
        risk_per_unit=round(risk, precision),
        tp1=round(tp1, precision),
        tp1_pct=round(tp1_pct, 1),
        tp2=round(tp2, precision),
        tp2_pct=round(tp2_pct, 1),
        tp3=round(tp3, precision),
        tp3_pct=round(tp3_pct, 1),
        risk_reward=round(risk_reward, 1),
        timeframe=timeframe,
        reasoning=reasoning,
        manipulation_check=manipulation,
        confluence_count=len(active_factors),
        confluence_factors=sorted(active_factors, key=lambda factor: factor.strength, reverse=True),
        suggested_risk_pct=suggested_risk_pct,
    )


def _cap_spot_targets(
    *,
    symbol: str,
    current_price: float,
    direction: str,
    tp1: float,
    tp2: float,
    tp3: float,
    stop_loss: float,
) -> tuple[float, float, float, float]:
    caps = BTC_TP_CAPS if symbol.upper() == "BTC" else SPOT_TP_CAPS
    return _cap_targets(
        current_price=current_price,
        direction=direction,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        stop_loss=stop_loss,
        tp_caps=caps,
        sl_max_pct=SPOT_SL_CAPS["sl_max_pct"],
    )


def _cap_stock_targets(
    *,
    symbol: str,
    current_price: float,
    direction: str,
    tp1: float,
    tp2: float,
    tp3: float,
    stop_loss: float,
) -> tuple[float, float, float, float]:
    del symbol
    return _cap_targets(
        current_price=current_price,
        direction=direction,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        stop_loss=stop_loss,
        tp_caps=STOCK_TP_CAPS,
        sl_max_pct=STOCK_SL_CAPS["sl_max_pct"],
    )


def _cap_targets(
    *,
    current_price: float,
    direction: str,
    tp1: float,
    tp2: float,
    tp3: float,
    stop_loss: float,
    tp_caps: Mapping[str, float],
    sl_max_pct: float,
) -> tuple[float, float, float, float]:
    if direction == "LONG":
        tp1 = min(tp1, current_price * (1 + tp_caps["tp1_max_pct"] / 100))
        tp2 = min(tp2, current_price * (1 + tp_caps["tp2_max_pct"] / 100))
        tp3 = min(tp3, current_price * (1 + tp_caps["tp3_max_pct"] / 100))
        stop_loss = max(stop_loss, current_price * (1 - sl_max_pct / 100))
        return tp1, tp2, tp3, stop_loss

    tp1 = max(tp1, current_price * (1 - tp_caps["tp1_max_pct"] / 100))
    tp2 = max(tp2, current_price * (1 - tp_caps["tp2_max_pct"] / 100))
    tp3 = max(tp3, current_price * (1 - tp_caps["tp3_max_pct"] / 100))
    stop_loss = min(stop_loss, current_price * (1 + sl_max_pct / 100))
    return tp1, tp2, tp3, stop_loss


def _rebuild_entry_zone(
    *,
    current_price: float,
    stop_loss: float,
    direction: str,
) -> tuple[float, float]:
    risk_distance = abs(current_price - stop_loss)
    entry_buffer = risk_distance * 0.1
    if direction == "LONG":
        return max(current_price - entry_buffer, 0.0000001), current_price
    return current_price, current_price + entry_buffer


def _entry_reference_from_zone(
    *,
    direction: str,
    entry_low: float,
    entry_high: float,
) -> float:
    return entry_low if direction == "LONG" else entry_high


def _apply_fibonacci_targets(
    *,
    current_price: float,
    direction: str,
    ta_signal: TASignal | None,
    tp1: float,
    tp2: float,
    tp3: float,
) -> tuple[float, float, float]:
    fib_targets = _fibonacci_targets_from_ta(ta_signal, direction)
    if fib_targets is None:
        return tp1, tp2, tp3

    fib_tp1, fib_tp2, fib_tp3 = fib_targets
    if direction == "LONG":
        if fib_tp1 > current_price:
            tp1 = min(tp1, fib_tp1)
        if fib_tp2 > current_price:
            tp2 = min(tp2, fib_tp2)
        if fib_tp3 > current_price:
            tp3 = min(tp3, fib_tp3)
        return tp1, tp2, tp3

    if fib_tp1 < current_price:
        tp1 = max(tp1, fib_tp1)
    if fib_tp2 < current_price:
        tp2 = max(tp2, fib_tp2)
    if fib_tp3 < current_price:
        tp3 = max(tp3, fib_tp3)
    return tp1, tp2, tp3


def _fibonacci_targets_from_ta(
    ta_signal: TASignal | None,
    direction: str,
) -> tuple[float, float, float] | None:
    if ta_signal is None or ta_signal._high is None or ta_signal._low is None:
        return None

    last_high, last_low = get_last_swing(ta_signal._high, ta_signal._low, order=5)
    if last_high is None or last_low is None or last_high.price <= last_low.price:
        return None

    fib = compute_fibonacci(
        last_high.price,
        last_low.price,
        "up" if direction == "LONG" else "down",
    )
    return fib.ext_1272, fib.ext_1618, fib.ext_2618


def _hybrid_stop_loss(
    *,
    current_price: float,
    direction: str,
    sl_distance: float,
    ta_signal: TASignal | None,
    sl_max_pct: float,
    min_price: float,
) -> float:
    atr_stop = current_price - sl_distance if direction == "LONG" else current_price + sl_distance
    structure_stop = _structure_stop_from_ta(ta_signal, direction)
    pct_cap = (
        current_price * (1 - sl_max_pct / 100)
        if direction == "LONG"
        else current_price * (1 + sl_max_pct / 100)
    )

    if direction == "LONG":
        candidates = [
            stop
            for stop in (atr_stop, structure_stop, pct_cap)
            if stop is not None and min_price <= stop < current_price
        ]
        if not candidates:
            return max(current_price * (1 - sl_max_pct / 100), min_price)
        return max(candidates)

    candidates = [
        stop
        for stop in (atr_stop, structure_stop, pct_cap)
        if stop is not None and stop > current_price
    ]
    if not candidates:
        return current_price * (1 + sl_max_pct / 100)
    return min(candidates)


def _structure_stop_from_ta(ta_signal: TASignal | None, direction: str) -> float | None:
    if ta_signal is None or ta_signal._high is None or ta_signal._low is None:
        return None

    last_high, last_low = get_last_swing(ta_signal._high, ta_signal._low, order=5)
    if direction == "LONG" and last_low is not None:
        return last_low.price - (last_low.price * 0.005)
    if direction == "SHORT" and last_high is not None:
        return last_high.price + (last_high.price * 0.005)
    return None


def _latest_stop_hunt(ta_signal: TASignal | None) -> dict[str, float | int | str] | None:
    if (
        ta_signal is None
        or ta_signal._high is None
        or ta_signal._low is None
        or ta_signal._close is None
        or ta_signal._open is None
    ):
        return None

    hunts = detect_stop_hunt(
        ta_signal._high,
        ta_signal._low,
        ta_signal._close,
        ta_signal._open,
    )
    if not hunts:
        return None

    latest = hunts[-1]
    if int(latest["index"]) < len(ta_signal._close) - 3:
        return None
    return latest


def _trend_bias_direction(ta_signal: TASignal) -> str:
    if ta_signal.trend in {"up", "strong_up"}:
        return "bullish"
    if ta_signal.trend in {"down", "strong_down"}:
        return "bearish"
    return "bullish" if ta_signal.macd_histogram >= 0 else "bearish"


def _has_strong_ta_conflict(direction: str, ta_signal: TASignal | None) -> bool:
    if ta_signal is None:
        return False
    if direction == "LONG" and ta_signal.overall_signal in {"sell", "strong_sell"} and ta_signal.confidence >= 0.6:
        return True
    if direction == "SHORT" and ta_signal.overall_signal in {"buy", "strong_buy"} and ta_signal.confidence >= 0.6:
        return True
    return False


def _normalize_targets(
    direction: str,
    entry_reference: float,
    risk: float,
    tp1: float,
    tp2: float,
    tp3: float,
) -> tuple[float, float, float]:
    min_step = max(risk * 0.25, entry_reference * 0.002, 0.0000001)
    if direction == "LONG":
        targets = sorted([tp1, tp2, tp3])
        first = max(targets[0], entry_reference + min_step)
        second = max(targets[1], first + min_step)
        third = max(targets[2], second + min_step)
        return first, second, third

    targets = sorted([tp1, tp2, tp3], reverse=True)
    first = min(targets[0], entry_reference - min_step)
    second = min(targets[1], first - min_step)
    third = min(targets[2], second - min_step)
    return first, second, third


def _build_reasoning(
    direction: str,
    regime: str,
    ta_signal: TASignal | None,
    composite: CompositeScore | None,
    verdict: DebateVerdict | None,
    event_data: Mapping[str, object] | None,
) -> str:
    reasons: list[str] = []
    if verdict is not None and verdict.prevailing_view in {"bullish", "bearish", "neutral"}:
        reasons.append(f"AI verdict is {verdict.prevailing_view} ({verdict.confidence:.0%})")

    event_type = str(event_data.get("event_type", "")) if event_data is not None else ""
    drift_pct = _event_float(event_data, "remaining_drift_pct")
    if event_type and event_type not in {"general_news", "unknown", "none"}:
        if drift_pct is not None:
            reasons.append(
                f"{event_type.replace('_', ' ')} still shows {drift_pct:+.1f}% remaining drift"
            )
        else:
            reasons.append(f"{event_type.replace('_', ' ')} is still influencing the setup")

    if composite is not None:
        if composite.direction in {"strong_buy", "buy"}:
            reasons.append("broader signal stack is leaning bullish")
        elif composite.direction in {"strong_sell", "sell"}:
            reasons.append("broader signal stack is leaning bearish")
        else:
            reasons.append("broader signal stack is still mixed")

    if ta_signal is not None:
        reasons.append(f"trend is {ta_signal.trend.replace('_', ' ')}")
        if ta_signal.rsi_signal == "oversold":
            reasons.append("momentum is washed out and could bounce")
        elif ta_signal.rsi_signal == "overbought":
            reasons.append("momentum is stretched and pullback risk is elevated")
        elif ta_signal.macd_signal == "bullish_cross":
            reasons.append("momentum is starting to turn up")
        elif ta_signal.macd_signal == "bearish_cross":
            reasons.append("momentum is starting to roll over")
        if ta_signal.bb_squeeze:
            reasons.append("volatility is compressed so a larger move may be close")

    reasons.append(f"market regime is {regime.replace('_', ' ')}")
    prefix = "Long setup because " if direction == "LONG" else "Short setup because "
    return prefix + ", ".join(reasons) + "."


def _check_manipulation(
    ta_signal: TASignal | None,
    composite: CompositeScore | None,
    regime: str,
) -> str:
    flags: list[str] = []
    if composite is not None:
        sentiment = composite.signal_breakdown.get("sentiment", 0.0)
        technical = composite.signal_breakdown.get("ta", 0.0)
        correlation = composite.signal_breakdown.get("correlation", 0.0)
        if sentiment > 0.05 and technical < -0.05:
            flags.append(
                "Sentiment and price action disagree: positive news but weak tape can mean distribution."
            )
        elif sentiment < -0.05 and technical > 0.05:
            flags.append(
                "Sentiment and price action disagree: negative news with firm tape can mean accumulation."
            )
        if abs(correlation) > 0.12 and abs(technical) > 0.05 and (correlation * technical) < 0:
            flags.append("Cross-market behavior is diverging from price action, which can be a warning sign.")

    if ta_signal is not None:
        if ta_signal.atr_pct > 8:
            flags.append("Volatility is unusually high, so the move may be event-driven or easily whipsawed.")
        if ta_signal.bb_squeeze and ta_signal.bb_position in {"above_upper", "below_lower"}:
            flags.append("Price just broke out of a tight range. Confirm follow-through before sizing up.")
        latest_hunt = _latest_stop_hunt(ta_signal)
        if latest_hunt is not None:
            flags.append(str(latest_hunt["description"]))

    if regime == "volatile":
        flags.append("The broader market is volatile. Smaller size and more patience make sense here.")

    return "No manipulation flags detected." if not flags else "\n".join(flags)


def _check_manipulation_v2(
    ta_signal: TASignal | None,
    composite: CompositeScore | None,
    regime: str,
    factors: list[ConfluenceFactor],
    *,
    oi: OpenInterestData | None = None,
    funding: FundingData | None = None,
) -> str:
    flags: list[str] = []
    if composite is not None:
        sentiment = composite.signal_breakdown.get("sentiment", 0.0)
        technical = composite.signal_breakdown.get("ta", 0.0)
        if sentiment > 0.05 and technical < -0.05:
            flags.append("Positive news but bearish price action - possible distribution by whales")
        elif sentiment < -0.05 and technical > 0.05:
            flags.append("Negative news but bullish price action - possible accumulation")

    if ta_signal is not None:
        if ta_signal.atr_pct > 10:
            flags.append("Extreme volatility - reduce position size, use wider stops")
        if ta_signal.bb_squeeze and ta_signal.bb_position in {"above_upper", "below_lower"}:
            flags.append("Breakout from squeeze - verify momentum before full entry")
        latest_hunt = _latest_stop_hunt(ta_signal)
        if latest_hunt is not None:
            flags.append(str(latest_hunt["description"]))

    strong_bull = sum(1 for factor in factors if factor.direction == "bullish" and factor.strength > 0.7)
    strong_bear = sum(1 for factor in factors if factor.direction == "bearish" and factor.strength > 0.7)
    if strong_bull > 0 and strong_bear > 0:
        flags.append("Strong signals conflicting - enter with smaller size, scale in on confirmation")

    if oi is not None and abs(oi.oi_change_4h_pct) > 10:
        flags.append(f"OI changed {oi.oi_change_4h_pct:+.1f}% in 4h - elevated liquidation cascade risk")

    if funding is not None and abs(funding.z_score) > 2.5:
        flags.append(
            f"Funding rate z-score at {funding.z_score:+.1f} - positioning is extreme and reversal risk is elevated"
        )

    if regime == "volatile":
        flags.append("Volatile regime - use 50% normal position size")

    return "No manipulation flags." if not flags else "\n".join(flags)


def _event_float(event_data: Mapping[str, object] | None, key: str) -> float | None:
    if event_data is None:
        return None
    value = event_data.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _decimals(price: float) -> int:
    if price > 1:
        return 2
    if price > 0.01:
        return 4
    return 6
