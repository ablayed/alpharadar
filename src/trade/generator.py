from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from src.agents.state import DebateVerdict
from src.signals.models import CompositeScore
from src.ta.signals import TASignal


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
    timestamp: float


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

    atr_value = ta_signal.atr_value if ta_signal is not None and ta_signal.atr_value > 0 else current_price * 0.03
    sl_distance = max(atr_value * 1.5, current_price * 0.015)

    if direction == "LONG":
        entry_high = current_price
        entry_low = max(current_price - (atr_value * 0.3), 0.0001)
        if ta_signal is not None:
            support_candidates = [entry_low]
            for support in (ta_signal.ema_20, ta_signal.bb_lower):
                if current_price - atr_value <= support <= current_price:
                    support_candidates.append(support)
            entry_low = max(min(support_candidates), 0.0001)
        stop_loss = max(entry_low - sl_distance, 0.0001)
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
        stop_loss = entry_high + sl_distance
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

    entry_reference = entry_high if direction == "LONG" else entry_low
    tp1, tp2, tp3 = _normalize_targets(direction, entry_reference, sl_distance, tp1, tp2, tp3)

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

    stop_loss_pct = (risk / entry_reference) * 100 if entry_reference > 0 else 0.0
    tp1_pct = (abs(tp1 - entry_reference) / entry_reference) * 100 if entry_reference > 0 else 0.0
    tp2_pct = (abs(tp2 - entry_reference) / entry_reference) * 100 if entry_reference > 0 else 0.0
    tp3_pct = (abs(tp3 - entry_reference) / entry_reference) * 100 if entry_reference > 0 else 0.0

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
        timestamp=time.time(),
    )


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
    min_step = max(risk * 0.25, entry_reference * 0.002, 0.0001)
    if direction == "LONG":
        targets = sorted([tp1, tp2, tp3])
        tp1 = max(targets[0], entry_reference + min_step)
        tp2 = max(targets[1], tp1 + min_step)
        tp3 = max(targets[2], tp2 + min_step)
        return tp1, tp2, tp3

    targets = sorted([tp1, tp2, tp3], reverse=True)
    tp1 = min(targets[0], entry_reference - min_step)
    tp2 = min(targets[1], tp1 - min_step)
    tp3 = min(targets[2], tp2 - min_step)
    return tp1, tp2, tp3


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
            reasons.append(f"{event_type.replace('_', ' ')} still shows {drift_pct:+.1f}% remaining drift")
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
        trend_text = ta_signal.trend.replace("_", " ")
        reasons.append(f"trend is {trend_text}")
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
            flags.append("Sentiment and price action disagree: positive news but weak tape can mean distribution.")
        elif sentiment < -0.05 and technical > 0.05:
            flags.append("Sentiment and price action disagree: negative news with firm tape can mean accumulation.")
        if abs(correlation) > 0.12 and abs(technical) > 0.05 and (correlation * technical) < 0:
            flags.append("Cross-market behavior is diverging from price action, which can be a warning sign.")

    if ta_signal is not None:
        if ta_signal.atr_pct > 8:
            flags.append("Volatility is unusually high, so the move may be event-driven or easily whipsawed.")
        if ta_signal.bb_squeeze and ta_signal.bb_position in {"above_upper", "below_lower"}:
            flags.append("Price just broke out of a tight range. Confirm follow-through before sizing up.")

    if regime == "volatile":
        flags.append("The broader market is volatile. Smaller size and more patience make sense here.")

    return "No manipulation flags detected." if not flags else "\n".join(flags)


def _event_float(event_data: Mapping[str, object] | None, key: str) -> float | None:
    if event_data is None:
        return None
    value = event_data.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None
