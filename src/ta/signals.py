from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ta.indicators import atr, bollinger_bands, ema, macd, rsi


@dataclass(slots=True)
class TASignal:
    symbol: str
    timeframe: str
    current_price: float
    rsi_value: float
    rsi_signal: str
    macd_line: float
    macd_signal_line: float
    macd_histogram: float
    macd_signal: str
    bb_upper: float
    bb_lower: float
    bb_position: str
    bb_squeeze: bool
    atr_value: float
    atr_pct: float
    volatility: str
    ema_20: float
    ema_50: float
    trend: str
    overall_signal: str
    confidence: float
    summary: str


def compute_signals(
    symbol: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    timeframe: str = "daily",
) -> TASignal:
    close_array = np.asarray(close, dtype=np.float64)
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    if not (len(close_array) == len(high_array) == len(low_array)):
        raise ValueError("close, high, and low arrays must have the same length")
    if len(close_array) < 60:
        raise ValueError("Need at least 60 candles to compute TA signals")

    current_price = float(close_array[-1])

    rsi_values = rsi(close_array, 14)
    rsi_value = _nan_to_zero(rsi_values[-1])
    if rsi_value < 30:
        rsi_signal = "oversold"
    elif rsi_value > 70:
        rsi_signal = "overbought"
    else:
        rsi_signal = "neutral"

    macd_line, macd_signal_line, macd_histogram = macd(close_array)
    if macd_histogram[-1] > 0 and macd_histogram[-2] <= 0:
        macd_signal = "bullish_cross"
    elif macd_histogram[-1] < 0 and macd_histogram[-2] >= 0:
        macd_signal = "bearish_cross"
    elif macd_line[-1] > macd_signal_line[-1]:
        macd_signal = "bullish"
    else:
        macd_signal = "bearish"

    bb_upper, bb_middle, bb_lower = bollinger_bands(close_array)
    current_middle = _nan_to_zero(bb_middle[-1], fallback=current_price)
    current_upper = _nan_to_zero(bb_upper[-1], fallback=current_price)
    current_lower = _nan_to_zero(bb_lower[-1], fallback=current_price)
    upper_half = current_middle + (0.5 * (current_upper - current_middle))
    lower_half = current_middle - (0.5 * (current_middle - current_lower))

    if current_price > current_upper:
        bb_position = "above_upper"
    elif current_price > upper_half:
        bb_position = "near_upper"
    elif current_price < current_lower:
        bb_position = "below_lower"
    elif current_price < lower_half:
        bb_position = "near_lower"
    else:
        bb_position = "middle"

    bandwidth = (bb_upper - bb_lower) / np.where(bb_middle == 0, 1e-10, bb_middle)
    recent_bandwidth = bandwidth[-5:]
    average_bandwidth = float(np.nanmean(bandwidth[-50:])) if len(bandwidth) >= 50 else float(np.nanmean(bandwidth))
    bb_squeeze = bool(np.nanmean(recent_bandwidth) < (0.75 * average_bandwidth)) if average_bandwidth > 0 else False

    atr_values = atr(high_array, low_array, close_array, 14)
    atr_value = _nan_to_zero(atr_values[-1])
    atr_pct = (atr_value / current_price) * 100 if current_price > 0 else 0.0
    if atr_pct < 1.5:
        volatility = "low"
    elif atr_pct > 4.0:
        volatility = "high"
    else:
        volatility = "normal"

    ema_20 = float(ema(close_array, 20)[-1])
    ema_50 = float(ema(close_array, 50)[-1])
    if current_price > ema_20 > ema_50:
        trend = "strong_up"
    elif current_price > ema_20:
        trend = "up"
    elif current_price < ema_20 < ema_50:
        trend = "strong_down"
    elif current_price < ema_20:
        trend = "down"
    else:
        trend = "neutral"

    score = 0.0
    score += {"oversold": 1.0, "neutral": 0.0, "overbought": -1.0}[rsi_signal]
    score += {"bullish_cross": 2.0, "bullish": 1.0, "bearish": -1.0, "bearish_cross": -2.0}[macd_signal]
    score += {"above_upper": -1.0, "near_upper": 0.0, "middle": 0.0, "near_lower": 0.0, "below_lower": 1.0}[bb_position]
    score += {"strong_up": 1.0, "up": 0.5, "neutral": 0.0, "down": -0.5, "strong_down": -1.0}[trend]

    if score >= 3:
        overall_signal = "strong_buy"
        confidence = min(0.9, 0.5 + (score * 0.1))
    elif score >= 1:
        overall_signal = "buy"
        confidence = min(0.8, 0.5 + (score * 0.08))
    elif score <= -3:
        overall_signal = "strong_sell"
        confidence = min(0.9, 0.5 + (abs(score) * 0.1))
    elif score <= -1:
        overall_signal = "sell"
        confidence = min(0.8, 0.5 + (abs(score) * 0.08))
    else:
        overall_signal = "neutral"
        confidence = 0.4

    summary_parts: list[str] = []
    if rsi_signal != "neutral":
        summary_parts.append(f"RSI {rsi_value:.0f} ({rsi_signal})")
    summary_parts.append(f"MACD {macd_signal.replace('_', ' ')}")
    if trend in {"strong_up", "strong_down"}:
        summary_parts.append(f"trend {trend.replace('_', ' ')}")
    if bb_squeeze:
        summary_parts.append("Bollinger squeeze")
    summary = (
        f"{symbol.upper()}: {', '.join(summary_parts)}. "
        f"Signal: {overall_signal.replace('_', ' ')} ({confidence:.0%})."
    )

    return TASignal(
        symbol=symbol.upper(),
        timeframe=timeframe,
        current_price=current_price,
        rsi_value=rsi_value,
        rsi_signal=rsi_signal,
        macd_line=float(macd_line[-1]),
        macd_signal_line=float(macd_signal_line[-1]),
        macd_histogram=float(macd_histogram[-1]),
        macd_signal=macd_signal,
        bb_upper=current_upper,
        bb_lower=current_lower,
        bb_position=bb_position,
        bb_squeeze=bb_squeeze,
        atr_value=atr_value,
        atr_pct=atr_pct,
        volatility=volatility,
        ema_20=ema_20,
        ema_50=ema_50,
        trend=trend,
        overall_signal=overall_signal,
        confidence=float(confidence),
        summary=summary,
    )


def _nan_to_zero(value: float, *, fallback: float = 0.0) -> float:
    if np.isnan(value):
        return fallback
    return float(value)
