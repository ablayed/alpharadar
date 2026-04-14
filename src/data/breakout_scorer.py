"""
Smart breakout scorer for crypto discovery.

Detects coins before they break out or break down using:
- volume accumulation / distribution
- volatility compression
- relative volume
- momentum divergence
- taker buy / sell pressure
- social velocity
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BreakoutScore:
    symbol: str
    total_score: int
    direction: str
    volume_accumulation: int
    volatility_compression: int
    price_position: int
    momentum: int
    taker_pressure: int
    social_velocity: int
    signals_firing: int
    description: str
    key_level: float | None


def score_breakout(
    symbol: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    open_: np.ndarray | None = None,
    taker_buy_ratio: float | None = None,
    is_trending: bool = False,
    price_change_24h: float = 0.0,
) -> BreakoutScore:
    close_array = np.asarray(close, dtype=np.float64)
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    open_array = np.asarray(open_, dtype=np.float64) if open_ is not None else None

    if min(len(close_array), len(high_array), len(low_array), len(volume_array)) < 20:
        return BreakoutScore(
            symbol=symbol,
            total_score=0,
            direction="neutral",
            volume_accumulation=0,
            volatility_compression=0,
            price_position=0,
            momentum=0,
            taker_pressure=0,
            social_velocity=0,
            signals_firing=0,
            description="Insufficient data",
            key_level=None,
        )

    obv = _compute_obv(close_array, volume_array)
    sma20 = _sma(close_array, 20)
    std20 = _rolling_std(close_array, 20)
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)
    bb_width = (bb_upper - bb_lower) / np.where(sma20 == 0, 1e-10, sma20)
    tr = _true_range(high_array, low_array, close_array)
    atr14 = _sma(tr, 14)
    rsi = _compute_rsi(close_array, 14)
    ema12 = _ema(close_array, 12)
    ema26 = _ema(close_array, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    vol_sma20 = _sma(volume_array, 20)
    rvol = volume_array[-1] / vol_sma20[-1] if vol_sma20[-1] > 0 else 1.0

    high_30d = float(np.max(high_array[-30:])) if len(high_array) >= 30 else float(np.max(high_array))
    low_30d = float(np.min(low_array[-30:])) if len(low_array) >= 30 else float(np.min(low_array))
    current_price = float(close_array[-1])

    vol_acc_long = 0
    vol_acc_short = 0
    vol_comp_long = 0
    vol_comp_short = 0
    price_pos_long = 0
    price_pos_short = 0
    momentum_long = 0
    momentum_short = 0
    taker_long = 0
    taker_short = 0
    social_long = 0
    social_short = 0
    signals_long = 0
    signals_short = 0
    descriptions_long: list[str] = []
    descriptions_short: list[str] = []
    key_level: float | None = None

    if len(obv) >= 7:
        obv_slope = (obv[-1] - obv[-7]) / max(abs(obv[-7]), 1e-10)
        price_slope = (close_array[-1] - close_array[-7]) / max(abs(close_array[-7]), 1e-10)
        if obv_slope > 0.05 and price_slope < 0.02:
            vol_acc_long += 15
            descriptions_long.append("OBV accumulation")
        if obv_slope < -0.05 and price_slope > -0.02:
            vol_acc_short += 15
            descriptions_short.append("OBV distribution")

    if rvol >= 3.0:
        vol_acc_long += 15
        vol_acc_short += 10
        descriptions_long.append(f"RVOL {rvol:.1f}x")
    elif rvol >= 2.0:
        vol_acc_long += 10
        vol_acc_short += 7
    elif rvol >= 1.5:
        vol_acc_long += 5
        vol_acc_short += 3

    vol_acc_long = min(vol_acc_long, 30)
    vol_acc_short = min(vol_acc_short, 30)
    if vol_acc_long >= 10:
        signals_long += 1
    if vol_acc_short >= 10:
        signals_short += 1

    if len(bb_width) >= 30:
        recent_bw = float(bb_width[-1])
        historical_bw = bb_width[-30:]
        finite_bw = historical_bw[np.isfinite(historical_bw)]
        if len(finite_bw) > 0:
            percentile = (np.sum(finite_bw < recent_bw) / len(finite_bw)) * 100
            if percentile < 10:
                vol_comp_long += 20
                vol_comp_short += 20
                descriptions_long.append("Extreme BB squeeze")
                descriptions_short.append("Extreme BB squeeze")
            elif percentile < 20:
                vol_comp_long += 15
                vol_comp_short += 15
            elif percentile < 30:
                vol_comp_long += 8
                vol_comp_short += 8

    if len(atr14) >= 20:
        atr_sma = _sma(atr14, 20)
        if atr14[-1] > 0 and atr_sma[-1] > 0 and atr14[-1] < (atr_sma[-1] * 0.8):
            if vol_comp_long < 20:
                vol_comp_long += 5
            if vol_comp_short < 20:
                vol_comp_short += 5

    vol_comp_long = min(vol_comp_long, 20)
    vol_comp_short = min(vol_comp_short, 20)
    if vol_comp_long >= 10:
        signals_long += 1
    if vol_comp_short >= 10:
        signals_short += 1

    pct_from_high = (current_price - high_30d) / max(high_30d, 1e-10)
    pct_from_low = (current_price - low_30d) / max(low_30d, 1e-10)

    if -0.02 <= pct_from_high <= 0.01:
        price_pos_long += 15
        key_level = high_30d
        descriptions_long.append(f"Testing 30d high {high_30d:.4g}")
    elif -0.05 <= pct_from_high < -0.02:
        price_pos_long += 8
        key_level = high_30d

    if -0.01 <= pct_from_low <= 0.02:
        price_pos_short += 15
        key_level = low_30d
        descriptions_short.append(f"Testing 30d low {low_30d:.4g}")
    elif 0.02 < pct_from_low <= 0.05:
        price_pos_short += 8
        key_level = low_30d

    if price_pos_long >= 8:
        signals_long += 1
    if price_pos_short >= 8:
        signals_short += 1

    current_rsi = float(rsi[-1])
    if 50 < current_rsi < 65:
        momentum_long += 8
    if macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2]:
        momentum_long += 7
        descriptions_long.append("MACD accelerating bullish")
    if 35 < current_rsi < 50:
        momentum_short += 8
    if macd_hist[-1] < 0 and macd_hist[-1] < macd_hist[-2]:
        momentum_short += 7
        descriptions_short.append("MACD accelerating bearish")

    if len(close_array) >= 5 and len(rsi) >= 5:
        if close_array[-1] < close_array[-4] and rsi[-1] > rsi[-4]:
            momentum_long += 5
            descriptions_long.append("Bullish RSI divergence")
        if close_array[-1] > close_array[-4] and rsi[-1] < rsi[-4]:
            momentum_short += 5
            descriptions_short.append("Bearish RSI divergence")

    if open_array is not None and len(open_array) >= 3:
        upper_wicks = high_array[-3:] - np.maximum(close_array[-3:], open_array[-3:])
        candle_ranges = np.maximum(high_array[-3:] - low_array[-3:], 1e-10)
        lower_wicks = np.minimum(close_array[-3:], open_array[-3:]) - low_array[-3:]
        if np.mean(upper_wicks / candle_ranges) > 0.45 and pct_from_high > -0.05:
            momentum_short += 5
            descriptions_short.append("Resistance rejection")
        if np.mean(lower_wicks / candle_ranges) > 0.45 and pct_from_low < 0.05:
            momentum_long += 5
            descriptions_long.append("Support absorption")

    momentum_long = min(momentum_long, 15)
    momentum_short = min(momentum_short, 15)
    if momentum_long >= 8:
        signals_long += 1
    if momentum_short >= 8:
        signals_short += 1

    if taker_buy_ratio is not None:
        if taker_buy_ratio > 0.55:
            taker_long += 10
            descriptions_long.append(f"Taker buy pressure {taker_buy_ratio:.0%}")
        elif taker_buy_ratio > 0.52:
            taker_long += 5
        if taker_buy_ratio < 0.45:
            taker_short += 10
            descriptions_short.append(f"Taker sell pressure {1 - taker_buy_ratio:.0%}")
        elif taker_buy_ratio < 0.48:
            taker_short += 5

    if taker_long >= 5:
        signals_long += 1
    if taker_short >= 5:
        signals_short += 1

    if is_trending:
        social_long += 7
        descriptions_long.append("Trending on CoinGecko")
    if is_trending and abs(price_change_24h) < 5:
        social_long += 3
        descriptions_long.append("Trending while price is flat")
    if is_trending and price_change_24h < -15:
        social_short += 7
        descriptions_short.append("Trending during downside panic")

    social_long = min(social_long, 10)
    social_short = min(social_short, 10)
    if social_long >= 5:
        signals_long += 1
    if social_short >= 5:
        signals_short += 1

    total_long = vol_acc_long + vol_comp_long + price_pos_long + momentum_long + taker_long + social_long
    total_short = vol_acc_short + vol_comp_short + price_pos_short + momentum_short + taker_short + social_short

    if price_change_24h < -20:
        total_long = max(0, total_long - 30)
    if price_change_24h > 30:
        total_short = max(0, total_short - 20)

    if total_long > total_short and total_long >= 30 and signals_long >= 2:
        descriptions = descriptions_long or ["Bullish setup forming"]
        return BreakoutScore(
            symbol=symbol,
            total_score=min(total_long, 100),
            direction="long_setup",
            volume_accumulation=vol_acc_long,
            volatility_compression=vol_comp_long,
            price_position=price_pos_long,
            momentum=momentum_long,
            taker_pressure=taker_long,
            social_velocity=social_long,
            signals_firing=signals_long,
            description=" | ".join(descriptions[:4]),
            key_level=key_level,
        )

    if total_short > total_long and total_short >= 30 and signals_short >= 2:
        descriptions = descriptions_short or ["Bearish setup forming"]
        return BreakoutScore(
            symbol=symbol,
            total_score=min(total_short, 100),
            direction="short_setup",
            volume_accumulation=vol_acc_short,
            volatility_compression=vol_comp_short,
            price_position=price_pos_short,
            momentum=momentum_short,
            taker_pressure=taker_short,
            social_velocity=social_short,
            signals_firing=signals_short,
            description=" | ".join(descriptions[:4]),
            key_level=key_level,
        )

    return BreakoutScore(
        symbol=symbol,
        total_score=max(total_long, total_short),
        direction="neutral",
        volume_accumulation=max(vol_acc_long, vol_acc_short),
        volatility_compression=max(vol_comp_long, vol_comp_short),
        price_position=max(price_pos_long, price_pos_short),
        momentum=max(momentum_long, momentum_short),
        taker_pressure=max(taker_long, taker_short),
        social_velocity=max(social_long, social_short),
        signals_firing=0,
        description="No clear setup yet",
        key_level=key_level,
    )


def _compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    obv = np.zeros(len(close), dtype=np.float64)
    for index in range(1, len(close)):
        if close[index] > close[index - 1]:
            obv[index] = obv[index - 1] + volume[index]
        elif close[index] < close[index - 1]:
            obv[index] = obv[index - 1] - volume[index]
        else:
            obv[index] = obv[index - 1]
    return obv


def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    result = np.zeros_like(values)
    alpha = 2.0 / (period + 1)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = (alpha * values[index]) + ((1 - alpha) * result[index - 1])
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    result = np.zeros_like(values)
    if period <= 0 or len(values) < period:
        return result
    cumsum = np.cumsum(values)
    result[period - 1 :] = (cumsum[period - 1 :] - np.concatenate(([0.0], cumsum[:-period]))) / period
    return result


def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    result = np.zeros_like(values)
    if period <= 0 or len(values) < period:
        return result
    for index in range(period - 1, len(values)):
        result[index] = float(np.std(values[index - period + 1 : index + 1]))
    return result
