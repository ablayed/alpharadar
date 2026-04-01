"""
Feature engineering for lightweight directional ML models.

The feature set focuses on the indicators that tend to work best for
tabular market data: momentum, trend, volatility, volume, price action,
lagged returns, and a few interaction terms.

All computations are pure NumPy.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-10
WARMUP_ROWS = 50


def engineer_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    open_: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Create a dense feature matrix from OHLCV arrays."""
    close_array, high_array, low_array, volume_array, open_array = _validate_inputs(
        close,
        high,
        low,
        volume,
        open_,
    )
    n = len(close_array)

    returns_1 = _percent_change(close_array, 1)
    ema_9 = _ema(close_array, 9)
    ema_12 = _ema(close_array, 12)
    ema_20 = _ema(close_array, 20)
    ema_26 = _ema(close_array, 26)
    ema_50 = _ema(close_array, 50)
    macd_line = ema_12 - ema_26
    macd_signal = _ema(macd_line, 9)
    macd_histogram = macd_line - macd_signal
    adx_14 = _compute_adx(high_array, low_array, close_array, 14)
    atr_7 = _atr(high_array, low_array, close_array, 7)
    atr_14 = _atr(high_array, low_array, close_array, 14)
    sma_20 = _sma(close_array, 20)
    std_20 = _rolling_std(close_array, 20)
    volume_sma_5 = _sma(volume_array, 5)
    volume_sma_20 = _sma(volume_array, 20)
    obv = _compute_obv(close_array, volume_array)
    obv_sma_5 = _sma(obv, 5)
    obv_abs_sma_10 = _sma(np.abs(obv), 10)
    full_range = np.maximum(high_array - low_array, EPSILON)
    log_hl = np.log(np.maximum(high_array, EPSILON) / np.maximum(low_array, EPSILON))
    log_co = np.log(np.maximum(close_array, EPSILON) / np.maximum(open_array, EPSILON))
    gk_volatility = _sma(0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2), 14)

    features: dict[str, np.ndarray] = {}

    # Momentum
    for period in (7, 14, 21):
        features[f"rsi_{period}"] = _rsi(close_array, period)

    for period in (14, 21):
        stochastic_k, stochastic_d = _stochastic(high_array, low_array, close_array, period)
        features[f"stoch_k_{period}"] = stochastic_k
        features[f"stoch_d_{period}"] = stochastic_d

    for period in (10, 20, 40):
        features[f"cci_{period}"] = _cci(high_array, low_array, close_array, period)

    for period in (1, 3, 5, 10, 20):
        features[f"roc_{period}"] = _percent_change(close_array, period) * 100.0

    features["momentum_5"] = close_array - _shift(close_array, 5)

    # Trend
    features["macd_line"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_histogram"] = macd_histogram
    features["adx_14"] = adx_14
    features["price_ema9_ratio"] = close_array / np.maximum(ema_9, EPSILON)
    features["price_ema20_ratio"] = close_array / np.maximum(ema_20, EPSILON)
    features["price_ema50_ratio"] = close_array / np.maximum(ema_50, EPSILON)
    features["ema9_ema20_ratio"] = ema_9 / np.maximum(ema_20, EPSILON)
    features["ema20_ema50_ratio"] = ema_20 / np.maximum(ema_50, EPSILON)
    features["ema20_slope_5"] = _percent_change(ema_20, 5)
    features["ema50_slope_10"] = _percent_change(ema_50, 10)

    # Volatility
    features["atr_7_pct"] = atr_7 / np.maximum(close_array, EPSILON) * 100.0
    features["atr_14_pct"] = atr_14 / np.maximum(close_array, EPSILON) * 100.0
    features["bb_width_20"] = (4.0 * std_20) / np.maximum(sma_20, EPSILON)
    features["bb_position_20"] = (close_array - sma_20) / np.maximum(2.0 * std_20, EPSILON)
    features["gk_volatility_14"] = gk_volatility
    features["realized_vol_5"] = _rolling_std(returns_1, 5)
    features["realized_vol_10"] = _rolling_std(returns_1, 10)
    features["range_pct"] = full_range / np.maximum(close_array, EPSILON)

    # Volume
    features["volume_ratio"] = volume_array / np.maximum(volume_sma_20, EPSILON)
    features["volume_momentum"] = volume_sma_5 / np.maximum(volume_sma_20, EPSILON)
    features["volume_zscore_20"] = _rolling_zscore(volume_array, 20)
    features["obv_slope_5"] = _percent_change(obv_sma_5, 5)
    features["obv_ratio_10"] = obv / np.maximum(obv_abs_sma_10, EPSILON)

    # Price action
    body = np.abs(close_array - open_array)
    upper_wick = high_array - np.maximum(close_array, open_array)
    lower_wick = np.minimum(close_array, open_array) - low_array
    features["body_ratio"] = body / full_range
    features["close_position"] = (close_array - low_array) / full_range
    features["upper_wick_ratio"] = upper_wick / full_range
    features["lower_wick_ratio"] = lower_wick / full_range
    features["intraday_return"] = (close_array - open_array) / np.maximum(open_array, EPSILON)
    previous_close = _shift(close_array, 1)
    features["gap_return"] = (open_array - previous_close) / np.maximum(np.abs(previous_close), EPSILON)

    # Lagged returns
    for lag in (1, 3, 7, 14, 21):
        features[f"return_{lag}d"] = _percent_change(close_array, lag)

    # Cross-features
    features["rsi14_x_volume"] = features["rsi_14"] * features["volume_ratio"]
    features["macd_hist_x_adx"] = macd_histogram * adx_14
    features["bb_pos_x_rsi14"] = features["bb_position_20"] * features["rsi_14"]

    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names]).astype(np.float64, copy=False)
    X[: min(n, WARMUP_ROWS), :] = np.nan
    return X, feature_names


def _validate_inputs(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    open_: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close_array = np.asarray(close, dtype=np.float64)
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    if not (len(close_array) == len(high_array) == len(low_array) == len(volume_array)):
        raise ValueError("close, high, low, and volume must have the same length")
    if len(close_array) == 0:
        raise ValueError("at least one candle is required")

    if open_ is None:
        open_array = _shift(close_array, 1)
        open_array[0] = close_array[0]
    else:
        open_array = np.asarray(open_, dtype=np.float64)
        if len(open_array) != len(close_array):
            raise ValueError("open array must have the same length as close")

    return close_array, high_array, low_array, volume_array, open_array


def _shift(data: np.ndarray, periods: int) -> np.ndarray:
    if periods <= 0:
        return np.asarray(data, dtype=np.float64)
    result = np.full(len(data), np.nan, dtype=np.float64)
    result[periods:] = data[:-periods]
    return result


def _percent_change(data: np.ndarray, periods: int) -> np.ndarray:
    shifted = _shift(data, periods)
    return (data - shifted) / np.maximum(np.abs(shifted), EPSILON)


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    result = np.empty(len(data), dtype=np.float64)
    alpha = 2.0 / (period + 1)
    result[0] = data[0]
    for index in range(1, len(data)):
        result[index] = (alpha * data[index]) + ((1.0 - alpha) * result[index - 1])
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(data), np.nan, dtype=np.float64)
    if period <= 0 or len(data) < period:
        return result
    for index in range(period - 1, len(data)):
        window = data[index - period + 1 : index + 1]
        if np.isnan(window).any():
            continue
        result[index] = float(np.mean(window))
    return result


def _rolling_min(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(data), np.nan, dtype=np.float64)
    for index in range(period - 1, len(data)):
        result[index] = float(np.min(data[index - period + 1 : index + 1]))
    return result


def _rolling_max(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(data), np.nan, dtype=np.float64)
    for index in range(period - 1, len(data)):
        result[index] = float(np.max(data[index - period + 1 : index + 1]))
    return result


def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(data), np.nan, dtype=np.float64)
    for index in range(period - 1, len(data)):
        result[index] = float(np.std(data[index - period + 1 : index + 1]))
    return result


def _rolling_mad(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(data), np.nan, dtype=np.float64)
    for index in range(period - 1, len(data)):
        window = data[index - period + 1 : index + 1]
        result[index] = float(np.mean(np.abs(window - np.mean(window))))
    return result


def _rolling_zscore(data: np.ndarray, period: int) -> np.ndarray:
    mean = _sma(data, period)
    std = _rolling_std(data, period)
    return (data - mean) / np.maximum(std, EPSILON)


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)
    rs = avg_gain / np.maximum(avg_loss, EPSILON)
    return 100.0 - (100.0 / (1.0 + rs))


def _stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    lowest = _rolling_min(low, period)
    highest = _rolling_max(high, period)
    denom = np.maximum(highest - lowest, EPSILON)
    percent_k = ((close - lowest) / denom) * 100.0
    percent_d = _sma(percent_k, 3)
    return percent_k, percent_d


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    typical_price = (high + low + close) / 3.0
    typical_sma = _sma(typical_price, period)
    typical_mad = _rolling_mad(typical_price, period)
    return (typical_price - typical_sma) / np.maximum(typical_mad * 0.015, EPSILON)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    previous_close = _shift(close, 1)
    true_range = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - previous_close),
            np.abs(low - previous_close),
        ),
    )
    true_range[0] = high[0] - low[0]
    return _ema(true_range, period)


def _compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    plus_dm = np.zeros(len(close), dtype=np.float64)
    minus_dm = np.zeros(len(close), dtype=np.float64)
    true_range = np.zeros(len(close), dtype=np.float64)

    for index in range(1, len(close)):
        up_move = high[index] - high[index - 1]
        down_move = low[index - 1] - low[index]
        plus_dm[index] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[index] = down_move if down_move > up_move and down_move > 0 else 0.0
        true_range[index] = max(
            high[index] - low[index],
            abs(high[index] - close[index - 1]),
            abs(low[index] - close[index - 1]),
        )

    atr = _ema(true_range, period)
    plus_di = 100.0 * _ema(plus_dm, period) / np.maximum(atr, EPSILON)
    minus_di = 100.0 * _ema(minus_dm, period) / np.maximum(atr, EPSILON)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, EPSILON)
    return _ema(dx, period)


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
