from __future__ import annotations

import numpy as np


def _as_float_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Indicators expect 1D arrays")
    return array


def sma(close: np.ndarray, period: int = 20) -> np.ndarray:
    close_array = _as_float_array(close)
    result = np.full(close_array.shape, np.nan, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(close_array) < period:
        return result

    cumulative = np.cumsum(close_array, dtype=np.float64)
    cumulative[period:] = cumulative[period:] - cumulative[:-period]
    result[period - 1 :] = cumulative[period - 1 :] / period
    return result


def ema(close: np.ndarray, period: int = 20) -> np.ndarray:
    close_array = _as_float_array(close)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(close_array) == 0:
        return np.array([], dtype=np.float64)

    alpha = 2.0 / (period + 1)
    result = np.full(close_array.shape, np.nan, dtype=np.float64)
    result[0] = close_array[0]
    for index in range(1, len(close_array)):
        result[index] = alpha * close_array[index] + (1.0 - alpha) * result[index - 1]
    return result


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    close_array = _as_float_array(close)
    result = np.full(close_array.shape, np.nan, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(close_array) <= period:
        return result

    deltas = np.diff(close_array)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    result[period] = _rsi_value(avg_gain, avg_loss)

    for index in range(period + 1, len(close_array)):
        gain = gains[index - 1]
        loss = losses[index - 1]
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        result[index] = _rsi_value(avg_gain, avg_loss)

    return result


def macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close_array = _as_float_array(close)
    fast_ema = ema(close_array, fast)
    slow_ema = ema(close_array, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close_array = _as_float_array(close)
    middle = sma(close_array, period)
    rolling_std = np.full(close_array.shape, np.nan, dtype=np.float64)
    if len(close_array) < period:
        return rolling_std, middle, rolling_std

    for index in range(period - 1, len(close_array)):
        rolling_std[index] = np.std(close_array[index - period + 1 : index + 1], ddof=0)

    upper = middle + (std_dev * rolling_std)
    lower = middle - (std_dev * rolling_std)
    return upper, middle, lower


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    high_array = _as_float_array(high)
    low_array = _as_float_array(low)
    close_array = _as_float_array(close)
    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("high, low, and close arrays must have matching lengths")
    if period <= 0:
        raise ValueError("period must be positive")

    result = np.full(close_array.shape, np.nan, dtype=np.float64)
    if len(close_array) < period:
        return result

    true_ranges = np.empty(close_array.shape, dtype=np.float64)
    true_ranges[0] = high_array[0] - low_array[0]
    for index in range(1, len(close_array)):
        true_ranges[index] = max(
            high_array[index] - low_array[index],
            abs(high_array[index] - close_array[index - 1]),
            abs(low_array[index] - close_array[index - 1]),
        )

    result[period - 1] = float(np.mean(true_ranges[:period]))
    for index in range(period, len(close_array)):
        result[index] = ((result[index - 1] * (period - 1)) + true_ranges[index]) / period
    return result


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
