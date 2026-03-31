from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SwingPoint:
    index: int
    price: float
    point_type: str


@dataclass(slots=True)
class FibonacciLevels:
    swing_high: float
    swing_low: float
    direction: str
    fib_382: float
    fib_500: float
    fib_618: float
    fib_650: float
    fib_786: float
    ext_1272: float
    ext_1618: float
    ext_2618: float


def detect_swing_points(
    high: np.ndarray,
    low: np.ndarray,
    order: int = 5,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    if high_array.ndim != 1 or low_array.ndim != 1:
        raise ValueError("high and low must be 1D arrays")
    if len(high_array) != len(low_array):
        raise ValueError("high and low arrays must have matching lengths")
    if order <= 0:
        raise ValueError("order must be positive")

    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for index in range(order, len(high_array) - order):
        window_highs = high_array[index - order : index + order + 1]
        if high_array[index] == np.max(window_highs) and np.sum(window_highs == high_array[index]) == 1:
            swing_highs.append(
                SwingPoint(index=index, price=float(high_array[index]), point_type="high")
            )

        window_lows = low_array[index - order : index + order + 1]
        if low_array[index] == np.min(window_lows) and np.sum(window_lows == low_array[index]) == 1:
            swing_lows.append(
                SwingPoint(index=index, price=float(low_array[index]), point_type="low")
            )

    return swing_highs, swing_lows


def get_last_swing(
    high: np.ndarray,
    low: np.ndarray,
    order: int = 5,
) -> tuple[SwingPoint | None, SwingPoint | None]:
    swing_highs, swing_lows = detect_swing_points(high, low, order)
    last_high = swing_highs[-1] if swing_highs else None
    last_low = swing_lows[-1] if swing_lows else None
    return last_high, last_low


def compute_fibonacci(
    swing_high: float,
    swing_low: float,
    direction: str = "up",
) -> FibonacciLevels:
    range_size = swing_high - swing_low
    if range_size <= 0:
        raise ValueError("swing_high must be greater than swing_low")

    if direction == "up":
        fib_382 = swing_high - range_size * 0.382
        fib_500 = swing_high - range_size * 0.500
        fib_618 = swing_high - range_size * 0.618
        fib_650 = swing_high - range_size * 0.650
        fib_786 = swing_high - range_size * 0.786
        ext_1272 = swing_low + range_size * 1.272
        ext_1618 = swing_low + range_size * 1.618
        ext_2618 = swing_low + range_size * 2.618
    elif direction == "down":
        fib_382 = swing_low + range_size * 0.382
        fib_500 = swing_low + range_size * 0.500
        fib_618 = swing_low + range_size * 0.618
        fib_650 = swing_low + range_size * 0.650
        fib_786 = swing_low + range_size * 0.786
        ext_1272 = swing_high - range_size * 1.272
        ext_1618 = swing_high - range_size * 1.618
        ext_2618 = swing_high - range_size * 2.618
    else:
        raise ValueError("direction must be 'up' or 'down'")

    return FibonacciLevels(
        swing_high=float(swing_high),
        swing_low=float(swing_low),
        direction=direction,
        fib_382=float(fib_382),
        fib_500=float(fib_500),
        fib_618=float(fib_618),
        fib_650=float(fib_650),
        fib_786=float(fib_786),
        ext_1272=float(ext_1272),
        ext_1618=float(ext_1618),
        ext_2618=float(ext_2618),
    )


def detect_stop_hunt(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int = 3,
) -> list[dict[str, float | int | str]]:
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    open_array = np.asarray(open_, dtype=np.float64)
    if not (
        len(high_array) == len(low_array) == len(close_array) == len(open_array)
    ):
        raise ValueError("high, low, close, and open arrays must have matching lengths")

    hunts: list[dict[str, float | int | str]] = []
    for index in range(max(1, lookback), len(close_array)):
        body = abs(close_array[index] - open_array[index])
        if body < 1e-10:
            body = 1e-10

        upper_wick = high_array[index] - max(close_array[index], open_array[index])
        lower_wick = min(close_array[index], open_array[index]) - low_array[index]

        if lower_wick / body > 2.0:
            recent_low = np.min(low_array[max(0, index - 20) : index])
            if low_array[index] < recent_low and close_array[index] > recent_low:
                hunts.append(
                    {
                        "index": index,
                        "type": "bearish_stop_hunt",
                        "direction_after": "bullish",
                        "wick_ratio": float(lower_wick / body),
                        "description": (
                            "Long lower wick swept below recent lows then recovered - "
                            "likely stop hunt, bullish reversal signal"
                        ),
                    }
                )

        if upper_wick / body > 2.0:
            recent_high = np.max(high_array[max(0, index - 20) : index])
            if high_array[index] > recent_high and close_array[index] < recent_high:
                hunts.append(
                    {
                        "index": index,
                        "type": "bullish_stop_hunt",
                        "direction_after": "bearish",
                        "wick_ratio": float(upper_wick / body),
                        "description": (
                            "Long upper wick swept above recent highs then rejected - "
                            "likely stop hunt, bearish reversal signal"
                        ),
                    }
                )

    return hunts


def detect_fair_value_gaps(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
) -> list[dict[str, float | int | str]]:
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    open_array = np.asarray(open_, dtype=np.float64)
    if not (
        len(high_array) == len(low_array) == len(close_array) == len(open_array)
    ):
        raise ValueError("high, low, close, and open arrays must have matching lengths")

    fvgs: list[dict[str, float | int | str]] = []
    for index in range(2, len(close_array)):
        if high_array[index - 2] < low_array[index]:
            top = float(low_array[index])
            bottom = float(high_array[index - 2])
            fvgs.append(
                {
                    "index": index,
                    "type": "bullish_fvg",
                    "top": top,
                    "bottom": bottom,
                    "midpoint": float((top + bottom) / 2),
                    "description": (
                        f"Bullish FVG: liquidity void between {bottom:.2f} and {top:.2f}"
                    ),
                }
            )

        if low_array[index - 2] > high_array[index]:
            top = float(low_array[index - 2])
            bottom = float(high_array[index])
            fvgs.append(
                {
                    "index": index,
                    "type": "bearish_fvg",
                    "top": top,
                    "bottom": bottom,
                    "midpoint": float((top + bottom) / 2),
                    "description": (
                        f"Bearish FVG: liquidity void between {bottom:.2f} and {top:.2f}"
                    ),
                }
            )

    return fvgs


def detect_break_of_structure(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    order: int = 5,
) -> list[dict[str, float | int | str]]:
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("high, low, and close arrays must have matching lengths")

    swing_highs, swing_lows = detect_swing_points(high_array, low_array, order)
    bos_events: list[dict[str, float | int | str]] = []

    for swing_high in swing_highs:
        for index in range(swing_high.index + 1, min(swing_high.index + 30, len(close_array))):
            if close_array[index] > high_array[swing_high.index]:
                bos_events.append(
                    {
                        "index": index,
                        "type": "bullish_bos",
                        "broken_level": float(high_array[swing_high.index]),
                        "close_price": float(close_array[index]),
                        "description": (
                            "Bullish BOS: closed above previous high's wick at "
                            f"{high_array[swing_high.index]:.2f}"
                        ),
                    }
                )
                break

    for swing_low in swing_lows:
        for index in range(swing_low.index + 1, min(swing_low.index + 30, len(close_array))):
            if close_array[index] < low_array[swing_low.index]:
                bos_events.append(
                    {
                        "index": index,
                        "type": "bearish_bos",
                        "broken_level": float(low_array[swing_low.index]),
                        "close_price": float(close_array[index]),
                        "description": (
                            "Bearish BOS: closed below previous low's wick at "
                            f"{low_array[swing_low.index]:.2f}"
                        ),
                    }
                )
                break

    bos_events.sort(key=lambda event: (int(event["index"]), str(event["type"])))
    return bos_events


def detect_order_blocks(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    order: int = 5,
) -> list[dict[str, float | int | str]]:
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    open_array = np.asarray(open_, dtype=np.float64)
    if not (
        len(high_array) == len(low_array) == len(close_array) == len(open_array)
    ):
        raise ValueError("high, low, close, and open arrays must have matching lengths")

    bos_events = detect_break_of_structure(high_array, low_array, close_array, order)
    order_blocks: list[dict[str, float | int | str]] = []
    seen_blocks: set[tuple[str, int]] = set()

    for bos in bos_events:
        index = int(bos["index"])
        start = max(index - 10, 0)
        if bos["type"] == "bearish_bos":
            for candle_index in range(index - 1, start - 1, -1):
                if close_array[candle_index] > open_array[candle_index]:
                    block_key = ("bearish_ob", candle_index)
                    if block_key in seen_blocks:
                        break
                    seen_blocks.add(block_key)
                    order_blocks.append(
                        {
                            "index": candle_index,
                            "type": "bearish_ob",
                            "top": float(high_array[candle_index]),
                            "bottom": float(low_array[candle_index]),
                            "midpoint": float((high_array[candle_index] + low_array[candle_index]) / 2),
                            "description": (
                                f"Bearish OB zone: {low_array[candle_index]:.2f} to "
                                f"{high_array[candle_index]:.2f} - expect resistance here"
                            ),
                        }
                    )
                    break
        elif bos["type"] == "bullish_bos":
            for candle_index in range(index - 1, start - 1, -1):
                if close_array[candle_index] < open_array[candle_index]:
                    block_key = ("bullish_ob", candle_index)
                    if block_key in seen_blocks:
                        break
                    seen_blocks.add(block_key)
                    order_blocks.append(
                        {
                            "index": candle_index,
                            "type": "bullish_ob",
                            "top": float(high_array[candle_index]),
                            "bottom": float(low_array[candle_index]),
                            "midpoint": float((high_array[candle_index] + low_array[candle_index]) / 2),
                            "description": (
                                f"Bullish OB zone: {low_array[candle_index]:.2f} to "
                                f"{high_array[candle_index]:.2f} - expect support here"
                            ),
                        }
                    )
                    break

    return order_blocks
