from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import aiosqlite

logger = logging.getLogger(__name__)

PriceFetcher = Callable[[str, str], Awaitable[float | None]]


class PredictionTracker:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def log_prediction(
        self,
        symbol: str,
        asset_type: str,
        predicted_direction: str,
        confidence: float,
        price_at_prediction: float,
        source: str,
        reasoning: str = "",
        event_type: str | None = None,
        predicted_target: float | None = None,
    ) -> int:
        normalized_direction = _normalize_prediction_direction(predicted_direction)
        normalized_source = source.strip().lower()
        async with aiosqlite.connect(self._db_path) as connection:
            cursor = await connection.execute(
                """
                INSERT INTO predictions
                (symbol, asset_type, event_type, predicted_direction, predicted_target,
                 confidence, reasoning, source, price_at_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.upper(),
                    asset_type,
                    event_type,
                    normalized_direction,
                    predicted_target,
                    confidence,
                    reasoning,
                    normalized_source,
                    price_at_prediction,
                ),
            )
            await connection.commit()
            return int(cursor.lastrowid)

    async def record_outcomes(self, price_fetcher: PriceFetcher) -> int:
        recorded_count = 0
        async with aiosqlite.connect(self._db_path) as connection:
            one_day_rows = await _fetch_prediction_rows(
                connection,
                start=_sqlite_timestamp(datetime.now(UTC) - timedelta(hours=28)),
                end=_sqlite_timestamp(datetime.now(UTC) - timedelta(hours=20)),
                where_clause="actual_price_1d IS NULL",
            )

            for row in one_day_rows:
                current_price = await price_fetcher(row["symbol"], row["asset_type"])
                if current_price is None:
                    continue

                predicted_price = float(row["price_at_prediction"])
                change_pct = ((current_price - predicted_price) / predicted_price) * 100
                actual_direction = _direction_from_change(change_pct, neutral_band=0.5)
                is_correct = 1 if row["predicted_direction"] == actual_direction else 0

                await connection.execute(
                    """
                    UPDATE predictions
                    SET actual_price_1d = ?,
                        actual_change_1d_pct = ?,
                        actual_direction = ?,
                        is_correct_1d = ?,
                        outcome_recorded = 1,
                        outcome_recorded_at = datetime('now')
                    WHERE id = ?
                    """,
                    (current_price, change_pct, actual_direction, is_correct, row["id"]),
                )
                recorded_count += 1

            five_day_rows = await _fetch_prediction_rows(
                connection,
                start=_sqlite_timestamp(datetime.now(UTC) - timedelta(days=6)),
                end=_sqlite_timestamp(datetime.now(UTC) - timedelta(days=4)),
                where_clause="actual_price_5d IS NULL AND actual_price_1d IS NOT NULL",
            )

            for row in five_day_rows:
                current_price = await price_fetcher(row["symbol"], row["asset_type"])
                if current_price is None:
                    continue

                predicted_price = float(row["price_at_prediction"])
                change_pct = ((current_price - predicted_price) / predicted_price) * 100
                actual_direction = _direction_from_change(change_pct, neutral_band=2.0)
                is_correct = 1 if row["predicted_direction"] == actual_direction else 0

                await connection.execute(
                    """
                    UPDATE predictions
                    SET actual_price_5d = ?,
                        actual_change_5d_pct = ?,
                        is_correct_5d = ?
                    WHERE id = ?
                    """,
                    (current_price, change_pct, is_correct, row["id"]),
                )
                recorded_count += 1

            await connection.commit()

        logger.info("Recorded %d prediction outcomes", recorded_count)
        return recorded_count


async def _fetch_prediction_rows(
    connection: aiosqlite.Connection,
    *,
    start: str,
    end: str,
    where_clause: str,
) -> list[aiosqlite.Row]:
    connection.row_factory = aiosqlite.Row
    cursor = await connection.execute(
        f"""
        SELECT id, symbol, asset_type, predicted_direction, price_at_prediction
        FROM predictions
        WHERE {where_clause}
          AND created_at BETWEEN ? AND ?
        """,
        (start, end),
    )
    rows = await cursor.fetchall()
    return rows


def _direction_from_change(change_pct: float, *, neutral_band: float) -> str:
    if change_pct > neutral_band:
        return "bullish"
    if change_pct < -neutral_band:
        return "bearish"
    return "neutral"


def _normalize_prediction_direction(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "long": "bullish",
        "buy": "bullish",
        "bullish": "bullish",
        "short": "bearish",
        "sell": "bearish",
        "bearish": "bearish",
        "neutral": "neutral",
    }
    return mapping.get(normalized, "neutral")


def _sqlite_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
