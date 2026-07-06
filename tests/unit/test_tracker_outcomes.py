from __future__ import annotations

from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from src.reflection.tracker import PredictionTracker


async def _insert_prediction(
    db_path: str,
    *,
    created_at: datetime,
    symbol: str,
    asset_type: str = "crypto",
    predicted_direction: str = "bullish",
    price_at_prediction: float = 100.0,
    outcome_recorded: int = 0,
    actual_price_1d: float | None = None,
) -> int:
    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            INSERT INTO predictions (
                created_at,
                symbol,
                asset_type,
                predicted_direction,
                confidence,
                reasoning,
                source,
                price_at_prediction,
                outcome_recorded,
                actual_price_1d
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at.strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                asset_type,
                predicted_direction,
                0.7,
                "test",
                "analyze",
                price_at_prediction,
                outcome_recorded,
                actual_price_1d,
            ),
        )
        await connection.commit()
        return int(cursor.lastrowid)


@pytest.mark.asyncio
async def test_open_ended_window_catches_all_ages(db) -> None:
    tracker = PredictionTracker(db.db_path)
    now = datetime.now(UTC)
    eligible_ids: list[int] = []

    for hour_offset in range(30):
        created_at = now - timedelta(hours=hour_offset)
        prediction_id = await _insert_prediction(
            db.db_path,
            created_at=created_at,
            symbol=f"COIN{hour_offset}",
        )
        if created_at < now - timedelta(hours=24):
            eligible_ids.append(prediction_id)

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert asset_type == "crypto"
        assert symbol.startswith("COIN")
        return 101.0

    updated = await tracker.record_outcomes(price_fetcher)
    assert updated == len(eligible_ids)

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE actual_price_1d IS NOT NULL
            """
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == len(eligible_ids)


@pytest.mark.asyncio
async def test_idempotency_preserved(db) -> None:
    tracker = PredictionTracker(db.db_path)
    created_at = datetime.now(UTC) - timedelta(days=8)
    await _insert_prediction(
        db.db_path,
        created_at=created_at,
        symbol="BTC",
    )

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "BTC"
        assert asset_type == "crypto"
        return 102.0

    first_update_count = await tracker.record_outcomes(price_fetcher)
    second_update_count = await tracker.record_outcomes(price_fetcher)

    assert first_update_count == 4
    assert second_update_count == 0


@pytest.mark.asyncio
async def test_five_day_still_requires_one_day_resolved_first(db) -> None:
    tracker = PredictionTracker(db.db_path)
    created_at = datetime.now(UTC) - timedelta(days=8)
    prediction_id = await _insert_prediction(
        db.db_path,
        created_at=created_at,
        symbol="ETH",
        predicted_direction="bearish",
        outcome_recorded=0,
        actual_price_1d=None,
    )

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "ETH"
        assert asset_type == "crypto"
        return None

    updated = await tracker.record_outcomes(price_fetcher)
    assert updated == 0

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT actual_price_1d, actual_price_5d, is_correct_5d
            FROM predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
