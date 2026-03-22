from __future__ import annotations

from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from src.reflection.evaluator import ReflectionEvaluator, expected_calibration_error
from src.reflection.tracker import PredictionTracker


@pytest.mark.asyncio
async def test_prediction_tracker_logs_and_records_outcomes(db) -> None:
    tracker = PredictionTracker(db.db_path)
    prediction_id = await tracker.log_prediction(
        symbol="NVDA",
        asset_type="stock",
        predicted_direction="bullish",
        confidence=0.75,
        price_at_prediction=100.0,
        source="analyze",
        reasoning="Momentum remains strong.",
    )
    assert prediction_id > 0

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            "UPDATE predictions SET created_at = ? WHERE id = ?",
            (stale_timestamp, prediction_id),
        )
        await connection.commit()

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "NVDA"
        assert asset_type == "stock"
        return 105.0

    updated = await tracker.record_outcomes(price_fetcher)
    assert updated == 1

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT actual_price_1d, actual_change_1d_pct, actual_direction, is_correct_1d
            FROM predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == 105.0
    assert row[1] == 5.0
    assert row[2] == "bullish"
    assert row[3] == 1


@pytest.mark.asyncio
async def test_reflection_evaluator_reads_accuracy_and_latest_reflection(db) -> None:
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            """
            INSERT INTO predictions
            (symbol, asset_type, predicted_direction, confidence, source, price_at_prediction,
             outcome_recorded, is_correct_1d)
            VALUES
            ('NVDA', 'stock', 'bullish', 0.8, 'briefing', 100.0, 1, 1),
            ('BTC', 'crypto', 'bearish', 0.6, 'analyze', 50_000.0, 1, 0)
            """
        )
        await connection.execute(
            """
            INSERT INTO reflections
            (reflection_type, period_start, period_end, total_predictions, accuracy_1d,
             avg_confidence, calibration_error, reflection_text, lessons_json)
            VALUES
            ('weekly', '2026-03-15 00:00:00', '2026-03-22 00:00:00', 2, 50.0,
             0.7, 0.1, '{"summary": "mixed"}', '["Cut false-positive bullish calls"]')
            """
        )
        await connection.commit()

    evaluator = ReflectionEvaluator(db.db_path)
    snapshot = await evaluator.accuracy_snapshot()
    reflection = await evaluator.latest_reflection()

    assert snapshot["all_time"].total == 2
    assert snapshot["all_time"].correct == 1
    assert reflection is not None
    assert reflection.total_predictions == 2
    assert reflection.accuracy_1d == 50.0
    assert await evaluator.latest_lessons() == ["Cut false-positive bullish calls"]


def test_expected_calibration_error_returns_zero_for_perfect_calibration() -> None:
    error = expected_calibration_error([0.0, 1.0], [0, 1], buckets=2)
    assert error == 0.0
