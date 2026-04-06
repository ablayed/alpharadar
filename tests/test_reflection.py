from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from src.agents.providers import LLMRole
from src.reflection.evaluator import ReflectionEvaluator, expected_calibration_error
from src.reflection.reflector import run_weekly_reflection
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

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
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
async def test_prediction_tracker_records_orphaned_predictions_older_than_28_hours(db) -> None:
    tracker = PredictionTracker(db.db_path)
    prediction_id = await tracker.log_prediction(
        symbol="SOL",
        asset_type="crypto",
        predicted_direction="bullish",
        confidence=0.82,
        price_at_prediction=150.0,
        source="analyze",
        reasoning="Momentum remains strong.",
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=39)).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            "UPDATE predictions SET created_at = ? WHERE id = ?",
            (stale_timestamp, prediction_id),
        )
        await connection.commit()

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "SOL"
        assert asset_type == "crypto"
        return 153.0

    updated = await tracker.record_outcomes(price_fetcher)
    assert updated == 1

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            "SELECT actual_price_1d, outcome_recorded FROM predictions WHERE id = ?",
            (prediction_id,),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == 153.0
    assert row[1] == 1


@pytest.mark.asyncio
async def test_prediction_tracker_backfills_3d_5d_and_7d_outcomes_without_upper_bounds(db) -> None:
    tracker = PredictionTracker(db.db_path)
    prediction_id = await tracker.log_prediction(
        symbol="ETH",
        asset_type="crypto",
        predicted_direction="bullish",
        confidence=0.7,
        price_at_prediction=3000.0,
        source="analyze",
        reasoning="Trend intact.",
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            """
            UPDATE predictions
            SET created_at = ?,
                actual_price_1d = ?,
                actual_change_1d_pct = ?,
                actual_direction = ?,
                is_correct_1d = ?,
                outcome_recorded = 1,
                outcome_recorded_at = datetime('now')
            WHERE id = ?
            """,
            (stale_timestamp, 3060.0, 2.0, "bullish", 1, prediction_id),
        )
        await connection.commit()

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "ETH"
        assert asset_type == "crypto"
        return 3150.0

    updated = await tracker.record_outcomes(price_fetcher)
    assert updated == 3

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT actual_price_3d, is_correct_3d,
                   actual_price_5d, is_correct_5d,
                   actual_price_7d, is_correct_7d
            FROM predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == 3150.0
    assert row[1] == 1
    assert row[2] == 3150.0
    assert row[3] == 1
    assert row[4] == 3150.0
    assert row[5] == 1


@pytest.mark.asyncio
async def test_prediction_tracker_can_backfill_all_timeframes_for_old_unresolved_prediction(db) -> None:
    tracker = PredictionTracker(db.db_path)
    prediction_id = await tracker.log_prediction(
        symbol="LINK",
        asset_type="crypto",
        predicted_direction="bullish",
        confidence=0.73,
        price_at_prediction=20.0,
        source="analyze",
        reasoning="Trend intact.",
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            "UPDATE predictions SET created_at = ? WHERE id = ?",
            (stale_timestamp, prediction_id),
        )
        await connection.commit()

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "LINK"
        assert asset_type == "crypto"
        return 21.0

    updated = await tracker.record_outcomes(price_fetcher)
    assert updated == 4

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT outcome_recorded,
                   actual_price_1d, actual_price_3d, actual_price_5d, actual_price_7d
            FROM predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == 1
    assert row[1] == 21.0
    assert row[2] == 21.0
    assert row[3] == 21.0
    assert row[4] == 21.0


@pytest.mark.asyncio
async def test_prediction_tracker_logs_skipped_symbols_when_price_missing(db, caplog: pytest.LogCaptureFixture) -> None:
    tracker = PredictionTracker(db.db_path)
    prediction_id = await tracker.log_prediction(
        symbol="TAO",
        asset_type="crypto",
        predicted_direction="bullish",
        confidence=0.66,
        price_at_prediction=400.0,
        source="analyze",
        reasoning="Signal fired.",
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            "UPDATE predictions SET created_at = ? WHERE id = ?",
            (stale_timestamp, prediction_id),
        )
        await connection.commit()

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "TAO"
        assert asset_type == "crypto"
        return None

    with caplog.at_level(logging.WARNING):
        updated = await tracker.record_outcomes(price_fetcher)

    assert updated == 0
    assert "Could not fetch price for TAO (crypto); skipping 1d outcome" in caplog.text

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            "SELECT outcome_recorded, actual_price_1d FROM predictions WHERE id = ?",
            (prediction_id,),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == 0
    assert row[1] is None


@pytest.mark.asyncio
async def test_prediction_tracker_allows_price_fetcher_db_writes(db) -> None:
    tracker = PredictionTracker(db.db_path)
    prediction_id = await tracker.log_prediction(
        symbol="SOL",
        asset_type="crypto",
        predicted_direction="bullish",
        confidence=0.68,
        price_at_prediction=120.0,
        source="analyze",
        reasoning="Signal fired.",
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            "UPDATE predictions SET created_at = ? WHERE id = ?",
            (stale_timestamp, prediction_id),
        )
        await connection.commit()

    async def price_fetcher(symbol: str, asset_type: str) -> float | None:
        assert symbol == "SOL"
        assert asset_type == "crypto"
        await db.increment_api_usage("coingecko", "/coins/markets")
        return 126.0

    updated = await tracker.record_outcomes(price_fetcher)

    assert updated == 1
    assert await db.get_monthly_api_usage("coingecko") == 1


@pytest.mark.asyncio
async def test_prediction_tracker_normalizes_trade_direction_and_source(db) -> None:
    tracker = PredictionTracker(db.db_path)
    prediction_id = await tracker.log_prediction(
        symbol="BTC",
        asset_type="crypto",
        predicted_direction="long",
        confidence=0.71,
        price_at_prediction=50_000.0,
        source="Trade",
        reasoning="Test trade setup.",
    )

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            "SELECT predicted_direction, source FROM predictions WHERE id = ?",
            (prediction_id,),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "bullish"
    assert row[1] == "trade"


@pytest.mark.asyncio
async def test_reflection_evaluator_reads_accuracy_and_latest_reflection(db) -> None:
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            """
            INSERT INTO predictions
            (symbol, asset_type, predicted_direction, confidence, source, price_at_prediction,
             outcome_recorded, is_correct_1d, is_correct_3d, is_correct_7d)
            VALUES
            ('NVDA', 'stock', 'bullish', 0.8, 'briefing', 100.0, 1, 1, 1, 1),
            ('BTC', 'crypto', 'bearish', 0.6, 'analyze', 50_000.0, 1, 0, 0, NULL),
            ('ETH', 'crypto', 'bullish', 0.75, 'trade', 3_000.0, 1, 1, NULL, NULL)
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

    assert snapshot.all_time_predictions == 3
    assert snapshot.one_day.total == 3
    assert snapshot.one_day.correct == 2
    assert snapshot.three_day.total == 2
    assert snapshot.three_day.correct == 1
    assert snapshot.seven_day.total == 1
    assert snapshot.seven_day.correct == 1
    assert snapshot.high_confidence_1d.total == 2
    assert snapshot.high_confidence_1d.correct == 2
    assert snapshot.low_confidence_1d.total == 1
    assert snapshot.low_confidence_1d.correct == 0
    assert snapshot.by_source_1d["briefing"].correct == 1
    assert snapshot.by_source_1d["trade"].total == 1
    assert reflection is not None
    assert reflection.total_predictions == 2
    assert reflection.accuracy_1d == 50.0
    assert await evaluator.latest_lessons() == ["Cut false-positive bullish calls"]


@pytest.mark.asyncio
async def test_weekly_reflection_prompt_includes_multi_timeframe_accuracy(db) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            """
            INSERT INTO predictions
            (created_at, symbol, asset_type, predicted_direction, confidence, reasoning, source,
             price_at_prediction, outcome_recorded, actual_direction, actual_price_1d,
             actual_change_1d_pct, is_correct_1d, actual_change_3d_pct, is_correct_3d,
             actual_change_5d_pct, is_correct_5d, actual_change_7d_pct, is_correct_7d)
            VALUES
            (?, 'BTC', 'crypto', 'bullish', 0.8, 'Trend intact', 'trade', 100.0, 1, 'bullish', 104.0, 4.0, 1, 6.0, 1, 8.0, 1, 10.0, 1),
            (?, 'ETH', 'crypto', 'bearish', 0.7, 'Momentum fading', 'analyze', 200.0, 1, 'bearish', 194.0, -3.0, 1, -1.0, 0, -4.0, 1, -6.0, 1),
            (?, 'SOL', 'crypto', 'bullish', 0.6, 'Breakout failed', 'playbook', 50.0, 1, 'bearish', 48.0, -4.0, 0, -6.0, 0, -5.0, 0, -3.0, 0)
            """,
            (now, now, now),
        )
        await connection.commit()

    class _FakeLLM:
        def __init__(self) -> None:
            self.user_prompt = ""

        async def generate_json(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
            role: LLMRole = LLMRole.GENERAL,
        ) -> dict[str, object]:
            del system_prompt, temperature, max_tokens, role
            self.user_prompt = user_prompt
            return {
                "performance_summary": "Mixed week.",
                "best_call": "BTC trend follow.",
                "worst_call": "SOL breakout.",
                "overconfidence_check": "Acceptable.",
                "lessons": ["Lesson 1", "Lesson 2", "Lesson 3"],
                "bias_detected": "None",
            }

    fake_llm = _FakeLLM()
    reflection_text = await run_weekly_reflection(db.db_path, fake_llm)

    assert reflection_text is not None
    assert "1-day accuracy:" in fake_llm.user_prompt
    assert "3-day accuracy:" in fake_llm.user_prompt
    assert "7-day accuracy:" in fake_llm.user_prompt
    assert "If 7-day accuracy is higher than 1-day accuracy" in fake_llm.user_prompt


def test_expected_calibration_error_returns_zero_for_perfect_calibration() -> None:
    error = expected_calibration_error([0.0, 1.0], [0, 1], buckets=2)
    assert error == 0.0
