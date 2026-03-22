from __future__ import annotations

import json
from dataclasses import dataclass

import aiosqlite

from src.reflection.models import ReflectionRecord


@dataclass(slots=True)
class AccuracyStats:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.correct / self.total) * 100.0


def expected_calibration_error(
    confidences: list[float],
    outcomes: list[int],
    *,
    buckets: int = 5,
) -> float:
    if not confidences or not outcomes or len(confidences) != len(outcomes):
        return 0.0

    bucket_totals = [0 for _ in range(buckets)]
    bucket_confidence = [0.0 for _ in range(buckets)]
    bucket_accuracy = [0.0 for _ in range(buckets)]

    for confidence, outcome in zip(confidences, outcomes, strict=True):
        index = min(int(confidence * buckets), buckets - 1)
        bucket_totals[index] += 1
        bucket_confidence[index] += confidence
        bucket_accuracy[index] += float(outcome)

    total = len(confidences)
    error = 0.0
    for index in range(buckets):
        count = bucket_totals[index]
        if count == 0:
            continue
        avg_confidence = bucket_confidence[index] / count
        avg_accuracy = bucket_accuracy[index] / count
        error += (count / total) * abs(avg_accuracy - avg_confidence)
    return error


class ReflectionEvaluator:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def accuracy_snapshot(self) -> dict[str, AccuracyStats]:
        async with aiosqlite.connect(self._db_path) as connection:
            return {
                "all_time": await _query_accuracy(
                    connection,
                    "SELECT COUNT(*), COALESCE(SUM(is_correct_1d), 0) FROM predictions WHERE outcome_recorded = 1",
                ),
                "last_7d": await _query_accuracy(
                    connection,
                    """
                    SELECT COUNT(*), COALESCE(SUM(is_correct_1d), 0)
                    FROM predictions
                    WHERE outcome_recorded = 1
                      AND created_at > datetime('now', '-7 days')
                    """,
                ),
                "high_confidence": await _query_accuracy(
                    connection,
                    """
                    SELECT COUNT(*), COALESCE(SUM(is_correct_1d), 0)
                    FROM predictions
                    WHERE outcome_recorded = 1
                      AND confidence >= 0.7
                    """,
                ),
                "low_confidence": await _query_accuracy(
                    connection,
                    """
                    SELECT COUNT(*), COALESCE(SUM(is_correct_1d), 0)
                    FROM predictions
                    WHERE outcome_recorded = 1
                      AND confidence < 0.7
                    """,
                ),
            }

    async def latest_reflection(self) -> ReflectionRecord | None:
        async with aiosqlite.connect(self._db_path) as connection:
            cursor = await connection.execute(
                """
                SELECT id, created_at, reflection_type, period_start, period_end,
                       total_predictions, accuracy_1d, accuracy_5d, avg_confidence,
                       calibration_error, reflection_text, lessons_json
                FROM reflections
                WHERE reflection_type = 'weekly'
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = await cursor.fetchone()
        if row is None:
            return None

        return ReflectionRecord(
            id=int(row[0]),
            created_at=str(row[1]),
            reflection_type=str(row[2]),
            period_start=str(row[3]),
            period_end=str(row[4]),
            total_predictions=int(row[5]),
            accuracy_1d=float(row[6]) if row[6] is not None else None,
            accuracy_5d=float(row[7]) if row[7] is not None else None,
            avg_confidence=float(row[8]) if row[8] is not None else None,
            calibration_error=float(row[9]) if row[9] is not None else None,
            reflection_text=str(row[10]),
            lessons_json=str(row[11] or "[]"),
        )

    async def latest_lessons(self) -> list[str]:
        reflection = await self.latest_reflection()
        if reflection is None:
            return []
        try:
            lessons = json.loads(reflection.lessons_json)
        except json.JSONDecodeError:
            return []
        if not isinstance(lessons, list):
            return []
        return [str(lesson).strip() for lesson in lessons if str(lesson).strip()]


async def _query_accuracy(connection: aiosqlite.Connection, query: str) -> AccuracyStats:
    cursor = await connection.execute(query)
    row = await cursor.fetchone()
    total = int(row[0]) if row and row[0] is not None else 0
    correct = int(row[1]) if row and row[1] is not None else 0
    return AccuracyStats(total=total, correct=correct)
