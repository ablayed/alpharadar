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


@dataclass(slots=True)
class AccuracySnapshot:
    all_time_predictions: int
    one_day: AccuracyStats
    three_day: AccuracyStats
    seven_day: AccuracyStats
    high_confidence_1d: AccuracyStats
    low_confidence_1d: AccuracyStats
    by_source_1d: dict[str, AccuracyStats]


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

    async def accuracy_snapshot(self) -> AccuracySnapshot:
        async with aiosqlite.connect(self._db_path) as connection:
            return AccuracySnapshot(
                all_time_predictions=await _query_total_predictions(connection),
                one_day=await _query_accuracy(
                    connection,
                    correctness_column="is_correct_1d",
                    where_clause="is_correct_1d IS NOT NULL",
                ),
                three_day=await _query_accuracy(
                    connection,
                    correctness_column="is_correct_3d",
                    where_clause="is_correct_3d IS NOT NULL",
                ),
                seven_day=await _query_accuracy(
                    connection,
                    correctness_column="is_correct_7d",
                    where_clause="is_correct_7d IS NOT NULL",
                ),
                high_confidence_1d=await _query_accuracy(
                    connection,
                    correctness_column="is_correct_1d",
                    where_clause="is_correct_1d IS NOT NULL AND confidence >= 0.7",
                ),
                low_confidence_1d=await _query_accuracy(
                    connection,
                    correctness_column="is_correct_1d",
                    where_clause="is_correct_1d IS NOT NULL AND confidence < 0.7",
                ),
                by_source_1d=await _query_accuracy_by_source(
                    connection,
                    correctness_column="is_correct_1d",
                ),
            )

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


async def _query_total_predictions(connection: aiosqlite.Connection) -> int:
    cursor = await connection.execute("SELECT COUNT(*) FROM predictions")
    row = await cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


async def _query_accuracy(
    connection: aiosqlite.Connection,
    *,
    correctness_column: str,
    where_clause: str,
) -> AccuracyStats:
    cursor = await connection.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM({correctness_column}), 0)
        FROM predictions
        WHERE {where_clause}
        """
    )
    row = await cursor.fetchone()
    total = int(row[0]) if row and row[0] is not None else 0
    correct = int(row[1]) if row and row[1] is not None else 0
    return AccuracyStats(total=total, correct=correct)


async def _query_accuracy_by_source(
    connection: aiosqlite.Connection,
    *,
    correctness_column: str,
) -> dict[str, AccuracyStats]:
    cursor = await connection.execute(
        f"""
        SELECT source, COUNT(*), COALESCE(SUM({correctness_column}), 0)
        FROM predictions
        WHERE {correctness_column} IS NOT NULL
        GROUP BY source
        ORDER BY source
        """
    )
    rows = await cursor.fetchall()
    return {
        str(row[0]): AccuracyStats(
            total=int(row[1]) if row[1] is not None else 0,
            correct=int(row[2]) if row[2] is not None else 0,
        )
        for row in rows
    }
