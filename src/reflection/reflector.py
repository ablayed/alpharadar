from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import aiosqlite

from src.agents.providers import LLMRole
from src.reflection.evaluator import expected_calibration_error

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are a financial analysis performance reviewer.
You review a week of market predictions vs actual outcomes.

Your job:
1. Identify patterns in correct predictions (what data signals were reliable)
2. Identify patterns in incorrect predictions (what went wrong, overconfidence, missed signals)
3. Check calibration: when confidence was high, was accuracy actually high?
4. Produce exactly 3 specific, actionable lessons for next week

Respond in JSON:
{
  "performance_summary": "2-3 sentences on overall accuracy",
  "best_call": "What was the best prediction and why it worked",
  "worst_call": "What was the worst miss and what was overlooked",
  "overconfidence_check": "Were high-confidence predictions actually more accurate?",
  "lessons": [
    "Lesson 1: specific and actionable",
    "Lesson 2: specific and actionable",
    "Lesson 3: specific and actionable"
  ],
  "bias_detected": "Any systematic bias (always bullish? ignoring macro? etc.)"
}"""


async def run_weekly_reflection(db_path: str, llm_client: object | None) -> str | None:
    if llm_client is None:
        logger.info("Weekly reflection skipped: no LLM client available")
        return None

    now = datetime.now(UTC)
    week_start = _sqlite_timestamp(now - timedelta(days=7))
    week_end = _sqlite_timestamp(now)

    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT symbol, predicted_direction, confidence, reasoning,
                   actual_direction, actual_change_1d_pct, is_correct_1d,
                   event_type, source, price_at_prediction, actual_price_1d,
                   actual_change_3d_pct, is_correct_3d,
                   actual_change_5d_pct, is_correct_5d,
                   actual_change_7d_pct, is_correct_7d
            FROM predictions
            WHERE outcome_recorded = 1
              AND created_at BETWEEN ? AND ?
            ORDER BY created_at
            """,
            (week_start, week_end),
        )
        rows = await cursor.fetchall()

    if len(rows) < 3:
        logger.info("Weekly reflection skipped: only %d resolved predictions", len(rows))
        return None

    accuracy_1d, correct_1d, total_1d = _accuracy_summary(rows, outcome_index=6)
    accuracy_3d, correct_3d, total_3d = _accuracy_summary(rows, outcome_index=12)
    accuracy_5d, correct_5d, total_5d = _accuracy_summary(rows, outcome_index=14)
    accuracy_7d, correct_7d, total_7d = _accuracy_summary(rows, outcome_index=16)
    total = len(rows)
    avg_confidence = sum(float(row[2]) for row in rows) / total
    calibration_error = expected_calibration_error(
        [float(row[2]) for row in rows],
        [int(row[6] or 0) for row in rows],
    )

    high_confidence_rows = [row for row in rows if float(row[2]) >= 0.7]
    low_confidence_rows = [row for row in rows if float(row[2]) < 0.7]
    high_accuracy = _accuracy_pct(high_confidence_rows)
    low_accuracy = _accuracy_pct(low_confidence_rows)

    hits_text = "\n".join(
        f"  {row[0]}: predicted {row[1]} ({float(row[2]):.0%}) -> {row[4]} "
        f"(1d {float(row[5] or 0):+.1f}%, 3d {float(row[11] or 0):+.1f}%, 7d {float(row[15] or 0):+.1f}%)"
        for row in rows
        if int(row[6] or 0) == 1
    ) or "  None"
    misses_text = "\n".join(
        f"  {row[0]}: predicted {row[1]} ({float(row[2]):.0%}) -> {row[4]} "
        f"(1d {float(row[5] or 0):+.1f}%, 3d {float(row[11] or 0):+.1f}%, 7d {float(row[15] or 0):+.1f}%)"
        for row in rows
        if int(row[6] or 0) == 0
    ) or "  None"

    user_prompt = f"""Week: {week_start[:10]} to {week_end[:10]}
Total predictions: {total}
Prediction outcomes at multiple timeframes:
- 1-day accuracy: {accuracy_1d:.1f}% ({correct_1d}/{total_1d})
- 3-day accuracy: {accuracy_3d:.1f}% ({correct_3d}/{total_3d})
- 5-day accuracy: {accuracy_5d:.1f}% ({correct_5d}/{total_5d})
- 7-day accuracy: {accuracy_7d:.1f}% ({correct_7d}/{total_7d})

If 7-day accuracy is higher than 1-day accuracy, the bot's directional calls are correct but the timing is early. Consider wider entry zones or longer hold periods.
If 7-day accuracy is lower than 1-day accuracy, the bot catches short-term moves but reversals happen later. Consider tighter take profits.

Average confidence: {avg_confidence:.0%}
Calibration error: {calibration_error:.3f}
High confidence (>=70%) accuracy: {high_accuracy:.1f}% ({len(high_confidence_rows)} predictions)
Low confidence (<70%) accuracy: {low_accuracy:.1f}% ({len(low_confidence_rows)} predictions)

CORRECT CALLS:
{hits_text}

INCORRECT CALLS:
{misses_text}

Analyze these results and provide the weekly reflection."""

    try:
        result = await llm_client.generate_json(
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=800,
            role=LLMRole.REFLECTION,
        )
        reflection_payload = result if isinstance(result, dict) else {"raw": str(result)}
        reflection_text = json.dumps(reflection_payload, indent=2)
        lessons_json = json.dumps(reflection_payload.get("lessons", []))

        async with aiosqlite.connect(db_path) as connection:
            await connection.execute(
                """
                INSERT INTO reflections
                (reflection_type, period_start, period_end, total_predictions,
                 accuracy_1d, accuracy_5d, avg_confidence, calibration_error, reflection_text, lessons_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "weekly",
                    week_start,
                    week_end,
                    total,
                    accuracy_1d,
                    accuracy_5d,
                    avg_confidence,
                    calibration_error,
                    reflection_text,
                    lessons_json,
                ),
            )
            await connection.commit()

        logger.info(
            "Weekly reflection complete: %d predictions, %.1f%% accuracy",
            total,
            accuracy_1d,
        )
        return reflection_text
    except Exception:
        logger.exception("Weekly reflection LLM call failed")
        return None


async def get_latest_lessons(db_path: str) -> str:
    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT lessons_json
            FROM reflections
            WHERE reflection_type = 'weekly'
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = await cursor.fetchone()
    if row is None or row[0] is None:
        return ""

    try:
        lessons = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return ""
    if not isinstance(lessons, list):
        return ""

    cleaned = [str(lesson).strip() for lesson in lessons if str(lesson).strip()]
    if not cleaned:
        return ""
    return "\n\nLESSONS FROM LAST WEEK'S REVIEW:\n" + "\n".join(f"- {lesson}" for lesson in cleaned)


def _accuracy_pct(rows: list[tuple[object, ...]]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if int(row[6] or 0) == 1)
    return (correct / len(rows)) * 100.0


def _accuracy_summary(
    rows: list[tuple[object, ...]],
    *,
    outcome_index: int,
) -> tuple[float, int, int]:
    eligible_rows = [row for row in rows if row[outcome_index] is not None]
    if not eligible_rows:
        return 0.0, 0, 0

    correct = sum(1 for row in eligible_rows if int(row[outcome_index] or 0) == 1)
    total = len(eligible_rows)
    return (correct / total) * 100.0, correct, total


def _sqlite_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
