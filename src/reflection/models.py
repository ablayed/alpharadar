from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PredictionRecord:
    symbol: str
    asset_type: str
    predicted_direction: str
    confidence: float
    price_at_prediction: float
    source: str
    reasoning: str = ""
    event_type: str | None = None
    predicted_target: float | None = None
    id: int | None = None
    created_at: str = ""
    outcome_recorded: bool = False
    actual_price_1d: float | None = None
    actual_price_5d: float | None = None
    actual_direction: str | None = None
    actual_change_1d_pct: float | None = None
    actual_change_5d_pct: float | None = None
    is_correct_1d: int | None = None
    is_correct_5d: int | None = None
    outcome_recorded_at: str | None = None


@dataclass(slots=True)
class ReflectionRecord:
    reflection_type: str
    period_start: str
    period_end: str
    total_predictions: int
    accuracy_1d: float | None
    accuracy_5d: float | None
    avg_confidence: float | None
    calibration_error: float | None
    reflection_text: str
    lessons_json: str = "[]"
    id: int | None = None
    created_at: str = ""
