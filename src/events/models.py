"""
Data models for corporate event detection and calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    """Corporate, crypto, and macro event taxonomy."""

    ACQUISITION = "acquisition"
    CLINICAL_TRIAL = "clinical_trial"
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    GUIDANCE_RAISE = "guidance_raise"
    GUIDANCE_CUT = "guidance_cut"
    NEW_CONTRACT = "new_contract"
    NEW_PRODUCT = "new_product"
    STOCK_BUYBACK = "stock_buyback"
    DIVIDEND_CHANGE = "dividend_change"
    EXECUTIVE_CHANGE = "executive_change"
    REGULATORY_ACTION = "regulatory_action"
    PARTNERSHIP = "partnership"
    LAYOFFS = "layoffs"
    LEGAL_ACTION = "legal_action"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"

    EXCHANGE_LISTING = "exchange_listing"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    HACK_EXPLOIT = "hack_exploit"
    TOKEN_UNLOCK = "token_unlock"
    WHALE_MOVEMENT = "whale_movement"
    REGULATORY_CRYPTO = "regulatory_crypto"
    DEFI_TVL_SHIFT = "defi_tvl_shift"

    FED_RATE_DECISION = "fed_rate_decision"
    INFLATION_DATA = "inflation_data"
    TRADE_POLICY = "trade_policy"
    GEOPOLITICAL = "geopolitical"

    GENERAL_NEWS = "general_news"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class DetectedEvent:
    """A classified event with attached calibration context."""

    event_type: EventType
    confidence: float
    headline: str
    related_symbols: list[str] = field(default_factory=list)
    expected_car_pct: float | None = None
    expected_car_window: str | None = None
    car_direction: str | None = None
    car_magnitude: str | None = None
    actual_move_pct: float | None = None
    remaining_drift_pct: float | None = None


@dataclass(slots=True)
class EventSummary:
    """Aggregate summary of detected events for a briefing."""

    total_events: int
    events_by_type: dict[str, int]
    highest_impact_event: DetectedEvent | None
    calibrated_signals: list[DetectedEvent] = field(default_factory=list)
