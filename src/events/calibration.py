"""
Cumulative abnormal return calibration for detected event types.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.events.models import EventType


@dataclass(slots=True)
class CARProfile:
    """Historical CAR profile for an event type."""

    avg_car_pct: float
    direction: str
    magnitude: str
    event_window: str
    typical_reaction_speed: str
    confidence_note: str


CAR_TABLE: dict[EventType, CARProfile] = {
    EventType.ACQUISITION: CARProfile(
        avg_car_pct=2.3,
        direction="positive",
        magnitude="moderate",
        event_window="[-1, +5] days",
        typical_reaction_speed="hours",
        confidence_note="Acquirer effect; targets usually react much more.",
    ),
    EventType.CLINICAL_TRIAL: CARProfile(
        avg_car_pct=5.0,
        direction="positive",
        magnitude="large",
        event_window="[0, +1] days",
        typical_reaction_speed="immediate",
        confidence_note="Positive trial outcomes react fast; failed trials can be far worse.",
    ),
    EventType.EARNINGS_BEAT: CARProfile(
        avg_car_pct=1.8,
        direction="positive",
        magnitude="moderate",
        event_window="[-1, +5] days",
        typical_reaction_speed="immediate",
        confidence_note="Post-earnings drift can extend well beyond the first session.",
    ),
    EventType.EARNINGS_MISS: CARProfile(
        avg_car_pct=-2.5,
        direction="negative",
        magnitude="moderate",
        event_window="[-1, +5] days",
        typical_reaction_speed="immediate",
        confidence_note="Negative drift usually lasts longer than positive surprises.",
    ),
    EventType.GUIDANCE_RAISE: CARProfile(
        avg_car_pct=2.0,
        direction="positive",
        magnitude="moderate",
        event_window="[-1, +3] days",
        typical_reaction_speed="immediate",
        confidence_note="Guidance revisions often matter more than the headline EPS number.",
    ),
    EventType.GUIDANCE_CUT: CARProfile(
        avg_car_pct=-3.5,
        direction="negative",
        magnitude="large",
        event_window="[-1, +5] days",
        typical_reaction_speed="immediate",
        confidence_note="Guidance cuts often cascade into downgrades and estimate resets.",
    ),
    EventType.NEW_CONTRACT: CARProfile(
        avg_car_pct=1.2,
        direction="positive",
        magnitude="small",
        event_window="[-1, +3] days",
        typical_reaction_speed="hours",
        confidence_note="Contract size relative to revenue base matters most.",
    ),
    EventType.NEW_PRODUCT: CARProfile(
        avg_car_pct=0.8,
        direction="positive",
        magnitude="small",
        event_window="[-1, +5] days",
        typical_reaction_speed="days",
        confidence_note="Launch hype is often partly priced in before release.",
    ),
    EventType.STOCK_BUYBACK: CARProfile(
        avg_car_pct=1.5,
        direction="positive",
        magnitude="moderate",
        event_window="[-1, +5] days",
        typical_reaction_speed="hours",
        confidence_note="Buybacks are usually read as a management confidence signal.",
    ),
    EventType.DIVIDEND_CHANGE: CARProfile(
        avg_car_pct=-4.0,
        direction="negative",
        magnitude="large",
        event_window="[-1, +3] days",
        typical_reaction_speed="immediate",
        confidence_note="Cuts drive the average; increases are typically milder.",
    ),
    EventType.EXECUTIVE_CHANGE: CARProfile(
        avg_car_pct=0.5,
        direction="neutral",
        magnitude="small",
        event_window="[-1, +5] days",
        typical_reaction_speed="hours",
        confidence_note="Forced departures are usually worse than planned transitions.",
    ),
    EventType.REGULATORY_ACTION: CARProfile(
        avg_car_pct=-1.5,
        direction="negative",
        magnitude="moderate",
        event_window="[-1, +10] days",
        typical_reaction_speed="hours",
        confidence_note="Severity varies widely; antitrust and enforcement tend to hit hardest.",
    ),
    EventType.PARTNERSHIP: CARProfile(
        avg_car_pct=1.0,
        direction="positive",
        magnitude="small",
        event_window="[-1, +3] days",
        typical_reaction_speed="hours",
        confidence_note="Smaller counterparties often react more than mega-caps.",
    ),
    EventType.LAYOFFS: CARProfile(
        avg_car_pct=0.5,
        direction="neutral",
        magnitude="small",
        event_window="[-1, +5] days",
        typical_reaction_speed="hours",
        confidence_note="Markets often like cost cuts even when the headline reads negatively.",
    ),
    EventType.LEGAL_ACTION: CARProfile(
        avg_car_pct=-1.0,
        direction="negative",
        magnitude="small",
        event_window="[-1, +5] days",
        typical_reaction_speed="days",
        confidence_note="Litigation impact can bleed in gradually as facts develop.",
    ),
    EventType.ANALYST_UPGRADE: CARProfile(
        avg_car_pct=1.5,
        direction="positive",
        magnitude="moderate",
        event_window="[0, +3] days",
        typical_reaction_speed="immediate",
        confidence_note="Tier-1 analysts with target changes tend to matter most.",
    ),
    EventType.ANALYST_DOWNGRADE: CARProfile(
        avg_car_pct=-2.0,
        direction="negative",
        magnitude="moderate",
        event_window="[0, +3] days",
        typical_reaction_speed="immediate",
        confidence_note="Downgrades usually move price more than upgrades.",
    ),
    EventType.EXCHANGE_LISTING: CARProfile(
        avg_car_pct=8.0,
        direction="positive",
        magnitude="large",
        event_window="[-1, +3] days",
        typical_reaction_speed="immediate",
        confidence_note="Major venue listings still matter, but the effect has faded over time.",
    ),
    EventType.PROTOCOL_UPGRADE: CARProfile(
        avg_car_pct=3.0,
        direction="positive",
        magnitude="moderate",
        event_window="[-7, +3] days",
        typical_reaction_speed="days",
        confidence_note="Run-up before the upgrade and sell-the-news behavior are common.",
    ),
    EventType.HACK_EXPLOIT: CARProfile(
        avg_car_pct=-15.0,
        direction="negative",
        magnitude="large",
        event_window="[0, +3] days",
        typical_reaction_speed="immediate",
        confidence_note="Loss size versus TVL and treasury determines the depth of damage.",
    ),
    EventType.TOKEN_UNLOCK: CARProfile(
        avg_car_pct=-3.0,
        direction="negative",
        magnitude="moderate",
        event_window="[-3, +3] days",
        typical_reaction_speed="days",
        confidence_note="Unlock pressure is often partly known in advance.",
    ),
    EventType.WHALE_MOVEMENT: CARProfile(
        avg_car_pct=2.0,
        direction="neutral",
        magnitude="small",
        event_window="[0, +1] days",
        typical_reaction_speed="hours",
        confidence_note="Direction depends on whether the flow implies accumulation or selling.",
    ),
    EventType.REGULATORY_CRYPTO: CARProfile(
        avg_car_pct=-5.0,
        direction="negative",
        magnitude="large",
        event_window="[0, +5] days",
        typical_reaction_speed="immediate",
        confidence_note="Restrictive actions dominate the average; constructive regulation can reverse it.",
    ),
    EventType.DEFI_TVL_SHIFT: CARProfile(
        avg_car_pct=2.5,
        direction="positive",
        magnitude="moderate",
        event_window="[-1, +3] days",
        typical_reaction_speed="hours",
        confidence_note="TVL acceleration usually matters more when it confirms a fresh narrative.",
    ),
    EventType.FED_RATE_DECISION: CARProfile(
        avg_car_pct=0.5,
        direction="neutral",
        magnitude="small",
        event_window="[0, +2] days",
        typical_reaction_speed="immediate",
        confidence_note="The surprise versus expectations matters more than the headline move.",
    ),
    EventType.INFLATION_DATA: CARProfile(
        avg_car_pct=0.3,
        direction="neutral",
        magnitude="small",
        event_window="[0, +1] days",
        typical_reaction_speed="immediate",
        confidence_note="Direction flips with the inflation surprise versus consensus.",
    ),
    EventType.TRADE_POLICY: CARProfile(
        avg_car_pct=-1.0,
        direction="negative",
        magnitude="small",
        event_window="[0, +5] days",
        typical_reaction_speed="hours",
        confidence_note="Tariffs and sanctions weigh more than trade-deal headlines help.",
    ),
    EventType.GEOPOLITICAL: CARProfile(
        avg_car_pct=-1.5,
        direction="negative",
        magnitude="moderate",
        event_window="[0, +5] days",
        typical_reaction_speed="immediate",
        confidence_note="Flight-to-safety reactions are often sharp but temporary.",
    ),
    EventType.GENERAL_NEWS: CARProfile(
        avg_car_pct=0.0,
        direction="neutral",
        magnitude="small",
        event_window="N/A",
        typical_reaction_speed="N/A",
        confidence_note="No calibrated event pattern available.",
    ),
    EventType.UNKNOWN: CARProfile(
        avg_car_pct=0.0,
        direction="neutral",
        magnitude="small",
        event_window="N/A",
        typical_reaction_speed="N/A",
        confidence_note="The headline could not be mapped reliably.",
    ),
}


def get_car_profile(event_type: EventType) -> CARProfile:
    """Return the CAR profile for a detected event."""

    return CAR_TABLE.get(event_type, CAR_TABLE[EventType.UNKNOWN])


def compute_remaining_drift(
    event_type: EventType,
    actual_move_pct: float,
) -> tuple[float | None, str]:
    """Compare the current move with the historical average CAR."""

    profile = get_car_profile(event_type)
    if profile.avg_car_pct == 0.0:
        return None, "No historical calibration available for this event type."

    expected = profile.avg_car_pct
    remaining = expected - actual_move_pct
    captured_pct = (actual_move_pct / expected * 100.0) if expected != 0 else 0.0

    if abs(remaining) < 0.3:
        interpretation = (
            f"Price has largely captured the typical {profile.direction} drift for "
            f"{event_type.value} events ({expected:+.1f}% over {profile.event_window})."
        )
    elif (remaining > 0 and expected > 0) or (remaining < 0 and expected < 0):
        interpretation = (
            f"Price has captured {abs(captured_pct):.0f}% of typical {event_type.value} drift "
            f"({expected:+.1f}% avg). Historical pattern suggests {remaining:+.1f}% remaining "
            f"over {profile.event_window}. {profile.confidence_note}"
        )
    else:
        interpretation = (
            f"Price has overshot the typical {event_type.value} drift ({expected:+.1f}% avg) "
            f"by {abs(remaining):.1f}pp. Historical mean reversion is possible. "
            f"{profile.confidence_note}"
        )

    return remaining, interpretation


def format_remaining_drift_status(
    event_type: EventType,
    actual_move_pct: float | None,
    *,
    capture_threshold: float = 0.5,
) -> str:
    """Return a concise one-line status for briefing and `/events` output."""

    if actual_move_pct is None:
        return "current move unavailable"

    profile = get_car_profile(event_type)
    remaining, _ = compute_remaining_drift(event_type, actual_move_pct)
    if remaining is None:
        return "no calibrated drift"

    if abs(remaining) <= capture_threshold:
        return "typical drift captured"
    if (profile.avg_car_pct > 0 and remaining > 0) or (profile.avg_car_pct < 0 and remaining < 0):
        direction = "upside" if remaining > 0 else "downside"
        return f"remaining {abs(remaining):.1f}% {direction} expected"
    return f"overshot by {abs(remaining):.1f}pp"
