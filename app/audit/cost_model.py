from __future__ import annotations

from dataclasses import dataclass

DEFAULT_FEE_BPS_ROUND_TRIP = 20.0
DEFAULT_SLIPPAGE_BPS_ROUND_TRIP = 10.0


@dataclass(frozen=True, slots=True)
class RoundTripCost:
    cost_bps: float
    cost_usd: float


def _settings_defaults() -> tuple[float, float]:
    try:
        from src.config import settings as project_settings

        fee_bps = float(getattr(project_settings, "fee_bps_round_trip", DEFAULT_FEE_BPS_ROUND_TRIP))
        slippage_bps = float(
            getattr(project_settings, "slippage_bps_round_trip", DEFAULT_SLIPPAGE_BPS_ROUND_TRIP)
        )
        return fee_bps, slippage_bps
    except Exception:
        return DEFAULT_FEE_BPS_ROUND_TRIP, DEFAULT_SLIPPAGE_BPS_ROUND_TRIP


def resolve_round_trip_cost_bps(
    *,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
) -> float:
    default_fee_bps, default_slippage_bps = _settings_defaults()
    resolved_fee_bps = default_fee_bps if fee_bps is None else float(fee_bps)
    resolved_slippage_bps = default_slippage_bps if slippage_bps is None else float(slippage_bps)
    return resolved_fee_bps + resolved_slippage_bps


def compute_round_trip_cost(
    *,
    notional_usd: float,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
) -> RoundTripCost:
    total_bps = resolve_round_trip_cost_bps(fee_bps=fee_bps, slippage_bps=slippage_bps)
    gross_notional = max(float(notional_usd), 0.0)
    cost_usd = round(gross_notional * total_bps / 10_000, 2)
    return RoundTripCost(cost_bps=total_bps, cost_usd=cost_usd)
