from __future__ import annotations

from app.audit.cost_model import compute_round_trip_cost


def test_round_trip_cost_basic() -> None:
    result = compute_round_trip_cost(notional_usd=1000, fee_bps=20, slippage_bps=10)
    assert result.cost_bps == 30
    assert result.cost_usd == 3.00


def test_round_trip_cost_zero_notional() -> None:
    result = compute_round_trip_cost(notional_usd=0, fee_bps=20, slippage_bps=10)
    assert result.cost_usd == 0.00


def test_round_trip_cost_config_default() -> None:
    result = compute_round_trip_cost(notional_usd=1000)
    assert result.cost_bps == 30
    assert result.cost_usd == 3.00


def test_cost_model_is_deterministic() -> None:
    first = compute_round_trip_cost(notional_usd=1000, fee_bps=20, slippage_bps=10)
    second = compute_round_trip_cost(notional_usd=1000, fee_bps=20, slippage_bps=10)
    assert first == second
