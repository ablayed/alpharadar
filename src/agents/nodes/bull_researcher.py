"""
Builds the strongest bull case for selected movers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """You are a bullish financial researcher.
Your job is to construct an aggressive, specific bull case for the given asset,
using the provided market data and analyst reports as evidence.

You MUST include:
- A specific upside price target with rationale
- A specific catalyst with expected timeframe
- A concrete data point from the provided market data

Do NOT use vague phrases like "poised for growth" without backing numbers.

If calibrated event-study data is available, treat it as objective evidence:
use supportive CAR data directly and acknowledge adverse CAR data as a risk.

You must argue the bull case even if you think the bear case is stronger.
This is a structured debate. Do not provide financial advice.
Confidence should reflect how compelling the bullish evidence is:
0.5 = slight lean, 0.7 = moderate conviction, 0.9 = strong conviction.

Respond in JSON:
{
  "thesis": "1-2 sentence core bull argument with a specific upside level when supported",
  "evidence": ["point 1", "point 2", "point 3"],
  "risks_acknowledged": ["risk that bulls accept"],
  "confidence": 0.7,
  "catalyst": "specific near-term event that could prove bulls right, with timeframe"
}"""


async def bull_researcher_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Generate bullish arguments for each selected mover."""
    movers = _dict_list(state.get("selected_movers"))
    market_report = _as_mapping(state.get("market_report"))
    sentiment_report = _as_mapping(state.get("sentiment_report"))
    technical_report = _as_mapping(state.get("technical_report"))
    event_data = _dict_list(state.get("event_data"))

    arguments: list[dict[str, object]] = []
    call_count = _int_or_zero(state.get("llm_call_count"))

    all_prices = _dict_list(state.get("stock_prices")) + _dict_list(state.get("crypto_prices"))
    for mover in movers:
        symbol = str(mover.get("symbol", "?"))
        asset_type = str(mover.get("asset_type", "unknown"))
        reason = str(mover.get("reason", "Selected mover"))
        price_data = next(
            (item for item in all_prices if str(item.get("symbol", "")).upper() == symbol.upper()),
            {},
        )
        relevant_events = [
            event
            for event in event_data
            if symbol.upper() in {related.upper() for related in _string_list(event.get("related_symbols"))}
        ]

        user_prompt = f"""Asset: {symbol} ({asset_type})
Selection reason: {reason}
Current price: {_fmt_price(price_data.get('current_price'))}
24h change: {_float_or_zero(price_data.get('change_24h_pct')):+.1f}%

Market analyst says: {market_report.get('summary', 'N/A')}
Sentiment analyst says: {sentiment_report.get('summary', 'N/A')}
Technical analyst says: {technical_report.get('summary', 'N/A')}

Calibrated event data:
{_format_symbol_events(relevant_events)}

Construct an aggressive, specific bull case for {symbol} with concrete upside levels and catalysts where the data supports them."""

        result = await llm.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        result["symbol"] = symbol
        arguments.append(result)
        call_count += 1

    return {"bull_arguments": arguments, "llm_call_count": call_count}


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _fmt_price(value: object) -> str:
    if isinstance(value, int | float):
        numeric = float(value)
        if abs(numeric) >= 1_000:
            return f"${numeric:,.0f}"
        if abs(numeric) >= 1:
            return f"${numeric:,.2f}"
        return f"${numeric:.4g}"
    return "n/a"


def _float_or_zero(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _format_symbol_events(events: list[dict[str, object]]) -> str:
    if not events:
        return "  None"

    lines: list[str] = []
    for event in events:
        remaining = event.get("remaining_drift_pct")
        remaining_text = "n/a"
        if isinstance(remaining, int | float):
            remaining_text = f"{float(remaining):+.1f}%"
        lines.append(
            "  "
            f"{event.get('event_type', 'unknown')} "
            f"({(_float_or_zero(event.get('confidence'))):.0%} confidence) | "
            f"hist {_float_or_zero(event.get('expected_car_pct')):+.1f}% over "
            f"{event.get('expected_car_window', 'N/A')} | "
            f"current {_float_or_zero(event.get('actual_move_pct')):+.1f}% | "
            f"remaining {remaining_text}"
        )
    return "\n".join(lines)


def _int_or_zero(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
