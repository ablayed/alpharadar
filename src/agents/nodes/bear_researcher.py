"""
Builds the strongest bear case for selected movers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """You are a bearish financial researcher.
Your job is to construct the strongest possible bear case for the given asset.
You have already seen the bull case and must directly counter it with opposing
evidence and risk analysis.

You must argue the bear case even if you think the bull case is stronger.
This is adversarial analysis. Do not provide financial advice.

Respond in JSON:
{
  "thesis": "1-2 sentence core bear argument",
  "evidence": ["point 1", "point 2", "point 3"],
  "bull_weaknesses": ["weakness 1", "weakness 2"],
  "confidence": 0.0,
  "risk_trigger": "specific event that could prove bears right"
}"""


async def bear_researcher_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Generate bearish arguments that counter the bull case."""
    movers = _dict_list(state.get("selected_movers"))
    bull_args = _dict_list(state.get("bull_arguments"))
    market_report = _as_mapping(state.get("market_report"))

    arguments: list[dict[str, object]] = []
    call_count = _int_or_zero(state.get("llm_call_count"))

    all_prices = _dict_list(state.get("stock_prices")) + _dict_list(state.get("crypto_prices"))
    for index, mover in enumerate(movers):
        symbol = str(mover.get("symbol", "?"))
        asset_type = str(mover.get("asset_type", "unknown"))
        price_data = next(
            (item for item in all_prices if str(item.get("symbol", "")).upper() == symbol.upper()),
            {},
        )
        bull_case = bull_args[index] if index < len(bull_args) else {}

        user_prompt = f"""Asset: {symbol} ({asset_type})
Current price: {_fmt_price(price_data.get('current_price'))}
24h change: {_float_or_zero(price_data.get('change_24h_pct')):+.1f}%

The bull case argues: {bull_case.get('thesis', 'N/A')}
Bull evidence: {bull_case.get('evidence', [])}
Bull confidence: {bull_case.get('confidence', 'N/A')}

Market context: {market_report.get('summary', 'N/A')}

Construct the strongest bear case for {symbol}, directly countering the bull thesis."""

        result = await llm.generate_json(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        result["symbol"] = symbol
        arguments.append(result)
        call_count += 1

    return {"bear_arguments": arguments, "llm_call_count": call_count}


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


def _int_or_zero(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
