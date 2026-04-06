"""
Analyzes relative strength and short-term momentum from price heuristics.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agents.providers import LLMRole


SYSTEM_PROMPT = """You are a technical market analyst.
You receive current prices, 24h changes, and 7-day changes for stocks and crypto.

Produce a concise technical analysis in JSON:
{
  "summary": "2-3 sentence technical overview",
  "signals": ["signal 1", "signal 2", "signal 3"],
  "momentum_leaders": ["SYMBOL1", "SYMBOL2"],
  "momentum_laggards": ["SYMBOL3", "SYMBOL4"],
  "regime": "trending_up | trending_down | range_bound | mixed"
}

Focus on relative strength, momentum, and whether the market is trending or choppy.
Use only the supplied data. Do not invent levels or indicators."""


async def technical_analyst_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Analyze price momentum and regime using heuristic inputs."""
    prices = _dict_list(state.get("stock_prices")) + _dict_list(state.get("crypto_prices"))
    current_count = _int_or_zero(state.get("llm_call_count"))

    if not prices:
        return {
            "technical_report": {
                "summary": "No price data available.",
                "signals": [],
                "momentum_leaders": [],
                "momentum_laggards": [],
                "regime": "mixed",
            },
            "llm_call_count": current_count,
        }

    data_lines: list[str] = []
    for price in prices:
        symbol = str(price.get("symbol", "?"))
        current_price = price.get("current_price")
        change_24h = _float_or_zero(price.get("change_24h_pct"))
        line = f"  {symbol}: {_fmt_price(current_price)} 24h={change_24h:+.1f}%"
        change_7d = price.get("change_7d_pct")
        if isinstance(change_7d, int | float):
            line += f" 7d={float(change_7d):+.1f}%"
        data_lines.append(line)

    user_prompt = (
        "Today's price data:\n\n"
        f"{'\n'.join(data_lines)}\n\n"
        "Analyze momentum, relative strength, and the overall market regime."
    )
    result = await llm.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        role=LLMRole.GENERAL,
    )
    return {
        "technical_report": result,
        "llm_call_count": current_count + 1,
    }


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


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
