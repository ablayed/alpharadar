"""
Analyzes price movements, index performance, and SEC filings.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """You are a senior market analyst specializing in US tech stocks and crypto.
You receive today's price data, index performance, and any SEC filings.

Produce a concise market analysis in JSON format:
{
  "summary": "2-3 sentence market overview",
  "signals": ["signal 1", "signal 2", "signal 3"],
  "asset_highlights": {"SYMBOL": "one-line assessment"}
}

Focus on unusual moves (>3% stocks, >5% crypto), divergences between indices
and individual assets, and any SEC filings that could be material.
Be specific with numbers. Do not give financial advice.
Include at most 5 asset highlights."""


async def market_analyst_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Analyze market context and return a structured report."""
    stock_data = _dict_list(state.get("stock_prices"))
    crypto_data = _dict_list(state.get("crypto_prices"))
    index_data = _dict_list(state.get("index_prices"))
    filings = _dict_list(state.get("sec_filings"))
    market_open = bool(state.get("market_is_open", False))

    user_prompt = (
        f"Today's market data:\n\n"
        f"Market status: {'Open' if market_open else 'Closed'}\n\n"
        f"INDICES:\n{_format_prices(index_data, kind='index')}\n\n"
        f"STOCKS:\n{_format_prices(stock_data, kind='stock')}\n\n"
        f"CRYPTO:\n{_format_prices(crypto_data, kind='crypto')}\n\n"
        f"SEC FILINGS (last 24h):\n{_format_filings(filings)}\n\n"
        "Analyze the key market dynamics today."
    )

    result = await llm.generate_json(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    return {
        "market_report": result,
        "llm_call_count": _int_or_zero(state.get("llm_call_count")) + 1,
    }


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _format_prices(prices: list[dict[str, object]], *, kind: str) -> str:
    if not prices:
        return "No data available"

    lines: list[str] = []
    for price in prices:
        symbol = str(price.get("symbol", "?"))
        current_price = price.get("current_price")
        change_24h = _float_or_zero(price.get("change_24h_pct"))
        if kind == "index":
            lines.append(f"  {symbol}: {_fmt_price(current_price)} ({change_24h:+.1f}%)")
            continue

        change_7d = price.get("change_7d_pct")
        suffix = ""
        if isinstance(change_7d, int | float):
            suffix = f", 7d {float(change_7d):+.1f}%"
        lines.append(
            f"  {symbol}: {_fmt_price(current_price)} ({change_24h:+.1f}%{suffix})"
        )
    return "\n".join(lines)


def _format_filings(filings: list[dict[str, object]]) -> str:
    if not filings:
        return "None"
    lines = []
    for filing in filings:
        ticker = str(filing.get("ticker", "?"))
        form_type = str(filing.get("form_type", "?"))
        description = str(filing.get("description", "N/A"))
        lines.append(f"  {ticker}: {form_type} - {description}")
    return "\n".join(lines)


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
