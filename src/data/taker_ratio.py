"""
Fetch taker buy and sell pressure from Binance public endpoints.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

BINANCE_AGG_TRADES_URL = "https://data-api.binance.vision/api/v3/aggTrades"


async def fetch_taker_buy_ratio(symbol: str, limit: int = 500) -> float | None:
    params = {"symbol": symbol.upper(), "limit": min(max(limit, 1), 1000)}
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(BINANCE_AGG_TRADES_URL, params=params)
            response.raise_for_status()
            trades = response.json()
    except Exception:
        logger.debug("Failed to fetch taker ratio for %s", symbol.upper(), exc_info=True)
        return None

    if not isinstance(trades, list) or not trades:
        return None

    total_qty = 0.0
    buyer_qty = 0.0
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        qty = float(trade.get("q", 0) or 0)
        total_qty += qty
        if not bool(trade.get("m", True)):
            buyer_qty += qty

    if total_qty <= 0:
        return None
    return round(buyer_qty / total_qty, 4)


async def fetch_taker_ratios_batch(symbols: list[str]) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for index in range(0, len(symbols), 5):
        batch = symbols[index : index + 5]
        ratios = await asyncio.gather(
            *(fetch_taker_buy_ratio(symbol) for symbol in batch),
            return_exceptions=True,
        )
        for symbol, ratio in zip(batch, ratios, strict=True):
            results[symbol] = None if isinstance(ratio, Exception) else ratio
        if index + 5 < len(symbols):
            await asyncio.sleep(1)
    return results
