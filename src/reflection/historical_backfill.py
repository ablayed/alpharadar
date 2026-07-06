from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta

import aiosqlite
import httpx

from src.ta.candles import COINGECKO_BASE, FINNHUB_BASE

HistoricalPriceFetcher = Callable[[str, str, datetime], Awaitable[float | None]]


class RepositoryHistoricalPriceFetcher:
    def __init__(
        self,
        *,
        db_path: str,
        finnhub_api_key: str | None = None,
    ) -> None:
        self._db_path = db_path
        self._finnhub_api_key = (finnhub_api_key or os.getenv("FINNHUB_API_KEY", "")).strip() or None
        self._coin_id_cache: dict[str, str | None] = {}
        self._price_cache: dict[tuple[str, str, str], float | None] = {}

    async def __call__(self, symbol: str, asset_type: str, target_at: datetime) -> float | None:
        target_date = target_at.astimezone(UTC).date().isoformat()
        cache_key = (symbol.upper(), asset_type.lower(), target_date)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        normalized_asset_type = asset_type.lower()
        if normalized_asset_type == "stock":
            price = await self._fetch_stock_price(symbol.upper(), target_at)
        elif normalized_asset_type == "crypto":
            price = await self._fetch_crypto_price(symbol.upper(), target_at)
        else:
            price = None

        self._price_cache[cache_key] = price
        return price

    async def _fetch_stock_price(self, symbol: str, target_at: datetime) -> float | None:
        if self._finnhub_api_key:
            price = await _fetch_finnhub_stock_close(symbol, target_at, self._finnhub_api_key)
            if price is not None:
                return price
        return await asyncio.to_thread(_download_yfinance_stock_close, symbol, target_at)

    async def _fetch_crypto_price(self, symbol: str, target_at: datetime) -> float | None:
        coin_id = await self._coin_id_for_symbol(symbol)
        if not coin_id:
            return None
        return await _fetch_coingecko_crypto_close(coin_id, target_at)

    async def _coin_id_for_symbol(self, symbol: str) -> str | None:
        symbol_upper = symbol.upper()
        if symbol_upper in self._coin_id_cache:
            return self._coin_id_cache[symbol_upper]

        async with aiosqlite.connect(self._db_path) as connection:
            cursor = await connection.execute(
                """
                SELECT coingecko_id
                FROM watchlist
                WHERE symbol = ? AND asset_type = 'crypto' AND is_active = 1 AND coingecko_id IS NOT NULL
                LIMIT 1
                """,
                (symbol_upper,),
            )
            row = await cursor.fetchone()
            if row is None:
                cursor = await connection.execute(
                    """
                    SELECT coingecko_id
                    FROM hot_list
                    WHERE symbol = ? AND coingecko_id IS NOT NULL
                    ORDER BY
                        CASE WHEN status = 'active' THEN 0 ELSE 1 END,
                        added_at DESC
                    LIMIT 1
                    """,
                    (symbol_upper,),
                )
                row = await cursor.fetchone()

        coin_id = None if row is None or row[0] is None else str(row[0])
        self._coin_id_cache[symbol_upper] = coin_id
        return coin_id


async def _fetch_finnhub_stock_close(symbol: str, target_at: datetime, api_key: str) -> float | None:
    from_timestamp = int((target_at - timedelta(days=7)).timestamp())
    to_timestamp = int((target_at + timedelta(days=7)).timestamp())
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{FINNHUB_BASE}/stock/candle",
            params={
                "symbol": symbol,
                "resolution": "D",
                "from": from_timestamp,
                "to": to_timestamp,
                "token": api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()

    closes = payload.get("c")
    timestamps = payload.get("t")
    if payload.get("s") != "ok" or not isinstance(closes, list) or not isinstance(timestamps, list):
        return None
    return _first_daily_close_on_or_after(target_at.date(), timestamps=timestamps, closes=closes)


def _download_yfinance_stock_close(symbol: str, target_at: datetime) -> float | None:
    try:
        import yfinance as yf
    except ImportError:
        return None

    start = (target_at - timedelta(days=7)).date().isoformat()
    end = (target_at + timedelta(days=8)).date().isoformat()
    history = yf.Ticker(symbol).history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
    )
    if history.empty or "Close" not in history:
        return None

    target_date = target_at.astimezone(UTC).date()
    for index, row in history.iterrows():
        if index.date() >= target_date:
            close_value = row.get("Close")
            if close_value is None:
                return None
            return float(close_value)
    return None


async def _fetch_coingecko_crypto_close(coin_id: str, target_at: datetime) -> float | None:
    now = datetime.now(UTC)
    days = max(7, math.ceil((now - target_at).total_seconds() / 86400) + 7)
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, list):
        return None

    timestamps: list[int] = []
    closes: list[float] = []
    for candle in payload:
        if not isinstance(candle, list) or len(candle) < 5:
            continue
        try:
            timestamps.append(int(candle[0]) // 1000)
            closes.append(float(candle[4]))
        except (TypeError, ValueError):
            continue

    if not timestamps:
        return None
    return _first_daily_close_on_or_after(target_at.date(), timestamps=timestamps, closes=closes)


def _first_daily_close_on_or_after(
    target_date: date,
    *,
    timestamps: list[int],
    closes: list[float],
) -> float | None:
    for unix_timestamp, close in zip(timestamps, closes, strict=False):
        candle_date = datetime.fromtimestamp(unix_timestamp, UTC).date()
        if candle_date >= target_date:
            return float(close)
    return None
