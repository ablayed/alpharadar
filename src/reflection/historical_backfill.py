from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta

import aiosqlite
import httpx

from src.data.aggregator import CRYPTO_SYMBOL_ALIASES
from src.ta.candles import COINGECKO_BASE, FINNHUB_BASE

HistoricalPriceFetcher = Callable[[str, str, datetime], Awaitable[float | None]]
logger = logging.getLogger(__name__)


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
        self._coingecko_min_interval_seconds = 1.5
        self._last_coingecko_request_monotonic: float | None = None

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
            try:
                price = await _fetch_finnhub_stock_close(symbol, target_at, self._finnhub_api_key)
            except Exception:
                logger.warning(
                    "Historical Finnhub lookup failed for %s at %s",
                    symbol,
                    target_at.date().isoformat(),
                    exc_info=True,
                )
                price = None
            if price is not None:
                return price
        try:
            return await asyncio.to_thread(_download_yfinance_stock_close, symbol, target_at)
        except Exception:
            logger.warning(
                "Historical yfinance lookup failed for %s at %s",
                symbol,
                target_at.date().isoformat(),
                exc_info=True,
            )
            return None

    async def _fetch_crypto_price(self, symbol: str, target_at: datetime) -> float | None:
        coin_id = await self._coin_id_for_symbol(symbol)
        if not coin_id:
            return None
        await self._respect_coingecko_rate_limit()
        return await _fetch_coingecko_crypto_close(coin_id, target_at)

    async def _respect_coingecko_rate_limit(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._last_coingecko_request_monotonic is not None:
            elapsed = now - self._last_coingecko_request_monotonic
            wait_seconds = self._coingecko_min_interval_seconds - elapsed
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = loop.time()
        self._last_coingecko_request_monotonic = now

    async def _coin_id_for_symbol(self, symbol: str) -> str | None:
        symbol_upper = symbol.upper()
        if symbol_upper in self._coin_id_cache:
            return self._coin_id_cache[symbol_upper]

        candidate_symbols = [symbol_upper]
        canonical_symbol = CRYPTO_SYMBOL_ALIASES.get(symbol_upper)
        if canonical_symbol is not None and canonical_symbol not in candidate_symbols:
            candidate_symbols.append(canonical_symbol)

        async with aiosqlite.connect(self._db_path) as connection:
            row = None
            for candidate_symbol in candidate_symbols:
                cursor = await connection.execute(
                    """
                    SELECT coingecko_id
                    FROM watchlist
                    WHERE symbol = ? AND asset_type = 'crypto' AND is_active = 1 AND coingecko_id IS NOT NULL
                    LIMIT 1
                    """,
                    (candidate_symbol,),
                )
                row = await cursor.fetchone()
                if row is not None:
                    break
            if row is None:
                for candidate_symbol in candidate_symbols:
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
                        (candidate_symbol,),
                    )
                    row = await cursor.fetchone()
                    if row is not None:
                        break

        coin_id = None if row is None or row[0] is None else str(row[0])
        self._coin_id_cache[symbol_upper] = coin_id
        return coin_id


async def _fetch_finnhub_stock_close(symbol: str, target_at: datetime, api_key: str) -> float | None:
    try:
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
    except Exception:
        logger.warning(
            "Historical Finnhub lookup failed for %s at %s",
            symbol,
            target_at.date().isoformat(),
            exc_info=True,
        )
        return None

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

    try:
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
    except Exception:
        logger.warning(
            "Historical yfinance lookup failed for %s at %s",
            symbol,
            target_at.date().isoformat(),
            exc_info=True,
        )
        return None


async def _fetch_coingecko_crypto_close(coin_id: str, target_at: datetime) -> float | None:
    date_param = target_at.astimezone(UTC).strftime("%d-%m-%Y")
    max_retries = 3

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{COINGECKO_BASE}/coins/{coin_id}/history",
                    params={
                        "date": date_param,
                        "localization": "false",
                    },
                )
        except Exception:
            logger.warning(
                "Historical CoinGecko lookup failed for %s at %s",
                coin_id,
                target_at.date().isoformat(),
                exc_info=True,
            )
            return None

        if response.status_code == 429:
            if attempt == max_retries - 1:
                logger.warning(
                    "Historical CoinGecko lookup rate-limited for %s at %s after %s attempts: %s",
                    coin_id,
                    target_at.date().isoformat(),
                    max_retries,
                    response.text[:500],
                )
                return None
            await asyncio.sleep(_coingecko_retry_delay_seconds(response, attempt))
            continue

        if response.status_code >= 400:
            logger.warning(
                "Historical CoinGecko lookup failed for %s at %s with status %s: %s",
                coin_id,
                target_at.date().isoformat(),
                response.status_code,
                response.text[:500],
            )
            return None

        try:
            payload = response.json()
        except Exception:
            logger.warning(
                "Historical CoinGecko lookup returned invalid JSON for %s at %s: %s",
                coin_id,
                target_at.date().isoformat(),
                response.text[:500],
                exc_info=True,
            )
            return None

        market_data = payload.get("market_data")
        if not isinstance(market_data, dict):
            return None
        current_price = market_data.get("current_price")
        if not isinstance(current_price, dict):
            return None
        usd_price = current_price.get("usd")
        try:
            return None if usd_price is None else float(usd_price)
        except (TypeError, ValueError):
            return None

    return None


def _coingecko_retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(8.0, 1.5 * (2**attempt))


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
