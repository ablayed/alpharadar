from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

from src.data.base import (
    AssetPrice,
    AssetType,
    BaseDataProvider,
    DataFreshness,
    EarningsEvent,
    NewsItem,
)
from src.storage.database import Database
from src.utils.rate_limiter import RateLimitedClient

logger = logging.getLogger(__name__)


class FinnhubQuoteResponse(BaseModel):
    c: float = 0.0
    d: float = 0.0
    dp: float = 0.0
    h: float = 0.0
    l: float = 0.0
    o: float = 0.0
    pc: float = 0.0
    t: int = 0


class FinnhubNewsResponse(BaseModel):
    category: str = ""
    datetime: int = 0
    headline: str = ""
    id: int = 0
    image: str = ""
    related: str = ""
    source: str = ""
    summary: str = ""
    url: str = ""


class FinnhubMarketStatusResponse(BaseModel):
    exchange: str
    holiday: str | None = None
    isOpen: bool = Field(default=False)
    session: str | None = None
    timezone: str | None = None
    t: int = 0


class FinnhubEarningsEntry(BaseModel):
    date: str
    epsEstimate: float | None = None
    hour: str = "dmh"
    revenueEstimate: float | None = None
    symbol: str


class FinnhubEarningsResponse(BaseModel):
    earningsCalendar: list[FinnhubEarningsEntry] = Field(default_factory=list)


class FinnhubClient(BaseDataProvider):
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str,
        db: Database,
        rpm: int = 45,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.db = db
        self._client: httpx.AsyncClient | None = None
        self._rate_client: RateLimitedClient | None = None
        self._limiter = AsyncLimiter(max_rate=rpm, time_period=60)
        self._transport = transport

    async def initialize(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            timeout=15.0,
            transport=self._transport,
            headers={"X-Finnhub-Token": self.api_key},
        )
        self._rate_client = RateLimitedClient(self._client, self._limiter, "finnhub")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._rate_client = None

    async def health_check(self) -> bool:
        try:
            status = await self.get_market_status()
            return "is_open" in status
        except Exception:
            logger.exception("Finnhub health check failed")
            return False

    async def get_quote(self, symbol: str) -> AssetPrice:
        market_status = await self.get_market_status()
        ttl = 60 if market_status["is_open"] else 3_600
        cache_key = f"finnhub:quote:{symbol.upper()}"
        payload = await self._request_json(
            endpoint="/quote",
            params={"symbol": symbol.upper()},
            cache_key=cache_key,
            ttl_seconds=ttl,
        )
        quote = FinnhubQuoteResponse.model_validate(payload)
        return AssetPrice(
            symbol=symbol.upper(),
            asset_type=AssetType.STOCK,
            current_price=float(quote.c),
            change_24h_pct=float(quote.dp),
            change_7d_pct=None,
            high_24h=(float(quote.h) if quote.h != 0 else None),
            low_24h=(float(quote.l) if quote.l != 0 else None),
            market_cap=None,
            volume_24h=None,
            previous_close=(float(quote.pc) if quote.pc != 0 else None),
            freshness=DataFreshness.REALTIME if market_status["is_open"] else DataFreshness.DELAYED_15M,
            source="finnhub",
        )

    async def get_quotes_batch(self, symbols: list[str]) -> list[AssetPrice]:
        quotes: list[AssetPrice] = []
        for idx, symbol in enumerate(symbols):
            quotes.append(await self.get_quote(symbol))
            if idx < len(symbols) - 1:
                await asyncio.sleep(0.1)
        return quotes

    async def get_company_news(self, symbol: str, days_back: int = 1) -> list[NewsItem]:
        today = date.today()
        from_date = today - timedelta(days=days_back)
        cache_key = f"finnhub:news:{symbol.upper()}:{from_date.isoformat()}:{today.isoformat()}"
        payload = await self._request_json(
            endpoint="/company-news",
            params={
                "symbol": symbol.upper(),
                "from": from_date.isoformat(),
                "to": today.isoformat(),
            },
            cache_key=cache_key,
            ttl_seconds=300,
        )
        if not isinstance(payload, list):
            return []
        parsed = [FinnhubNewsResponse.model_validate(item) for item in payload]
        parsed.sort(key=lambda item: item.datetime, reverse=True)
        news_items: list[NewsItem] = []
        for item in parsed[:3]:
            related_symbols = [
                related.strip().upper()
                for related in item.related.split(",")
                if related.strip()
            ]
            published = datetime.fromtimestamp(item.datetime, tz=UTC).isoformat() if item.datetime else ""
            news_items.append(
                NewsItem(
                    headline=item.headline,
                    summary=item.summary,
                    source=item.source,
                    url=item.url,
                    published_at=published,
                    related_symbols=related_symbols,
                )
            )
        return news_items

    async def get_market_status(self) -> dict[str, str | bool | None]:
        cache_key = "finnhub:market-status:US"
        payload = await self._request_json(
            endpoint="/stock/market-status",
            params={"exchange": "US"},
            cache_key=cache_key,
            ttl_seconds=300,
        )
        status = FinnhubMarketStatusResponse.model_validate(payload)
        return {
            "is_open": status.isOpen,
            "session": status.session,
        }

    async def get_earnings_calendar(self, date_value: str) -> list[EarningsEvent]:
        cache_key = f"finnhub:earnings:{date_value}"
        payload = await self._request_json(
            endpoint="/calendar/earnings",
            params={"from": date_value, "to": date_value},
            cache_key=cache_key,
            ttl_seconds=3_600,
        )
        calendar = FinnhubEarningsResponse.model_validate(payload)
        return [
            EarningsEvent(
                symbol=item.symbol.upper(),
                date=item.date,
                hour=item.hour,
                eps_estimate=item.epsEstimate,
                revenue_estimate=item.revenueEstimate,
            )
            for item in calendar.earningsCalendar
        ]

    async def _request_json(
        self,
        endpoint: str,
        params: dict[str, Any],
        cache_key: str,
        ttl_seconds: int,
    ) -> Any:
        cached = await self.db.get_cached(cache_key)
        if cached is not None:
            return json.loads(cached)

        if self._rate_client is None:
            raise RuntimeError("FinnhubClient is not initialized")

        url = f"{self.BASE_URL}{endpoint}"
        response = await self._rate_client.get(url, params=params)
        payload: Any = response.json()
        await self.db.set_cached(cache_key, json.dumps(payload), ttl_seconds=ttl_seconds)
        return payload

