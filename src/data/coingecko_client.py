from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

from src.data.base import AssetPrice, AssetType, BaseDataProvider, DataFreshness
from src.storage.database import Database
from src.utils.rate_limiter import RateLimitedClient

logger = logging.getLogger(__name__)


class CoinMarketsResponse(BaseModel):
    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float | None = None
    total_volume: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    price_change_percentage_24h_in_currency: float | None = None
    price_change_percentage_7d_in_currency: float | None = None
    price_change_percentage_24h: float | None = None


class TrendingCoinData(BaseModel):
    price: float | None = None
    price_change_percentage_24h: dict[str, float] | None = None
    market_cap: str | None = None
    total_volume: str | None = None


class TrendingCoinItem(BaseModel):
    id: str
    name: str
    symbol: str
    market_cap_rank: int | None = None
    score: int | None = None
    data: TrendingCoinData | None = None


class TrendingCoinWrapper(BaseModel):
    item: TrendingCoinItem


class TrendingResponse(BaseModel):
    coins: list[TrendingCoinWrapper] = Field(default_factory=list)


class CoinGeckoClient(BaseDataProvider):
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        api_key: str,
        db: Database,
        rpm: int = 25,
        monthly_limit: int = 10_000,
        warning_threshold: float = 0.8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.db = db
        self.monthly_limit = monthly_limit
        self.warning_threshold = warning_threshold
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
            headers={"x-cg-demo-api-key": self.api_key},
        )
        self._rate_client = RateLimitedClient(self._client, self._limiter, "coingecko")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._rate_client = None

    async def health_check(self) -> bool:
        try:
            payload = await self._request_json(endpoint="/ping", params={}, cache_key="coingecko:ping", ttl=300)
            return isinstance(payload, dict)
        except Exception:
            logger.exception("CoinGecko health check failed")
            return False

    async def get_market_data(self, coin_ids: list[str]) -> list[AssetPrice]:
        if not coin_ids:
            return []
        unique_ids = sorted(set(coin_ids))
        cache_key = f"coingecko:coins:markets:{','.join(unique_ids)}"
        payload = await self._request_json(
            endpoint="/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": ",".join(unique_ids),
                "price_change_percentage": "24h,7d",
                "per_page": 250,
                "page": 1,
                "sparkline": "false",
            },
            cache_key=cache_key,
            ttl=120,
        )
        if payload is None or not isinstance(payload, list):
            return []
        parsed = [CoinMarketsResponse.model_validate(item) for item in payload]
        return [
            AssetPrice(
                symbol=coin.symbol.upper(),
                asset_type=AssetType.CRYPTO,
                current_price=float(coin.current_price),
                change_24h_pct=float(
                    coin.price_change_percentage_24h_in_currency
                    if coin.price_change_percentage_24h_in_currency is not None
                    else (coin.price_change_percentage_24h or 0.0)
                ),
                change_7d_pct=(
                    float(coin.price_change_percentage_7d_in_currency)
                    if coin.price_change_percentage_7d_in_currency is not None
                    else None
                ),
                high_24h=coin.high_24h,
                low_24h=coin.low_24h,
                market_cap=coin.market_cap,
                volume_24h=coin.total_volume,
                previous_close=None,
                freshness=DataFreshness.REALTIME,
                source="coingecko",
            )
            for coin in parsed
        ]

    async def get_trending(self) -> list[dict[str, object]]:
        payload = await self._request_json(
            endpoint="/search/trending",
            params={},
            cache_key="coingecko:search:trending",
            ttl=600,
        )
        if payload is None:
            return []
        parsed = TrendingResponse.model_validate(payload)
        result: list[dict[str, object]] = []
        for wrapper in parsed.coins[:5]:
            item = wrapper.item
            result.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "symbol": item.symbol.upper(),
                    "market_cap_rank": item.market_cap_rank,
                }
            )
        return result

    async def get_simple_price(self, coin_ids: list[str]) -> dict[str, dict[str, float]]:
        if not coin_ids:
            return {}
        unique_ids = sorted(set(coin_ids))
        cache_key = f"coingecko:simple:price:{','.join(unique_ids)}"
        payload = await self._request_json(
            endpoint="/simple/price",
            params={
                "ids": ",".join(unique_ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_last_updated_at": "true",
            },
            cache_key=cache_key,
            ttl=60,
        )
        if payload is None or not isinstance(payload, dict):
            return {}
        return {
            str(key): {
                str(metric): float(value)
                for metric, value in data.items()
                if isinstance(value, (int, float))
            }
            for key, data in payload.items()
            if isinstance(data, dict)
        }

    async def coin_id_exists(self, coin_id: str) -> bool:
        prices = await self.get_simple_price([coin_id])
        return coin_id in prices and "usd" in prices[coin_id]

    async def get_remaining_budget(self) -> tuple[int, int]:
        used = await self.db.get_monthly_api_usage("coingecko")
        return used, self.monthly_limit

    async def _request_json(
        self,
        endpoint: str,
        params: dict[str, Any],
        cache_key: str,
        ttl: int,
    ) -> Any:
        cached = await self.db.get_cached(cache_key)
        if cached is not None:
            return json.loads(cached)

        budget_ok = await self._check_budget(endpoint=endpoint)
        if not budget_ok:
            return None

        if self._rate_client is None:
            raise RuntimeError("CoinGeckoClient is not initialized")

        url = f"{self.BASE_URL}{endpoint}"
        made_call = False
        try:
            response = await self._rate_client.get(url, params=params)
            made_call = True
            payload: Any = response.json()
            await self.db.set_cached(cache_key, json.dumps(payload), ttl_seconds=ttl)
            return payload
        finally:
            if made_call:
                await self.db.increment_api_usage("coingecko", endpoint)

    async def _check_budget(self, endpoint: str) -> bool:
        used = await self.db.get_monthly_api_usage("coingecko")
        warning_cutoff = int(self.monthly_limit * self.warning_threshold)
        hard_cutoff = int(self.monthly_limit * 0.95)
        if used >= hard_cutoff:
            logger.error(
                "CoinGecko monthly budget exhausted (%s/%s), skipping endpoint %s",
                used,
                self.monthly_limit,
                endpoint,
            )
            return False
        if used >= warning_cutoff:
            logger.warning(
                "CoinGecko usage warning (%s/%s) for endpoint %s",
                used,
                self.monthly_limit,
                endpoint,
            )
        return True

