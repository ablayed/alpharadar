from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

from src.data.base import BaseDataProvider, DeFiProtocol
from src.storage.database import Database
from src.utils.rate_limiter import RateLimitedClient

logger = logging.getLogger(__name__)


class HistoricalTVLEntry(BaseModel):
    date: int
    tvl: float | None = None


class ProtocolEntry(BaseModel):
    id: str | int | None = None
    name: str
    symbol: str | None = None
    category: str | None = None
    chains: list[str] = Field(default_factory=list)
    tvl: float | None = None
    change_1d: float | None = None
    change_7d: float | None = None


class ProtocolTVLResponse(BaseModel):
    tvl: list[HistoricalTVLEntry] | float | None = None
    currentChainTvls: dict[str, float] | None = None


class DeFiLlamaClient(BaseDataProvider):
    BASE_URL = "https://api.llama.fi"
    COINS_BASE_URL = "https://coins.llama.fi"

    def __init__(
        self,
        db: Database,
        rpm: int = 400,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.db = db
        self._client: httpx.AsyncClient | None = None
        self._rate_client: RateLimitedClient | None = None
        self._coins_client: httpx.AsyncClient | None = None
        self._coins_rate_client: RateLimitedClient | None = None
        self._limiter = AsyncLimiter(max_rate=rpm, time_period=60)
        self._transport = transport

    async def initialize(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(timeout=20.0, transport=self._transport)
        self._rate_client = RateLimitedClient(self._client, self._limiter, "defillama")
        self._coins_client = httpx.AsyncClient(timeout=20.0, transport=self._transport)
        self._coins_rate_client = RateLimitedClient(self._coins_client, self._limiter, "defillama-coins")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._rate_client = None
        if self._coins_client is not None:
            await self._coins_client.aclose()
            self._coins_client = None
            self._coins_rate_client = None

    async def health_check(self) -> bool:
        try:
            payload = await self._request_json(
                endpoint="/v2/chains",
                cache_key="defillama:chains",
                ttl=600,
            )
            return isinstance(payload, list)
        except Exception:
            logger.exception("DeFi Llama health check failed")
            return False

    async def get_total_tvl(self) -> tuple[float | None, float | None]:
        payload = await self._request_json(
            endpoint="/v2/historicalChainTvl",
            cache_key="defillama:historical_chain_tvl",
            ttl=600,
        )
        if not isinstance(payload, list):
            return None, None

        entries = [item for item in payload if isinstance(item, dict)]
        valid = [item for item in entries if isinstance(item.get("tvl"), (int, float))]
        if not valid:
            return None, None

        latest_tvl = float(valid[-1]["tvl"])
        if len(valid) < 2:
            return latest_tvl, None

        previous_tvl = float(valid[-2]["tvl"])
        if previous_tvl <= 0:
            return latest_tvl, None

        change_pct = ((latest_tvl - previous_tvl) / previous_tvl) * 100
        return latest_tvl, change_pct

    async def get_top_movers(self, n: int = 5) -> list[DeFiProtocol]:
        payload = await self._request_json(
            endpoint="/protocols",
            cache_key="defillama:protocols",
            ttl=300,
        )
        if not isinstance(payload, list):
            return []

        # Filter malformed records before validation to avoid null TVL crashes.
        raw_valid = [
            item
            for item in payload
            if isinstance(item, dict)
            and isinstance(item.get("tvl"), (int, float))
            and float(item["tvl"]) > 0
            and isinstance(item.get("change_1d"), (int, float))
        ]

        protocols: list[ProtocolEntry] = []
        for item in raw_valid:
            try:
                protocols.append(ProtocolEntry.model_validate(item))
            except Exception:
                continue

        protocols.sort(key=lambda p: abs(p.change_1d or 0.0), reverse=True)
        top = protocols[:n]
        return [
            DeFiProtocol(
                name=item.name,
                tvl=float(item.tvl or 0.0),
                change_1d_pct=float(item.change_1d or 0.0),
                change_7d_pct=float(item.change_7d or 0.0),
                category=item.category or "Unknown",
                chain=item.chains[0] if item.chains else None,
            )
            for item in top
        ]

    async def get_protocol_tvl(self, protocol_slug: str) -> float:
        payload = await self._request_json(
            endpoint=f"/protocol/{protocol_slug}",
            cache_key=f"defillama:protocol:{protocol_slug}",
            ttl=300,
        )
        if not isinstance(payload, dict):
            return 0.0
        parsed = ProtocolTVLResponse.model_validate(payload)
        if isinstance(parsed.tvl, (int, float)):
            return float(parsed.tvl)
        if isinstance(parsed.tvl, list) and parsed.tvl:
            latest = parsed.tvl[-1]
            if isinstance(latest, HistoricalTVLEntry):
                return float(latest.tvl)
        return 0.0

    async def get_coingecko_prices(self, coin_ids: list[str]) -> dict[str, float]:
        if not coin_ids:
            return {}
        if self._coins_rate_client is None:
            raise RuntimeError("DeFiLlamaClient is not initialized")
        keys = [f"coingecko:{coin_id}" for coin_id in sorted(set(coin_ids))]
        endpoint = "/prices/current/" + ",".join(keys)
        cache_key = f"defillama:coins:{','.join(keys)}"
        cached = await self.db.get_cached(cache_key)
        if cached is not None:
            payload = json.loads(cached)
        else:
            response = await self._coins_rate_client.get(f"{self.COINS_BASE_URL}{endpoint}")
            payload = response.json()
            await self.db.set_cached(cache_key, json.dumps(payload), ttl_seconds=120)
        if not isinstance(payload, dict):
            return {}
        coins = payload.get("coins")
        if not isinstance(coins, dict):
            return {}
        result: dict[str, float] = {}
        for full_key, value in coins.items():
            if not isinstance(full_key, str) or not isinstance(value, dict):
                continue
            if not full_key.startswith("coingecko:"):
                continue
            coin_id = full_key.split("coingecko:", 1)[1]
            price = value.get("price")
            if isinstance(price, (int, float)):
                result[coin_id] = float(price)
        return result

    async def _request_json(self, endpoint: str, cache_key: str, ttl: int) -> Any:
        cached = await self.db.get_cached(cache_key)
        if cached is not None:
            return json.loads(cached)

        if self._rate_client is None:
            raise RuntimeError("DeFiLlamaClient is not initialized")

        response = await self._rate_client.get(f"{self.BASE_URL}{endpoint}")
        payload: Any = response.json()
        await self.db.set_cached(cache_key, json.dumps(payload), ttl_seconds=ttl)
        return payload
