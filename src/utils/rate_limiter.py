from __future__ import annotations

import logging
from typing import Any

import httpx
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)


class RateLimitedClient:
    """Wraps httpx.AsyncClient with per-API rate limiting."""

    def __init__(self, client: httpx.AsyncClient, limiter: AsyncLimiter, api_name: str) -> None:
        self.client = client
        self.limiter = limiter
        self.api_name = api_name

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        async with self.limiter:
            logger.debug("[%s] GET %s", self.api_name, url)
            response = await self.client.get(url, **kwargs)
            response.raise_for_status()
            return response
