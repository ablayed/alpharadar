from __future__ import annotations

import httpx
import pytest

from src.data.coingecko_client import CoinGeckoClient


@pytest.mark.asyncio
async def test_market_data_parses_7d_change_and_tracks_budget(db) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-cg-demo-api-key") == "test-key"
        if request.url.path.endswith("/coins/markets"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "bitcoin",
                        "symbol": "btc",
                        "name": "Bitcoin",
                        "current_price": 70_187,
                        "market_cap": 1_381_651_251_183,
                        "total_volume": 20_154_184_933,
                        "high_24h": 70_215,
                        "low_24h": 68_060,
                        "price_change_percentage_24h_in_currency": 3.125,
                        "price_change_percentage_7d_in_currency": 8.42,
                    }
                ],
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = CoinGeckoClient(
        api_key="test-key",
        db=db,
        transport=httpx.MockTransport(handler),
        monthly_limit=10_000,
        warning_threshold=0.8,
    )
    await client.initialize()
    prices = await client.get_market_data(["bitcoin"])
    await client.close()

    assert len(prices) == 1
    assert prices[0].symbol == "BTC"
    assert prices[0].change_7d_pct == pytest.approx(8.42)
    usage = await db.get_monthly_api_usage("coingecko")
    assert usage == 1

