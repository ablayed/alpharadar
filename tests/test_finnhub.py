from __future__ import annotations

import httpx
import pytest

from src.data.finnhub_client import FinnhubClient


@pytest.mark.asyncio
async def test_get_quote_parses_asset_price(db) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stock/market-status"):
            return httpx.Response(
                200,
                json={
                    "exchange": "US",
                    "holiday": None,
                    "isOpen": True,
                    "session": "regular",
                    "timezone": "America/New_York",
                    "t": 1_697_018_041,
                },
            )
        if request.url.path.endswith("/quote"):
            return httpx.Response(
                200,
                json={
                    "c": 142.65,
                    "d": 1.15,
                    "dp": 0.8127,
                    "h": 142.92,
                    "l": 139.11,
                    "o": 141.9,
                    "pc": 141.5,
                    "t": 1_633_118_403,
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = FinnhubClient(api_key="test", db=db, transport=httpx.MockTransport(handler))
    await client.initialize()
    quote = await client.get_quote("AAPL")
    await client.close()

    assert quote.symbol == "AAPL"
    assert quote.current_price == pytest.approx(142.65)
    assert quote.change_24h_pct == pytest.approx(0.8127)
    assert quote.previous_close == pytest.approx(141.5)


@pytest.mark.asyncio
async def test_get_quote_handles_zero_price(db) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stock/market-status"):
            return httpx.Response(
                200,
                json={
                    "exchange": "US",
                    "holiday": None,
                    "isOpen": False,
                    "session": "post-market",
                    "timezone": "America/New_York",
                    "t": 1_697_018_041,
                },
            )
        if request.url.path.endswith("/quote"):
            return httpx.Response(200, json={"c": 0, "d": 0, "dp": 0, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0})
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = FinnhubClient(api_key="test", db=db, transport=httpx.MockTransport(handler))
    await client.initialize()
    quote = await client.get_quote("INVALID")
    await client.close()

    assert quote.current_price == 0
    assert quote.previous_close is None


@pytest.mark.asyncio
async def test_get_company_news_empty_array(db) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/company-news"):
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = FinnhubClient(api_key="test", db=db, transport=httpx.MockTransport(handler))
    await client.initialize()
    news = await client.get_company_news("AAPL")
    await client.close()

    assert news == []

