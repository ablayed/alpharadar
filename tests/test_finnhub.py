from __future__ import annotations

import httpx
import pytest

from src.data.finnhub_client import FinnhubClient, _is_relevant_news


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


@pytest.mark.asyncio
async def test_get_company_news_filters_to_latest_relevant_headline(db) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/company-news"):
            return httpx.Response(
                200,
                json=[
                    {
                        "category": "company",
                        "datetime": 1_700_000_300,
                        "headline": "Rosenblatt is Bullish on Lumentum Holdings",
                        "id": 1,
                        "image": "",
                        "related": "NVDA,LITE",
                        "source": "Test",
                        "summary": "Analyst note focused on Lumentum.",
                        "url": "https://example.com/lite",
                    },
                    {
                        "category": "company",
                        "datetime": 1_700_000_200,
                        "headline": "Jensen Huang says NVIDIA demand remains strong",
                        "id": 2,
                        "image": "",
                        "related": "NVDA",
                        "source": "Test",
                        "summary": "NVIDIA datacenter orders continue to grow.",
                        "url": "https://example.com/nvda",
                    },
                    {
                        "category": "company",
                        "datetime": 1_700_000_100,
                        "headline": "NVIDIA launches new enterprise AI tools",
                        "id": 3,
                        "image": "",
                        "related": "NVDA",
                        "source": "Test",
                        "summary": "Older but still relevant.",
                        "url": "https://example.com/nvda-older",
                    },
                ],
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = FinnhubClient(api_key="test", db=db, transport=httpx.MockTransport(handler))
    await client.initialize()
    news = await client.get_company_news("NVDA", company_name="NVIDIA")
    await client.close()

    assert len(news) == 1
    assert news[0].headline == "Jensen Huang says NVIDIA demand remains strong"
    assert news[0].related_symbols[0] == "NVDA"


def test_is_relevant_news_requires_symbol_or_company_match() -> None:
    assert _is_relevant_news(
        "NVIDIA launches new Blackwell systems",
        "",
        "NVDA",
        "NVIDIA",
    )
    assert _is_relevant_news(
        "Analysts cite Jensen Huang as AI demand stays robust",
        "",
        "NVDA",
        "NVIDIA",
    )
    assert not _is_relevant_news(
        "Rosenblatt is Bullish on Lumentum Holdings",
        "Analyst note focused on Lumentum.",
        "NVDA",
        "NVIDIA",
    )


def test_is_relevant_news_rejects_headline_led_by_other_company() -> None:
    assert not _is_relevant_news(
        "NVIDIA partners with ByteDance on new AI systems",
        "Google Cloud was briefly mentioned as an industry comparison.",
        "GOOGL",
        "Google",
    )
