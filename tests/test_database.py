from __future__ import annotations

import aiosqlite
import pytest

from src.sentiment.analyzer import SentimentResult
from src.sentiment.cache import SentimentCache


@pytest.mark.asyncio
async def test_watchlist_crud(db) -> None:
    added = await db.add_to_watchlist("TEST", "stock", "Test Corp")
    assert added is True

    duplicate = await db.add_to_watchlist("TEST", "stock", "Test Corp")
    assert duplicate is False

    stocks = await db.get_watchlist("stock")
    assert any(item.symbol == "TEST" for item in stocks)

    removed = await db.remove_from_watchlist("TEST", "stock")
    assert removed is True

    removed_again = await db.remove_from_watchlist("TEST", "stock")
    assert removed_again is False


@pytest.mark.asyncio
async def test_api_usage_tracking(db) -> None:
    await db.increment_api_usage("coingecko", "/coins/markets")
    await db.increment_api_usage("coingecko", "/coins/markets")
    await db.increment_api_usage("coingecko", "/search/trending")
    usage = await db.get_monthly_api_usage("coingecko")
    assert usage == 3


@pytest.mark.asyncio
async def test_cache_expiry_logic(db) -> None:
    await db.set_cached("cache:key", '{"ok": true}', ttl_seconds=1)
    cached = await db.get_cached("cache:key")
    assert cached is not None

    async with aiosqlite.connect(db.db_path) as conn:
        await conn.execute(
            "UPDATE api_cache SET cached_at = datetime('now', '-10 seconds') WHERE cache_key = ?",
            ("cache:key",),
        )
        await conn.commit()

    expired = await db.get_cached("cache:key")
    assert expired is None


@pytest.mark.asyncio
async def test_sentiment_cache_round_trip(db) -> None:
    cache = SentimentCache(db)
    headline = "NVIDIA reports record quarterly revenue of $35 billion"
    result = SentimentResult(
        label="positive",
        score=0.82,
        confidence=0.91,
        positive=0.89,
        negative=0.07,
        neutral=0.04,
    )

    initial = await cache.get_cached([headline])
    assert initial[headline] is None

    await cache.store_results([headline], [result])

    cached = await cache.get_cached([headline])
    assert cached[headline] == result
    assert await db.get_sentiment_cache_count() == 1
