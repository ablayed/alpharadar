from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import aiosqlite
import pytest

from src.sentiment.analyzer import SentimentResult
from src.sentiment.cache import SentimentCache
from src.storage.database import Database


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


@pytest.mark.asyncio
async def test_initialize_adds_multi_timeframe_prediction_columns_to_legacy_db() -> None:
    test_dir = Path(".test_tmp")
    test_dir.mkdir(exist_ok=True)
    db_path = test_dir / f"{uuid4().hex}.db"

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                event_type TEXT,
                predicted_direction TEXT NOT NULL,
                predicted_target REAL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                source TEXT NOT NULL,
                price_at_prediction REAL NOT NULL,
                outcome_recorded INTEGER NOT NULL DEFAULT 0,
                actual_price_1d REAL,
                actual_price_5d REAL,
                actual_direction TEXT,
                actual_change_1d_pct REAL,
                actual_change_5d_pct REAL,
                is_correct_1d INTEGER,
                is_correct_5d INTEGER,
                outcome_recorded_at TEXT
            )
            """
        )
        await conn.commit()

    database = Database(str(db_path))
    await database.initialize()

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute("PRAGMA table_info(predictions)")
        rows = await cursor.fetchall()

    columns = {str(row[1]) for row in rows}
    assert {
        "actual_price_3d",
        "actual_change_3d_pct",
        "is_correct_3d",
        "actual_price_7d",
        "actual_change_7d_pct",
        "is_correct_7d",
        "metadata",
    }.issubset(columns)
    db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_initialize_adds_dex_columns_to_legacy_hot_list_table() -> None:
    test_dir = Path(".test_tmp")
    test_dir.mkdir(exist_ok=True)
    db_path = test_dir / f"{uuid4().hex}.db"

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            CREATE TABLE hot_list (
                symbol TEXT PRIMARY KEY,
                coingecko_id TEXT,
                name TEXT,
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT NOT NULL,
                trigger_reason TEXT,
                initial_price REAL,
                initial_volume REAL,
                market_cap REAL,
                signal_score INTEGER NOT NULL DEFAULT 50,
                price_change_24h REAL,
                status TEXT NOT NULL DEFAULT 'active',
                binance_symbol TEXT
            )
            """
        )
        await conn.commit()

    database = Database(str(db_path))
    await database.initialize()

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute("PRAGMA table_info(hot_list)")
        rows = await cursor.fetchall()

    columns = {str(row[1]) for row in rows}
    assert {
        "source",
        "direction",
        "breakout_description",
        "signals_firing",
        "key_level",
        "chain_id",
        "dex_id",
        "token_address",
        "dex_url",
        "liquidity_usd",
    }.issubset(columns)
    db_path.unlink(missing_ok=True)
