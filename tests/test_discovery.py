from __future__ import annotations

import aiosqlite
import pytest

from src.data.discovery import apply_safety_filters, score_candidate
from src.data.hot_list import (
    MAX_SIZE,
    add_to_hot_list,
    expire_old_entries,
    get_active_hot_list,
    get_latest_discovery_run,
    get_merged_watchlist,
    init_hot_list_table,
    record_discovery_run,
)


def test_apply_safety_filters_rejects_unsafe_candidates() -> None:
    candidates = [
        {
            "id": "safe-coin",
            "symbol": "SAFE",
            "market_cap": 25_000_000,
            "volume_24h": 2_000_000,
            "age_days": 365,
            "categories": ["layer-1"],
        },
        {
            "id": "tether",
            "symbol": "USDT",
            "market_cap": 100_000_000,
            "volume_24h": 5_000_000,
            "age_days": 365,
            "categories": ["stablecoin"],
        },
        {
            "id": "tiny-cap",
            "symbol": "MICRO",
            "market_cap": 5_000_000,
            "volume_24h": 5_000_000,
            "age_days": 365,
            "categories": ["gaming"],
        },
        {
            "id": "young-coin",
            "symbol": "YOUNG",
            "market_cap": 50_000_000,
            "volume_24h": 5_000_000,
            "age_days": 30,
            "categories": ["defi"],
        },
        {
            "id": "wrapped-btc",
            "symbol": "WBTCX",
            "market_cap": 50_000_000,
            "volume_24h": 5_000_000,
            "age_days": 365,
            "categories": ["wrapped asset"],
        },
    ]

    filtered = apply_safety_filters(candidates, {"BTC", "ETH"})

    assert [coin["symbol"] for coin in filtered] == ["SAFE"]


def test_score_candidate_rewards_multi_source_confirmation() -> None:
    single_source = {
        "price_change_24h": 22.0,
        "volume_24h": 12_000_000,
        "market_cap_rank": 120,
        "source": "top_gainer",
        "sources": ["top_gainer"],
    }
    multi_source = {
        "price_change_24h": 22.0,
        "volume_24h": 12_000_000,
        "market_cap_rank": 120,
        "source": "trending_coingecko",
        "sources": ["trending_coingecko", "binance_volume"],
    }

    assert score_candidate(multi_source) > score_candidate(single_source)


@pytest.mark.asyncio
async def test_hot_list_add_expire_merge_and_discovery_run(db) -> None:
    await init_hot_list_table(db.db_path)

    added = await add_to_hot_list(
        db.db_path,
        symbol="WIF",
        coingecko_id="dogwifcoin",
        name="dogwifhat",
        price=2.5,
        volume=8_000_000,
        market_cap=200_000_000,
        trigger_reason="trending_coingecko",
        signal_score=72,
        price_change_24h=12.4,
    )
    assert added is True

    active = await get_active_hot_list(db.db_path)
    assert [coin["symbol"] for coin in active] == ["WIF"]

    merged = await get_merged_watchlist(db.db_path, ["BTC", "ETH"])
    assert merged == ["BTC", "ETH", "WIF"]

    await record_discovery_run(
        db.db_path,
        candidates_scanned=18,
        candidates_safe=6,
        added_count=1,
        expired_count=0,
    )
    latest_run = await get_latest_discovery_run(db.db_path)
    assert latest_run is not None
    assert latest_run["added_count"] == 1

    async with aiosqlite.connect(db.db_path) as connection:
        await connection.execute(
            "UPDATE hot_list SET expires_at = datetime('now', '-1 day') WHERE symbol = 'WIF'"
        )
        await connection.commit()

    expired = await expire_old_entries(db.db_path)
    assert expired == 1
    assert await get_active_hot_list(db.db_path) == []


@pytest.mark.asyncio
async def test_hot_list_evicts_lowest_score_when_full(db) -> None:
    await init_hot_list_table(db.db_path)

    for index in range(MAX_SIZE):
        added = await add_to_hot_list(
            db.db_path,
            symbol=f"C{index}",
            coingecko_id=f"coin-{index}",
            name=f"Coin {index}",
            price=1.0 + index,
            volume=2_000_000 + index,
            market_cap=20_000_000 + index,
            trigger_reason="top_gainer",
            signal_score=40 + index,
            price_change_24h=5.0,
        )
        assert added is True

    rejected = await add_to_hot_list(
        db.db_path,
        symbol="LOW",
        coingecko_id="low-coin",
        name="Low Coin",
        price=1.0,
        volume=2_000_000,
        market_cap=20_000_000,
        trigger_reason="top_gainer",
        signal_score=39,
        price_change_24h=4.0,
    )
    assert rejected is False

    added = await add_to_hot_list(
        db.db_path,
        symbol="HIGH",
        coingecko_id="high-coin",
        name="High Coin",
        price=3.0,
        volume=9_000_000,
        market_cap=50_000_000,
        trigger_reason="trending_coingecko",
        signal_score=95,
        price_change_24h=18.0,
    )
    assert added is True

    active_symbols = {coin["symbol"] for coin in await get_active_hot_list(db.db_path)}
    assert "HIGH" in active_symbols
    assert "C0" not in active_symbols
    assert len(active_symbols) == MAX_SIZE
