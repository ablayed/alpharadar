from __future__ import annotations

import time

import pytest

from src.data import dexscreener
from src.data.dexscreener import DexToken, apply_dex_safety_filters, parse_pair_to_dex_token


def _sample_pair(*, symbol: str = "GRIFF", pair_created_at: int | None = None) -> dict[str, object]:
    return {
        "chainId": "solana",
        "pairAddress": "pair-1",
        "dexId": "raydium",
        "priceUsd": "0.1234",
        "pairCreatedAt": pair_created_at if pair_created_at is not None else int((time.time() - 10 * 24 * 3600) * 1000),
        "fdv": 12_000_000,
        "marketCap": 8_000_000,
        "url": "https://dexscreener.com/solana/pair-1",
        "info": {"imageUrl": "https://example.com/token.png"},
        "baseToken": {
            "symbol": symbol,
            "name": "Griffain",
            "address": "token-1",
        },
        "volume": {
            "h24": 720_000,
            "h6": 220_000,
            "h1": 48_000,
        },
        "liquidity": {"usd": 180_000},
        "txns": {
            "h24": {
                "buys": 240,
                "sells": 120,
            }
        },
        "priceChange": {
            "h24": 14.2,
            "h6": 8.4,
            "h1": 2.1,
        },
    }


def _token(*, boost_count: int = 300) -> DexToken:
    token = parse_pair_to_dex_token(_sample_pair(), boost_count=boost_count)
    assert token is not None
    return token


def test_parse_pair_to_dex_token_computes_ratios() -> None:
    token = parse_pair_to_dex_token(_sample_pair(), boost_count=250)

    assert token is not None
    assert token.symbol == "GRIFF"
    assert token.dex_id == "raydium"
    assert token.volume_24h == 720_000
    assert token.liquidity_usd == 180_000
    assert token.vl_ratio == 4.0
    assert token.buy_sell_ratio == 2.0
    assert token.boost_count == 250


def test_apply_dex_safety_filters_rejects_new_pairs() -> None:
    young_pair = _sample_pair(pair_created_at=int((time.time() - 12 * 3600) * 1000))
    token = parse_pair_to_dex_token(young_pair, boost_count=150)
    assert token is not None

    is_safe, reason = apply_dex_safety_filters(token)

    assert is_safe is False
    assert "Too new" in reason


def test_score_penalizes_excessive_boosts() -> None:
    moderate = dexscreener.dex_token_to_candidate(_token(boost_count=300))
    excessive = dexscreener.dex_token_to_candidate(_token(boost_count=4_500))

    assert moderate["signal_score"] > excessive["signal_score"]


@pytest.mark.asyncio
async def test_discover_dex_candidates_skips_binance_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch_top_boosted() -> list[dict[str, object]]:
        return [
            {"chainId": "solana", "tokenAddress": "token-1", "amount": 250},
            {"chainId": "solana", "tokenAddress": "token-2", "amount": 250},
        ]

    async def fake_fetch_trending_metas() -> list[dict[str, object]]:
        return [{"name": "AI"}]

    async def fake_fetch_token_pairs(chain_id: str, token_address: str) -> list[dict[str, object]]:
        del chain_id
        if token_address == "token-1":
            return [_sample_pair(symbol="GRIFF")]
        return [_sample_pair(symbol="SOL")]

    monkeypatch.setattr(dexscreener, "fetch_top_boosted", fake_fetch_top_boosted)
    monkeypatch.setattr(dexscreener, "fetch_trending_metas", fake_fetch_trending_metas)
    monkeypatch.setattr(dexscreener, "fetch_token_pairs", fake_fetch_token_pairs)

    candidates = await dexscreener.discover_dex_candidates(
        binance_symbols={"SOL"},
        max_candidates=5,
    )

    assert [coin["symbol"] for coin in candidates] == ["GRIFF"]
    assert candidates[0]["source"] == "dexscreener"
    assert candidates[0]["chain_id"] == "solana"
    assert candidates[0]["dex_id"] == "raydium"
    assert candidates[0]["liquidity_usd"] == 180_000
