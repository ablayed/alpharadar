"""
DexScreener API client for AlphaRadar.

Discovers DEX-native tokens that are not yet trading on Binance, then applies
AlphaRadar-specific safety and momentum scoring.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.dexscreener.com"
BINANCE_EXCHANGE_INFO_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
MIN_DELAY = 0.25

MIN_LIQUIDITY_USD = 50_000
MIN_VOLUME_24H = 100_000
MIN_PAIR_AGE_HOURS = 168
MAX_VL_RATIO = 50.0
MAX_BUY_SELL_RATIO = 10.0
MIN_SELL_COUNT = 5

CHAIN_TO_GOPLUS = {
    "ethereum": "1",
    "bsc": "56",
    "polygon": "137",
    "arbitrum": "42161",
    "base": "8453",
    "optimism": "10",
    "avalanche": "43114",
    "solana": "solana",
}

_last_request_at: float = 0.0
_request_lock = asyncio.Lock()


@dataclass(slots=True)
class DexToken:
    symbol: str
    name: str
    chain_id: str
    token_address: str
    pair_address: str
    dex_id: str
    price_usd: float
    volume_24h: float
    volume_6h: float
    volume_1h: float
    liquidity_usd: float
    price_change_24h: float
    price_change_6h: float
    price_change_1h: float
    buys_24h: int
    sells_24h: int
    pair_created_at: int
    fdv: float
    market_cap: float
    boost_count: int
    has_profile: bool
    vl_ratio: float
    buy_sell_ratio: float
    url: str


async def _dex_request(path: str, params: dict[str, object] | None = None) -> dict[str, Any] | list[Any] | None:
    global _last_request_at

    async with _request_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_DELAY:
            await asyncio.sleep(MIN_DELAY - elapsed)
        _last_request_at = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(f"{BASE_URL}{path}", params=params)
            if response.status_code == 429:
                logger.warning("DexScreener rate limit hit on %s", path)
                await asyncio.sleep(5)
                return None
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.debug("DexScreener request failed for %s", path, exc_info=True)
        return None

    if isinstance(payload, dict | list):
        return payload
    return None


async def fetch_top_boosted() -> list[dict[str, Any]]:
    data = await _dex_request("/token-boosts/top/v1")
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


async def fetch_trending_metas() -> list[dict[str, Any]]:
    data = await _dex_request("/metas/trending/v1")
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


async def fetch_token_pairs(chain_id: str, token_address: str) -> list[dict[str, Any]]:
    data = await _dex_request(f"/token-pairs/v1/{chain_id}/{token_address}")
    if isinstance(data, dict):
        pairs = data.get("pairs", [])
        if isinstance(pairs, list):
            return [item for item in pairs if isinstance(item, dict)]
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


async def search_token(query: str) -> list[dict[str, Any]]:
    data = await _dex_request("/latest/dex/search", {"q": query})
    if not isinstance(data, dict):
        return []
    pairs = data.get("pairs", [])
    if not isinstance(pairs, list):
        return []
    return [item for item in pairs if isinstance(item, dict)]


def parse_pair_to_dex_token(pair: dict[str, Any], boost_count: int = 0) -> DexToken | None:
    try:
        base = pair.get("baseToken", {})
        volume = pair.get("volume", {})
        liquidity = pair.get("liquidity", {})
        txns = pair.get("txns", {})
        price_change = pair.get("priceChange", {})

        if not isinstance(base, dict):
            base = {}
        if not isinstance(volume, dict):
            volume = {}
        if not isinstance(liquidity, dict):
            liquidity = {}
        if not isinstance(txns, dict):
            txns = {}
        if not isinstance(price_change, dict):
            price_change = {}

        vol_24h = float(volume.get("h24", 0) or 0)
        vol_6h = float(volume.get("h6", 0) or 0)
        vol_1h = float(volume.get("h1", 0) or 0)
        liq_usd = float(liquidity.get("usd", 0) or 0)

        txns_24h = txns.get("h24", {})
        if not isinstance(txns_24h, dict):
            txns_24h = {}
        buys_24h = int(txns_24h.get("buys", 0) or 0)
        sells_24h = int(txns_24h.get("sells", 0) or 0)

        price_usd_raw = pair.get("priceUsd", "0")
        price_usd = float(price_usd_raw or 0)
        vl_ratio = vol_24h / max(liq_usd, 1.0)
        buy_sell_ratio = buys_24h / max(sells_24h, 1)

        return DexToken(
            symbol=str(base.get("symbol", "") or "").upper(),
            name=str(base.get("name", "") or ""),
            chain_id=str(pair.get("chainId", "") or ""),
            token_address=str(base.get("address", "") or ""),
            pair_address=str(pair.get("pairAddress", "") or ""),
            dex_id=str(pair.get("dexId", "") or ""),
            price_usd=price_usd,
            volume_24h=vol_24h,
            volume_6h=vol_6h,
            volume_1h=vol_1h,
            liquidity_usd=liq_usd,
            price_change_24h=float(price_change.get("h24", 0) or 0),
            price_change_6h=float(price_change.get("h6", 0) or 0),
            price_change_1h=float(price_change.get("h1", 0) or 0),
            buys_24h=buys_24h,
            sells_24h=sells_24h,
            pair_created_at=int(pair.get("pairCreatedAt", 0) or 0),
            fdv=float(pair.get("fdv", 0) or 0),
            market_cap=float(pair.get("marketCap", 0) or 0),
            boost_count=boost_count,
            has_profile=bool(pair.get("info")),
            vl_ratio=round(vl_ratio, 2),
            buy_sell_ratio=round(buy_sell_ratio, 2),
            url=str(pair.get("url", "") or ""),
        )
    except (TypeError, ValueError):
        logger.debug("Failed to parse DexScreener pair", exc_info=True)
        return None


def apply_dex_safety_filters(token: DexToken) -> tuple[bool, str]:
    stablecoins = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "PYUSD", "USDE"}

    if token.liquidity_usd < MIN_LIQUIDITY_USD:
        return False, f"Low liquidity: ${token.liquidity_usd:,.0f}"

    if token.volume_24h < MIN_VOLUME_24H:
        return False, f"Low volume: ${token.volume_24h:,.0f}"

    if token.pair_created_at > 0:
        age_hours = ((time.time() * 1000) - token.pair_created_at) / (1000 * 3600)
        if age_hours < MIN_PAIR_AGE_HOURS:
            return False, f"Too new: {age_hours:.0f}h"

    if token.vl_ratio > MAX_VL_RATIO:
        return False, f"Suspicious V/L ratio: {token.vl_ratio:.1f}x"

    if token.sells_24h < MIN_SELL_COUNT:
        return False, f"Possible honeypot: only {token.sells_24h} sells"

    if token.buy_sell_ratio > MAX_BUY_SELL_RATIO:
        return False, f"Extreme buy pressure: {token.buy_sell_ratio:.1f}x"

    if token.symbol in stablecoins:
        return False, "Stablecoin"

    return True, ""


def _score_dex_candidate(token: DexToken) -> int:
    score = 0

    if 2.0 <= token.vl_ratio <= 10.0:
        score += 25
    elif 1.0 <= token.vl_ratio <= 20.0:
        score += 15
    elif token.volume_24h >= 500_000:
        score += 10
    elif token.volume_24h >= 100_000:
        score += 5

    if 1.5 <= token.buy_sell_ratio <= 3.0:
        score += 20
    elif 1.2 <= token.buy_sell_ratio <= 5.0:
        score += 12
    elif 1.0 <= token.buy_sell_ratio <= 1.2:
        score += 5

    if token.price_change_1h > 0 and token.price_change_6h > 0:
        score += 20 if token.price_change_24h < 30 else 8
    elif token.price_change_6h > 5:
        score += 12
    elif token.price_change_24h > 0:
        score += 5

    if token.liquidity_usd >= 500_000:
        score += 15
    elif token.liquidity_usd >= 250_000:
        score += 12
    elif token.liquidity_usd >= 100_000:
        score += 8
    elif token.liquidity_usd >= 50_000:
        score += 4

    if token.has_profile:
        score += 5
    if token.market_cap > 0 or token.fdv > 0:
        score += 5

    if 100 <= token.boost_count <= 800:
        score += 10
    elif 50 <= token.boost_count <= 1500:
        score += 5
    elif token.boost_count > 3000:
        score -= 10

    return max(0, min(score, 100))


def _count_dex_signals(token: DexToken) -> int:
    count = 0
    if token.vl_ratio >= 2.0:
        count += 1
    if token.buy_sell_ratio >= 1.3:
        count += 1
    if token.price_change_1h > 0 and token.price_change_6h > 0:
        count += 1
    if token.liquidity_usd >= 250_000:
        count += 1
    if token.has_profile:
        count += 1
    if 100 <= token.boost_count <= 800:
        count += 1
    return count


def _build_dex_description(token: DexToken) -> str:
    parts: list[str] = []
    if token.vl_ratio >= 2.0:
        parts.append(f"V/L {token.vl_ratio:.1f}x")
    if token.buy_sell_ratio >= 1.3:
        parts.append(
            f"buy pressure {token.buy_sell_ratio:.1f}x"
        )
    if token.price_change_1h > 0 and token.price_change_6h > 0:
        parts.append(
            f"momentum +{token.price_change_1h:.1f}% (1h), +{token.price_change_6h:.1f}% (6h)"
        )
    if token.liquidity_usd >= 250_000:
        parts.append(f"liq ${token.liquidity_usd:,.0f}")
    if not parts:
        parts.append(f"{token.dex_id} on {token.chain_id}")
    return " | ".join(parts[:4])


def dex_token_to_candidate(token: DexToken) -> dict[str, object]:
    score = _score_dex_candidate(token)
    return {
        "id": "",
        "symbol": token.symbol,
        "name": token.name,
        "market_cap_rank": None,
        "market_cap": token.market_cap or token.fdv,
        "volume_24h": token.volume_24h,
        "price_change_24h": token.price_change_24h,
        "price_usd": token.price_usd,
        "source": "dexscreener",
        "sources": ["dexscreener"],
        "signal_score": score,
        "direction": "long_setup" if token.price_change_6h > 0 and token.buy_sell_ratio > 1.2 else "neutral",
        "breakout_description": _build_dex_description(token),
        "signals_firing": _count_dex_signals(token),
        "binance_symbol": None,
        "chain_id": token.chain_id,
        "token_address": token.token_address,
        "dex_url": token.url,
        "liquidity_usd": token.liquidity_usd,
        "dex_id": token.dex_id,
        "price_change_6h": token.price_change_6h,
        "price_change_1h": token.price_change_1h,
        "buys_24h": token.buys_24h,
        "sells_24h": token.sells_24h,
        "buy_sell_ratio": token.buy_sell_ratio,
        "vl_ratio": token.vl_ratio,
        "boost_count": token.boost_count,
        "has_profile": token.has_profile,
    }


async def discover_dex_candidates(
    *,
    binance_symbols: set[str] | None = None,
    max_candidates: int = 15,
) -> list[dict[str, object]]:
    excluded_symbols = {symbol.upper() for symbol in (binance_symbols or set())}
    boosted = await fetch_top_boosted()
    _ = await fetch_trending_metas()

    candidates: list[dict[str, object]] = []
    seen_symbols: set[str] = set()

    for entry in boosted[:40]:
        chain_id = str(entry.get("chainId", "") or "")
        token_address = str(entry.get("tokenAddress", "") or "")
        boost_amount = int(entry.get("amount", 0) or 0)
        if not chain_id or not token_address:
            continue

        pairs = await fetch_token_pairs(chain_id, token_address)
        if not pairs:
            continue

        best_pair = max(
            pairs,
            key=lambda pair: float(
                (pair.get("volume", {}) if isinstance(pair.get("volume"), dict) else {}).get("h24", 0) or 0
            ),
        )
        token = parse_pair_to_dex_token(best_pair, boost_count=boost_amount)
        if token is None or not token.symbol:
            continue
        if token.symbol in excluded_symbols or token.symbol in seen_symbols:
            continue

        is_safe, reason = apply_dex_safety_filters(token)
        if not is_safe:
            logger.debug("DexScreener rejected %s: %s", token.symbol, reason)
            continue

        seen_symbols.add(token.symbol)
        candidates.append(dex_token_to_candidate(token))
        if len(candidates) >= max_candidates:
            break

    candidates.sort(key=lambda coin: int(coin.get("signal_score", 0) or 0), reverse=True)
    logger.info(
        "DexScreener discovery: accepted=%d boosted_scanned=%d",
        len(candidates),
        len(boosted),
    )
    return candidates


async def get_binance_listed_symbols() -> set[str]:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(BINANCE_EXCHANGE_INFO_URL)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.warning("Failed to fetch Binance exchange info", exc_info=True)
        return set()

    symbols: set[str] = set()
    if not isinstance(payload, dict):
        return symbols
    for item in payload.get("symbols", []):
        if not isinstance(item, dict):
            continue
        base_asset = str(item.get("baseAsset", "") or "").upper()
        if base_asset:
            symbols.add(base_asset)
    return symbols


async def goplus_check(chain_id: str, token_address: str) -> tuple[bool, list[str]]:
    goplus_chain = CHAIN_TO_GOPLUS.get(chain_id)
    if not goplus_chain or not token_address:
        return True, []

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(
                f"https://api.gopluslabs.io/api/v1/token_security/{goplus_chain}",
                params={"contract_addresses": token_address},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.debug("GoPlus check failed for %s on %s", token_address, chain_id, exc_info=True)
        return True, []

    if not isinstance(payload, dict):
        return True, []
    result = payload.get("result", {})
    if not isinstance(result, dict):
        return True, []
    token_result = result.get(token_address.lower(), {})
    if not isinstance(token_result, dict):
        return True, []

    flags: list[str] = []
    if token_result.get("is_honeypot") == "1":
        flags.append("HONEYPOT")
    if token_result.get("is_open_source") == "0":
        flags.append("UNVERIFIED_CONTRACT")
    if token_result.get("is_mintable") == "1":
        flags.append("MINTABLE_SUPPLY")
    if token_result.get("owner_change_balance") == "1":
        flags.append("OWNER_CAN_MODIFY_BALANCE")
    if token_result.get("hidden_owner") == "1":
        flags.append("HIDDEN_OWNER")

    sell_tax_value = token_result.get("sell_tax")
    try:
        sell_tax = float(sell_tax_value or 0)
    except (TypeError, ValueError):
        sell_tax = 0.0
    if sell_tax > 0.10:
        flags.append(f"HIGH_SELL_TAX_{sell_tax * 100:.0f}%")

    return len(flags) == 0, flags
