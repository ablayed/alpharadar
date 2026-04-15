"""
Dynamic coin discovery for AlphaRadar.

Discovers trending and high-volume coins from:
1. CoinGecko /search/trending
2. CoinGecko /coins/markets
3. Binance /api/v3/ticker/24hr

Applies safety filters for:
- minimum market cap
- minimum 24h volume
- minimum age
- stablecoin / wrapped-token exclusion
- fixed-watchlist exclusion
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlencode

import aiosqlite
import httpx

from src.data.breakout_scorer import score_breakout
from src.data.taker_ratio import fetch_taker_buy_ratio

logger = logging.getLogger(__name__)

MIN_MARKET_CAP_USD = 10_000_000
MIN_VOLUME_24H_USD = 500_000
MIN_AGE_DAYS = 90
MAX_HOT_LIST_SIZE = 15
DEFAULT_TTL_DAYS = 5
VOLUME_SPIKE_THRESHOLD = 3.0

STABLECOIN_IDS = {
    "tether",
    "usd-coin",
    "dai",
    "binance-usd",
    "trueusd",
    "paxos-standard",
    "first-digital-usd",
    "usdd",
    "frax",
    "tusd",
    "gusd",
    "usdp",
    "pyusd",
    "ethena-usde",
    "usual-usd",
}

STABLECOIN_SYMBOLS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "PYUSD", "USDE", "USDP", "FRAX"}
EXCLUDED_CATEGORY_KEYWORDS = {"bridged", "wrapped", "stablecoin"}
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
BINANCE_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
COINGECKO_MIN_REQUEST_INTERVAL_SECONDS = 2.5

_coingecko_request_lock = asyncio.Lock()
_coingecko_next_request_at = 0.0


@dataclass(slots=True)
class DiscoveredCoin:
    coingecko_id: str
    symbol: str
    name: str
    market_cap: float
    volume_24h: float
    price_change_24h: float
    price_usd: float
    trigger_reason: str
    signal_score: int
    binance_symbol: str | None = None


def _coingecko_headers(api_key: str) -> dict[str, str]:
    return {"x-cg-demo-api-key": api_key} if api_key else {}


async def fetch_coingecko_trending(api_key: str, db_path: str | None = None) -> list[dict]:
    """Fetch CoinGecko trending coins."""
    payload = await _coingecko_request_json(
        endpoint="/search/trending",
        params={},
        api_key=api_key,
        db_path=db_path,
        ttl_seconds=600,
    )
    if not isinstance(payload, dict):
        return []

    results: list[dict] = []
    for item in payload.get("coins", [])[:15]:
        if not isinstance(item, dict):
            continue
        coin = item.get("item", {})
        if not isinstance(coin, dict):
            continue
        coin_data = coin.get("data", {})
        if not isinstance(coin_data, dict):
            coin_data = {}

        price_change = 0.0
        raw_change = coin_data.get("price_change_percentage_24h")
        if isinstance(raw_change, dict):
            usd_value = raw_change.get("usd")
            if isinstance(usd_value, int | float):
                price_change = float(usd_value)
        elif isinstance(raw_change, int | float):
            price_change = float(raw_change)

        results.append(
            {
                "id": str(coin.get("id", "") or ""),
                "symbol": str(coin.get("symbol", "") or "").upper(),
                "name": str(coin.get("name", "") or ""),
                "market_cap_rank": coin.get("market_cap_rank"),
                "market_cap": _parse_dollar_string(coin_data.get("market_cap")),
                "volume_24h": _parse_dollar_string(coin_data.get("total_volume")),
                "price_change_24h": price_change,
                "price_usd": float(coin_data.get("price", 0) or 0),
                "source": "trending_coingecko",
                "sources": ["trending_coingecko"],
            }
        )
    logger.info("Discovery: fetched %d CoinGecko trending candidates", len(results))
    return results


async def fetch_coingecko_top_gainers(
    api_key: str,
    limit: int = 100,
    db_path: str | None = None,
) -> list[dict]:
    """Fetch top movers from CoinGecko markets data."""
    payload = await _coingecko_request_json(
        endpoint="/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "volume_desc",
            "per_page": min(limit, 250),
            "page": 1,
            "price_change_percentage": "24h,7d",
            "sparkline": "false",
        },
        api_key=api_key,
        db_path=db_path,
        ttl_seconds=900,
    )
    if not isinstance(payload, list):
        return []

    results: list[dict] = []
    for coin in payload:
        if not isinstance(coin, dict):
            continue
        results.append(
            {
                "id": str(coin.get("id", "") or ""),
                "symbol": str(coin.get("symbol", "") or "").upper(),
                "name": str(coin.get("name", "") or ""),
                "market_cap_rank": coin.get("market_cap_rank"),
                "market_cap": float(coin.get("market_cap", 0) or 0),
                "volume_24h": float(coin.get("total_volume", 0) or 0),
                "price_change_24h": float(coin.get("price_change_percentage_24h", 0) or 0),
                "price_usd": float(coin.get("current_price", 0) or 0),
                "source": "top_gainer",
                "sources": ["top_gainer"],
            }
        )

    results.sort(key=lambda item: abs(float(item.get("price_change_24h", 0.0))), reverse=True)
    logger.info("Discovery: fetched %d CoinGecko market movers", len(results))
    return results


async def fetch_binance_volume_spikes() -> list[dict]:
    """Fetch Binance USDT pairs ranked by 24h move and volume."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(BINANCE_TICKER_URL)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.exception("Discovery: Binance volume scan failed")
        return []

    if not isinstance(payload, list):
        return []

    usdt_pairs = [
        ticker
        for ticker in payload
        if isinstance(ticker, dict)
        and str(ticker.get("symbol", "")).endswith("USDT")
        and float(ticker.get("quoteVolume", 0) or 0) >= MIN_VOLUME_24H_USD
    ]
    usdt_pairs.sort(
        key=lambda ticker: abs(float(ticker.get("priceChangePercent", 0) or 0)),
        reverse=True,
    )

    results: list[dict] = []
    for ticker in usdt_pairs[:50]:
        binance_symbol = str(ticker.get("symbol", "") or "")
        symbol = binance_symbol.removesuffix("USDT")
        results.append(
            {
                "id": "",
                "symbol": symbol.upper(),
                "name": symbol.upper(),
                "market_cap_rank": None,
                "market_cap": 0.0,
                "volume_24h": float(ticker.get("quoteVolume", 0) or 0),
                "price_change_24h": float(ticker.get("priceChangePercent", 0) or 0),
                "price_usd": float(ticker.get("weightedAvgPrice", 0) or 0),
                "source": "binance_volume",
                "sources": ["binance_volume"],
                "binance_symbol": binance_symbol,
            }
        )
    logger.info("Discovery: fetched %d Binance volume candidates", len(results))
    return results


async def resolve_coingecko_id(symbol: str, api_key: str, db_path: str | None = None) -> str | None:
    """Resolve a CoinGecko id from a ticker symbol."""
    query = symbol.strip().upper()
    if not query:
        return None

    payload = await _coingecko_request_json(
        endpoint="/search",
        params={"query": query},
        api_key=api_key,
        db_path=db_path,
        ttl_seconds=7 * 24 * 3600,
    )
    if not isinstance(payload, dict):
        return None

    candidates = payload.get("coins", [])
    if not isinstance(candidates, list):
        return None

    exact_matches = [
        coin
        for coin in candidates
        if isinstance(coin, dict) and str(coin.get("symbol", "")).upper() == query
    ]
    ranked = sorted(
        exact_matches or [coin for coin in candidates if isinstance(coin, dict)],
        key=lambda coin: _safe_int(coin.get("market_cap_rank"), default=1_000_000),
    )
    best = ranked[0] if ranked else None
    return str(best.get("id", "") or "") if isinstance(best, dict) else None


async def fetch_coingecko_coin_details(
    coin_id: str,
    api_key: str,
    db_path: str | None = None,
) -> dict | None:
    """Fetch cached CoinGecko coin details for safety filtering."""
    if not coin_id:
        return None

    payload = await _coingecko_request_json(
        endpoint=f"/coins/{coin_id}",
        params={
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false",
        },
        api_key=api_key,
        db_path=db_path,
        ttl_seconds=7 * 24 * 3600,
    )
    if not isinstance(payload, dict):
        return None

    market_data = payload.get("market_data", {})
    if not isinstance(market_data, dict):
        market_data = {}
    current_price = market_data.get("current_price", {})
    market_cap = market_data.get("market_cap", {})
    total_volume = market_data.get("total_volume", {})
    if not isinstance(current_price, dict):
        current_price = {}
    if not isinstance(market_cap, dict):
        market_cap = {}
    if not isinstance(total_volume, dict):
        total_volume = {}

    categories = payload.get("categories", [])
    category_list = [
        str(category).lower()
        for category in categories
        if isinstance(category, str)
    ]
    age_days = _coin_age_days(payload)

    return {
        "id": str(payload.get("id", "") or coin_id),
        "symbol": str(payload.get("symbol", "") or "").upper(),
        "name": str(payload.get("name", "") or coin_id),
        "market_cap_rank": payload.get("market_cap_rank"),
        "market_cap": float(market_cap.get("usd", 0) or 0),
        "volume_24h": float(total_volume.get("usd", 0) or 0),
        "price_change_24h": float(market_data.get("price_change_percentage_24h", 0) or 0),
        "price_usd": float(current_price.get("usd", 0) or 0),
        "categories": category_list,
        "age_days": age_days,
    }


async def enrich_candidates(
    candidates: list[dict],
    api_key: str,
    db_path: str | None = None,
) -> list[dict]:
    """Resolve CoinGecko metadata required for the full safety filter."""
    enriched: list[dict] = []
    for candidate in candidates:
        coin = dict(candidate)
        coin_id = str(coin.get("id", "") or "")
        if not coin_id:
            coin_id = await resolve_coingecko_id(str(coin.get("symbol", "")), api_key, db_path=db_path) or ""
            coin["id"] = coin_id

        details = await fetch_coingecko_coin_details(coin_id, api_key, db_path=db_path) if coin_id else None
        if details is not None:
            coin["id"] = str(details.get("id", "") or coin.get("id", ""))
            coin["name"] = str(details.get("name", "") or coin.get("name", ""))
            coin["symbol"] = str(details.get("symbol", "") or coin.get("symbol", "")).upper()
            coin["market_cap_rank"] = details.get("market_cap_rank") or coin.get("market_cap_rank")
            coin["market_cap"] = float(details.get("market_cap", 0) or coin.get("market_cap", 0) or 0)
            coin["volume_24h"] = float(details.get("volume_24h", 0) or coin.get("volume_24h", 0) or 0)
            existing_change = coin.get("price_change_24h")
            if existing_change in (None, 0, 0.0):
                coin["price_change_24h"] = float(details.get("price_change_24h", 0) or 0)
            else:
                coin["price_change_24h"] = float(existing_change or 0)
            coin["price_usd"] = float(details.get("price_usd", 0) or coin.get("price_usd", 0) or 0)
            coin["categories"] = list(details.get("categories", []))
            coin["age_days"] = details.get("age_days")
        else:
            coin.setdefault("categories", [])
            coin.setdefault("age_days", None)
        enriched.append(coin)
    return enriched


def merge_candidate_sources(*candidate_groups: list[dict]) -> list[dict]:
    """Merge duplicate symbols across discovery sources."""
    merged: dict[str, dict] = {}
    source_priority = {
        "trending_coingecko": 3,
        "binance_volume": 2,
        "top_gainer": 1,
    }

    for group in candidate_groups:
        for candidate in group:
            symbol = str(candidate.get("symbol", "") or "").upper()
            if not symbol:
                continue

            existing = merged.get(symbol)
            if existing is None:
                merged[symbol] = {
                    **candidate,
                    "symbol": symbol,
                    "sources": list(dict.fromkeys(candidate.get("sources", [candidate.get("source", "unknown")]))),
                }
                continue

            existing["id"] = str(existing.get("id", "") or candidate.get("id", "") or "")
            existing["name"] = str(existing.get("name", "") or candidate.get("name", "") or symbol)
            if _safe_int(candidate.get("market_cap_rank"), default=1_000_000) < _safe_int(
                existing.get("market_cap_rank"),
                default=1_000_000,
            ):
                existing["market_cap_rank"] = candidate.get("market_cap_rank")
            for field_name in ("market_cap", "volume_24h", "price_usd"):
                existing[field_name] = max(
                    float(existing.get(field_name, 0) or 0),
                    float(candidate.get(field_name, 0) or 0),
                )
            if abs(float(candidate.get("price_change_24h", 0) or 0)) > abs(float(existing.get("price_change_24h", 0) or 0)):
                existing["price_change_24h"] = float(candidate.get("price_change_24h", 0) or 0)
            if candidate.get("binance_symbol") and not existing.get("binance_symbol"):
                existing["binance_symbol"] = candidate.get("binance_symbol")

            existing_sources = {
                str(source)
                for source in existing.get("sources", [])
                if source
            }
            existing_sources.update(
                str(source)
                for source in candidate.get("sources", [candidate.get("source", "unknown")])
                if source
            )
            ordered_sources = sorted(
                existing_sources,
                key=lambda source: source_priority.get(source, 0),
                reverse=True,
            )
            existing["sources"] = ordered_sources
            existing["source"] = ordered_sources[0] if ordered_sources else "unknown"
    return list(merged.values())


def apply_safety_filters(candidates: list[dict], fixed_watchlist: set[str]) -> list[dict]:
    """Apply anti-scam and quality gates to discovery candidates."""
    filtered: list[dict] = []
    fixed_symbols = {symbol.upper() for symbol in fixed_watchlist}

    for candidate in candidates:
        symbol = str(candidate.get("symbol", "") or "").upper()
        coin_id = str(candidate.get("id", "") or "").lower()
        categories = [str(category).lower() for category in candidate.get("categories", []) if isinstance(category, str)]
        market_cap = float(candidate.get("market_cap", 0) or 0)
        volume = float(candidate.get("volume_24h", 0) or 0)
        age_days = candidate.get("age_days")

        if not symbol or symbol in fixed_symbols:
            continue
        if coin_id in STABLECOIN_IDS or symbol in STABLECOIN_SYMBOLS:
            continue
        if any(keyword in category for category in categories for keyword in EXCLUDED_CATEGORY_KEYWORDS):
            continue
        if volume < MIN_VOLUME_24H_USD:
            continue
        if market_cap < MIN_MARKET_CAP_USD:
            continue
        if not isinstance(age_days, int | float) or float(age_days) < MIN_AGE_DAYS:
            continue
        filtered.append(candidate)
    return filtered


async def enrich_and_score(
    candidates: list[dict],
    coingecko_api_key: str,
    *,
    limit: int = 20,
) -> list[dict]:
    """
    Fetch OHLCV and taker pressure, then run smart breakout scoring.

    Candidates without usable OHLCV fall back to a lighter momentum/liquidity score.
    """
    del coingecko_api_key  # Reserved for future CoinGecko-backed OHLCV fallbacks.

    ranked = sorted(candidates, key=_enrichment_rank, reverse=True)[:limit]
    scored: list[dict] = []

    for candidate in ranked:
        coin = dict(candidate)
        symbol = str(coin.get("symbol", "") or "").upper()
        binance_symbol = str(coin.get("binance_symbol", "") or f"{symbol}USDT")
        history = await _fetch_yfinance_ohlcv(symbol)
        if history is None:
            _apply_fallback_score(coin, "Insufficient OHLCV data for smart scoring")
            scored.append(coin)
            continue

        close, high, low, volume, open_ = history
        taker_ratio = await fetch_taker_buy_ratio(binance_symbol)
        result = score_breakout(
            symbol=symbol,
            close=close,
            high=high,
            low=low,
            volume=volume,
            open_=open_,
            taker_buy_ratio=taker_ratio,
            is_trending="trending_coingecko" in {
                str(source)
                for source in coin.get("sources", [coin.get("source", "unknown")])
                if source
            },
            price_change_24h=float(coin.get("price_change_24h", 0) or 0),
        )

        coin["signal_score"] = int(result.total_score)
        coin["direction"] = result.direction
        coin["breakout_description"] = result.description
        coin["signals_firing"] = int(result.signals_firing)
        coin["key_level"] = result.key_level
        coin["taker_buy_ratio"] = taker_ratio
        coin["_close"] = close
        coin["_high"] = high
        coin["_low"] = low
        coin["_volume"] = volume
        coin["_open"] = open_
        logger.info(
            "Discovery: scored %s %d/100 %s (%d signals)",
            symbol,
            result.total_score,
            result.direction,
            result.signals_firing,
        )
        scored.append(coin)

    scored.sort(
        key=lambda item: (
            int(item.get("signal_score", 0) or 0),
            int(item.get("signals_firing", 0) or 0),
            float(item.get("volume_24h", 0) or 0),
        ),
        reverse=True,
    )
    return scored


def score_candidate(coin: dict) -> int:
    """Fallback score retained for candidates without smart-scoring data."""
    return _basic_score(coin)


def _basic_score(coin: dict) -> int:
    """Fallback score used when smart-scoring data is unavailable."""
    score = 0

    pct = abs(float(coin.get("price_change_24h", 0) or 0))
    if pct >= 20:
        score += 25
    elif pct >= 10:
        score += 15
    elif pct >= 5:
        score += 8

    volume = float(coin.get("volume_24h", 0) or 0)
    if volume >= 10_000_000:
        score += 20
    elif volume >= 5_000_000:
        score += 15
    elif volume >= 1_000_000:
        score += 10

    sources = [
        str(source)
        for source in coin.get("sources", [coin.get("source", "unknown")])
        if source
    ]
    if "trending_coingecko" in sources:
        score += 15
    return min(score, 100)


def build_trigger_reason(coin: dict) -> str:
    sources = [
        str(source)
        for source in coin.get("sources", [coin.get("source", "unknown")])
        if source
    ]
    return "+".join(dict.fromkeys(sources)) if sources else "unknown"


def _apply_fallback_score(coin: dict, description: str) -> None:
    coin["signal_score"] = _basic_score(coin)
    coin["direction"] = "neutral"
    coin["breakout_description"] = description
    coin["signals_firing"] = 0
    coin["key_level"] = None


def _enrichment_rank(candidate: dict) -> float:
    score = 0.0
    score += abs(float(candidate.get("price_change_24h", 0) or 0))
    score += min(float(candidate.get("volume_24h", 0) or 0) / 5_000_000, 20.0)
    sources = {
        str(source)
        for source in candidate.get("sources", [candidate.get("source", "unknown")])
        if source
    }
    if "trending_coingecko" in sources:
        score += 10
    if "binance_volume" in sources:
        score += 8
    if len(sources) > 1:
        score += 5
    return score


async def _fetch_yfinance_ohlcv(
    symbol: str,
) -> tuple[object, object, object, object, object] | None:
    return await asyncio.to_thread(_download_yfinance_ohlcv, symbol.upper())


def _download_yfinance_ohlcv(
    symbol: str,
) -> tuple[object, object, object, object, object] | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; skipping smart scoring for %s", symbol.upper())
        return None

    history = yf.Ticker(f"{symbol.upper()}-USD").history(period="90d", interval="1d", auto_adjust=False)
    required_columns = ("Open", "Close", "High", "Low", "Volume")
    if history.empty or any(column not in history for column in required_columns):
        return None

    cleaned = history.dropna(subset=list(required_columns))
    if len(cleaned) < 20:
        return None

    return (
        cleaned["Close"].to_numpy(dtype=float),
        cleaned["High"].to_numpy(dtype=float),
        cleaned["Low"].to_numpy(dtype=float),
        cleaned["Volume"].to_numpy(dtype=float),
        cleaned["Open"].to_numpy(dtype=float),
    )


def _parse_dollar_string(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def _coin_age_days(payload: dict[str, Any]) -> int | None:
    candidate_dates: list[date] = []

    genesis_date = payload.get("genesis_date")
    if isinstance(genesis_date, str):
        try:
            candidate_dates.append(date.fromisoformat(genesis_date))
        except ValueError:
            pass

    market_data = payload.get("market_data", {})
    if isinstance(market_data, dict):
        for key in ("atl_date", "ath_date"):
            values = market_data.get(key, {})
            if not isinstance(values, dict):
                continue
            raw_value = values.get("usd")
            parsed_date = _iso_to_date(raw_value)
            if parsed_date is not None:
                candidate_dates.append(parsed_date)

    if not candidate_dates:
        return None
    first_seen = min(candidate_dates)
    return max((datetime.now(UTC).date() - first_seen).days, 0)


def _iso_to_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def _coingecko_request_json(
    *,
    endpoint: str,
    params: dict[str, Any],
    api_key: str,
    db_path: str | None,
    ttl_seconds: int,
) -> Any:
    global _coingecko_next_request_at

    query = urlencode(sorted((key, str(value)) for key, value in params.items()))
    cache_key = f"coingecko:{endpoint}:{query}"

    if db_path is not None:
        cached = await _get_cached_json(db_path, cache_key)
        if cached is not None:
            return cached

    async with _coingecko_request_lock:
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait_seconds = _coingecko_next_request_at - now
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
            now = loop.time()
        _coingecko_next_request_at = now + COINGECKO_MIN_REQUEST_INTERVAL_SECONDS

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(
                f"{COINGECKO_BASE_URL}{endpoint}",
                params=params,
                headers=_coingecko_headers(api_key),
            )
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.exception("Discovery: CoinGecko request failed for %s", endpoint)
        return None

    if db_path is not None:
        await _set_cached_json(db_path, cache_key, payload, ttl_seconds=ttl_seconds)
        await _increment_api_usage(db_path, endpoint)
    return payload


async def _get_cached_json(db_path: str, cache_key: str) -> Any | None:
    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT response_json
            FROM api_cache
            WHERE cache_key = ?
              AND datetime(cached_at, '+' || ttl_seconds || ' seconds') > datetime('now')
            """,
            (cache_key,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    try:
        return json.loads(str(row[0]))
    except json.JSONDecodeError:
        return None


async def _set_cached_json(
    db_path: str,
    cache_key: str,
    payload: Any,
    *,
    ttl_seconds: int,
) -> None:
    async with aiosqlite.connect(db_path) as connection:
        await connection.execute(
            """
            INSERT INTO api_cache (cache_key, response_json, cached_at, ttl_seconds)
            VALUES (?, ?, datetime('now'), ?)
            ON CONFLICT(cache_key)
            DO UPDATE SET
                response_json = excluded.response_json,
                cached_at = datetime('now'),
                ttl_seconds = excluded.ttl_seconds
            """,
            (cache_key, json.dumps(payload), ttl_seconds),
        )
        await connection.commit()


async def _increment_api_usage(db_path: str, endpoint: str) -> None:
    async with aiosqlite.connect(db_path) as connection:
        await connection.execute(
            """
            INSERT INTO api_usage (api_name, endpoint, call_date, call_count)
            VALUES ('coingecko', ?, date('now'), 1)
            ON CONFLICT(api_name, endpoint, call_date)
            DO UPDATE SET call_count = call_count + 1
            """,
            (endpoint,),
        )
        await connection.commit()
