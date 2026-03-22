from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping

import aiosqlite
import httpx
import numpy as np

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


async def fetch_stock_candles(
    symbol: str,
    api_key: str,
    days: int = 365,
    *,
    db_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    now = int(time.time())
    start = now - (days * 86400)
    cache_key = f"ta:finnhub:candles:{symbol.upper()}:{days}"
    cached = await _get_cached_json(db_path, cache_key)
    if isinstance(cached, dict):
        return _parse_finnhub_candles(cached)

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(
                f"{FINNHUB_BASE}/stock/candle",
                params={
                    "symbol": symbol.upper(),
                    "resolution": "D",
                    "from": start,
                    "to": now,
                    "token": api_key,
                },
            )
            response.raise_for_status()
            payload = response.json()
            await _set_cached_json(db_path, cache_key, payload, ttl_seconds=6 * 3600)
            return _parse_finnhub_candles(payload)
        except Exception:
            logger.exception("Failed to fetch stock candles for %s", symbol.upper())
            return None


async def fetch_crypto_candles(
    coin_id: str,
    days: int = 365,
    *,
    db_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    cache_key = f"ta:coingecko:candles:{coin_id}:{days}"
    cached = await _get_cached_json(db_path, cache_key)
    if isinstance(cached, list):
        return _parse_coingecko_candles(cached)

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(
                f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": days},
            )
            response.raise_for_status()
            payload = response.json()
            await _set_cached_json(db_path, cache_key, payload, ttl_seconds=3600)
            await _increment_coingecko_usage(db_path, "/coins/{id}/ohlc")
            return _parse_coingecko_candles(payload)
        except Exception:
            logger.exception("Failed to fetch crypto candles for %s", coin_id)
            return None


def _parse_finnhub_candles(
    payload: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    status = payload.get("s")
    close = payload.get("c")
    high = payload.get("h")
    low = payload.get("l")
    volume = payload.get("v")
    if status != "ok" or not all(isinstance(values, list) for values in (close, high, low, volume)):
        return None

    close_array = np.asarray(close, dtype=np.float64)
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    if len(close_array) < 60:
        return None
    return close_array, high_array, low_array, volume_array


def _parse_coingecko_candles(
    payload: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not isinstance(payload, list) or len(payload) < 60:
        return None

    try:
        array = np.asarray(payload, dtype=np.float64)
    except Exception:
        return None
    if array.ndim != 2 or array.shape[1] < 5:
        return None

    high_array = array[:, 2]
    low_array = array[:, 3]
    close_array = array[:, 4]
    volume_array = np.zeros_like(close_array)
    return close_array, high_array, low_array, volume_array


async def _get_cached_json(db_path: str | None, cache_key: str) -> object | None:
    if db_path is None:
        return None
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
    return json.loads(str(row[0]))


async def _set_cached_json(
    db_path: str | None,
    cache_key: str,
    payload: object,
    *,
    ttl_seconds: int,
) -> None:
    if db_path is None:
        return
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


async def _increment_coingecko_usage(db_path: str | None, endpoint: str) -> None:
    if db_path is None:
        return
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
