from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import aiosqlite

AssetTypeLiteral = Literal["stock", "crypto"]

DEFAULT_STOCKS: list[tuple[str, str, str, None]] = [
    ("NVDA", "stock", "NVIDIA", None),
    ("AAPL", "stock", "Apple", None),
    ("MSFT", "stock", "Microsoft", None),
    ("GOOGL", "stock", "Alphabet", None),
    ("META", "stock", "Meta", None),
    ("AMZN", "stock", "Amazon", None),
    ("AMD", "stock", "AMD", None),
    ("TSLA", "stock", "Tesla", None),
    ("ARM", "stock", "ARM Holdings", None),
    ("PLTR", "stock", "Palantir", None),
]

DEFAULT_CRYPTO: list[tuple[str, str, str, str]] = [
    ("BTC", "crypto", "Bitcoin", "bitcoin"),
    ("ETH", "crypto", "Ethereum", "ethereum"),
    ("SOL", "crypto", "Solana", "solana"),
    ("LINK", "crypto", "Chainlink", "chainlink"),
    ("RNDR", "crypto", "Render", "render-token"),
    ("FET", "crypto", "Fetch.ai", "fetch-ai"),
    ("TAO", "crypto", "Bittensor", "bittensor"),
    ("ONDO", "crypto", "Ondo Finance", "ondo-finance"),
    ("AKT", "crypto", "Akash", "akash-network"),
    ("HNT", "crypto", "Helium", "helium"),
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK (asset_type IN ('stock', 'crypto')),
    display_name TEXT NOT NULL,
    coingecko_id TEXT,
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (symbol, asset_type)
);

CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    call_date TEXT NOT NULL DEFAULT (date('now')),
    call_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(api_name, endpoint, call_date)
);

CREATE TABLE IF NOT EXISTS briefing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at TEXT NOT NULL DEFAULT (datetime('now')),
    briefing_type TEXT NOT NULL CHECK (briefing_type IN ('daily', 'weekend', 'on_demand')),
    content_length INTEGER NOT NULL,
    delivery_status TEXT NOT NULL CHECK (delivery_status IN ('sent', 'failed', 'partial')),
    error_message TEXT,
    sections_included TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS api_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    cached_at TEXT NOT NULL DEFAULT (datetime('now')),
    ttl_seconds INTEGER NOT NULL DEFAULT 300
);

CREATE TABLE IF NOT EXISTS sentiment_cache (
    headline_hash TEXT PRIMARY KEY,
    headline_text TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    positive REAL NOT NULL,
    negative REAL NOT NULL,
    neutral REAL NOT NULL,
    analyzed_at TEXT NOT NULL DEFAULT (datetime('now')),
    model_version TEXT NOT NULL DEFAULT 'finbert-onnx-v1'
);

CREATE TABLE IF NOT EXISTS event_cache (
    headline_hash TEXT PRIMARY KEY,
    headline_text TEXT NOT NULL,
    event_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    classified_at TEXT NOT NULL DEFAULT (datetime('now')),
    model_version TEXT NOT NULL DEFAULT 'gemini-2.5-flash-event-v1'
);
"""


@dataclass(slots=True)
class WatchlistItem:
    symbol: str
    asset_type: AssetTypeLiteral
    display_name: str
    coingecko_id: str | None


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._write_lock = asyncio.Lock()

    @staticmethod
    async def _configure_connection(conn: aiosqlite.Connection, *, enable_wal: bool = False) -> None:
        if enable_wal:
            await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")

    async def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._configure_connection(conn, enable_wal=True)
                await conn.executescript(SCHEMA_SQL)
                await conn.commit()

                cursor = await conn.execute("SELECT COUNT(*) FROM watchlist WHERE is_active = 1")
                row = await cursor.fetchone()
                count = int(row[0]) if row else 0

                if count == 0:
                    await conn.executemany(
                        """
                        INSERT OR IGNORE INTO watchlist (symbol, asset_type, display_name, coingecko_id)
                        VALUES (?, ?, ?, ?)
                        """,
                        [*DEFAULT_STOCKS, *DEFAULT_CRYPTO],
                    )
                    await conn.commit()

    async def get_watchlist(self, asset_type: str | None = None) -> list[WatchlistItem]:
        async with aiosqlite.connect(self.db_path) as conn:
            await self._configure_connection(conn)
            conn.row_factory = aiosqlite.Row
            if asset_type is None:
                cursor = await conn.execute(
                    """
                    SELECT symbol, asset_type, display_name, coingecko_id
                    FROM watchlist
                    WHERE is_active = 1
                    ORDER BY asset_type, symbol
                    """
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT symbol, asset_type, display_name, coingecko_id
                    FROM watchlist
                    WHERE is_active = 1 AND asset_type = ?
                    ORDER BY symbol
                    """,
                    (asset_type,),
                )
            rows = await cursor.fetchall()
            return [
                WatchlistItem(
                    symbol=str(row["symbol"]),
                    asset_type=cast(AssetTypeLiteral, str(row["asset_type"])),
                    display_name=str(row["display_name"]),
                    coingecko_id=(str(row["coingecko_id"]) if row["coingecko_id"] is not None else None),
                )
                for row in rows
            ]

    async def add_to_watchlist(
        self,
        symbol: str,
        asset_type: str,
        display_name: str,
        coingecko_id: str | None = None,
    ) -> bool:
        normalized_symbol = symbol.upper()
        normalized_type = asset_type.lower()
        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._configure_connection(conn)
                cursor = await conn.execute(
                    """
                    INSERT OR IGNORE INTO watchlist (symbol, asset_type, display_name, coingecko_id, is_active)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (normalized_symbol, normalized_type, display_name, coingecko_id),
                )
                await conn.commit()
                return cursor.rowcount > 0

    async def remove_from_watchlist(self, symbol: str, asset_type: str) -> bool:
        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._configure_connection(conn)
                cursor = await conn.execute(
                    """
                    DELETE FROM watchlist
                    WHERE symbol = ? AND asset_type = ?
                    """,
                    (symbol.upper(), asset_type.lower()),
                )
                await conn.commit()
                return cursor.rowcount > 0

    async def increment_api_usage(self, api_name: str, endpoint: str) -> None:
        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._configure_connection(conn)
                await conn.execute(
                    """
                    INSERT INTO api_usage (api_name, endpoint, call_date, call_count)
                    VALUES (?, ?, date('now'), 1)
                    ON CONFLICT(api_name, endpoint, call_date)
                    DO UPDATE SET call_count = call_count + 1
                    """,
                    (api_name, endpoint),
                )
                await conn.commit()

    async def get_monthly_api_usage(self, api_name: str) -> int:
        async with aiosqlite.connect(self.db_path) as conn:
            await self._configure_connection(conn)
            cursor = await conn.execute(
                """
                SELECT COALESCE(SUM(call_count), 0)
                FROM api_usage
                WHERE api_name = ?
                AND strftime('%Y-%m', call_date) = strftime('%Y-%m', date('now'))
                """,
                (api_name,),
            )
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def log_briefing(
        self,
        briefing_type: str,
        content_length: int,
        status: str,
        error: str | None = None,
        sections: list[str] | None = None,
    ) -> None:
        sections_json = json.dumps(sections or [])
        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._configure_connection(conn)
                await conn.execute(
                    """
                    INSERT INTO briefing_history
                    (briefing_type, content_length, delivery_status, error_message, sections_included)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (briefing_type, content_length, status, error, sections_json),
                )
                await conn.commit()

    async def get_cached(self, cache_key: str) -> str | None:
        async with aiosqlite.connect(self.db_path) as conn:
            await self._configure_connection(conn)
            cursor = await conn.execute(
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
            return str(row[0])

    async def set_cached(self, cache_key: str, response_json: str, ttl_seconds: int = 300) -> None:
        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._configure_connection(conn)
                await conn.execute(
                    """
                    INSERT INTO api_cache (cache_key, response_json, cached_at, ttl_seconds)
                    VALUES (?, ?, datetime('now'), ?)
                    ON CONFLICT(cache_key)
                    DO UPDATE SET
                        response_json = excluded.response_json,
                        cached_at = datetime('now'),
                        ttl_seconds = excluded.ttl_seconds
                    """,
                    (cache_key, response_json, ttl_seconds),
                )
                await conn.commit()

    async def get_sentiment_cache_count(self) -> int:
        async with aiosqlite.connect(self.db_path) as conn:
            await self._configure_connection(conn)
            cursor = await conn.execute("SELECT COUNT(*) FROM sentiment_cache")
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def get_event_cache_count(self) -> int:
        async with aiosqlite.connect(self.db_path) as conn:
            await self._configure_connection(conn)
            cursor = await conn.execute("SELECT COUNT(*) FROM event_cache")
            row = await cursor.fetchone()
            return int(row[0]) if row else 0
