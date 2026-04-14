from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

MAX_SIZE = 15
DEFAULT_TTL_DAYS = 5


def _utc_sql_timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(UTC)
    return current.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


async def init_hot_list_table(db_path: str) -> None:
    """Create discovery tables if they do not already exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS hot_list (
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
                direction TEXT NOT NULL DEFAULT 'neutral',
                breakout_description TEXT,
                signals_firing INTEGER NOT NULL DEFAULT 0,
                key_level REAL,
                status TEXT NOT NULL DEFAULT 'active',
                binance_symbol TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_hot_list_status_score
                ON hot_list(status, signal_score DESC, expires_at);

            CREATE TABLE IF NOT EXISTS discovery_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT NOT NULL DEFAULT (datetime('now')),
                candidates_scanned INTEGER NOT NULL DEFAULT 0,
                candidates_safe INTEGER NOT NULL DEFAULT 0,
                added_count INTEGER NOT NULL DEFAULT 0,
                expired_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_discovery_runs_run_at
                ON discovery_runs(run_at DESC, id DESC);
            """
        )
        await _ensure_hot_list_columns(db)
        await db.commit()


async def get_active_hot_list(db_path: str) -> list[dict]:
    """Return active hot-list entries ordered by score."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT *
            FROM hot_list
            WHERE status = 'active'
              AND datetime(expires_at) > datetime('now')
            ORDER BY signal_score DESC, added_at DESC, symbol ASC
            """
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def add_to_hot_list(
    db_path: str,
    symbol: str,
    coingecko_id: str,
    name: str,
    price: float,
    volume: float,
    market_cap: float,
    trigger_reason: str,
    signal_score: int,
    price_change_24h: float = 0.0,
    direction: str = "neutral",
    breakout_description: str | None = None,
    signals_firing: int = 0,
    key_level: float | None = None,
    binance_symbol: str | None = None,
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> bool:
    """
    Add a coin to the hot list.

    Returns True when a new active entry is inserted, otherwise False.
    """
    symbol_upper = symbol.upper()
    expires_at = _utc_sql_timestamp(datetime.now(UTC) + timedelta(days=ttl_days))

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT symbol, signal_score
            FROM hot_list
            WHERE symbol = ? AND status = 'active' AND datetime(expires_at) > datetime('now')
            """,
            (symbol_upper,),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            existing_score = int(existing["signal_score"] or 0)
            if signal_score > existing_score:
                await db.execute(
                    """
                    UPDATE hot_list
                    SET coingecko_id = ?,
                        name = ?,
                        expires_at = ?,
                        trigger_reason = ?,
                        initial_price = ?,
                        initial_volume = ?,
                        market_cap = ?,
                        signal_score = ?,
                        price_change_24h = ?,
                        direction = ?,
                        breakout_description = ?,
                        signals_firing = ?,
                        key_level = ?,
                        status = 'active',
                        binance_symbol = ?
                    WHERE symbol = ?
                    """,
                    (
                        coingecko_id,
                        name,
                        expires_at,
                        trigger_reason,
                        price,
                        volume,
                        market_cap,
                        signal_score,
                        price_change_24h,
                        direction,
                        breakout_description,
                        signals_firing,
                        key_level,
                        binance_symbol,
                        symbol_upper,
                    ),
                )
                await db.commit()
                logger.info("Hot list: refreshed %s to score=%d", symbol_upper, signal_score)
            return False

        cursor = await db.execute(
            """
            SELECT COUNT(*)
            FROM hot_list
            WHERE status = 'active' AND datetime(expires_at) > datetime('now')
            """
        )
        row = await cursor.fetchone()
        active_count = int(row[0] or 0) if row is not None else 0

        if active_count >= MAX_SIZE:
            cursor = await db.execute(
                """
                SELECT symbol, signal_score
                FROM hot_list
                WHERE status = 'active' AND datetime(expires_at) > datetime('now')
                ORDER BY signal_score ASC, added_at ASC, symbol ASC
                LIMIT 1
                """
            )
            lowest = await cursor.fetchone()
            lowest_score = int(lowest["signal_score"] or 0) if lowest is not None else 0
            if lowest is None or signal_score <= lowest_score:
                logger.info(
                    "Hot list: rejected %s score=%d because list is full",
                    symbol_upper,
                    signal_score,
                )
                return False
            await db.execute(
                "UPDATE hot_list SET status = 'evicted' WHERE symbol = ?",
                (str(lowest["symbol"]).upper(),),
            )
            logger.info(
                "Hot list: evicted %s score=%d for %s score=%d",
                str(lowest["symbol"]).upper(),
                lowest_score,
                symbol_upper,
                signal_score,
            )

        await db.execute(
            """
            INSERT INTO hot_list (
                symbol,
                coingecko_id,
                name,
                added_at,
                expires_at,
                trigger_reason,
                initial_price,
                initial_volume,
                market_cap,
                signal_score,
                price_change_24h,
                direction,
                breakout_description,
                signals_firing,
                key_level,
                status,
                binance_symbol
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            ON CONFLICT(symbol) DO UPDATE SET
                coingecko_id = excluded.coingecko_id,
                name = excluded.name,
                added_at = excluded.added_at,
                expires_at = excluded.expires_at,
                trigger_reason = excluded.trigger_reason,
                initial_price = excluded.initial_price,
                initial_volume = excluded.initial_volume,
                market_cap = excluded.market_cap,
                signal_score = excluded.signal_score,
                price_change_24h = excluded.price_change_24h,
                direction = excluded.direction,
                breakout_description = excluded.breakout_description,
                signals_firing = excluded.signals_firing,
                key_level = excluded.key_level,
                status = excluded.status,
                binance_symbol = excluded.binance_symbol
            """,
            (
                symbol_upper,
                coingecko_id,
                name,
                _utc_sql_timestamp(),
                expires_at,
                trigger_reason,
                price,
                volume,
                market_cap,
                signal_score,
                price_change_24h,
                direction,
                breakout_description,
                signals_firing,
                key_level,
                binance_symbol,
            ),
        )
        await db.commit()
    logger.info(
        "Hot list: added %s (%s) score=%d reason=%s",
        symbol_upper,
        name,
        signal_score,
        trigger_reason,
    )
    return True


async def _ensure_hot_list_columns(db: aiosqlite.Connection) -> None:
    statements = (
        "ALTER TABLE hot_list ADD COLUMN direction TEXT NOT NULL DEFAULT 'neutral'",
        "ALTER TABLE hot_list ADD COLUMN breakout_description TEXT",
        "ALTER TABLE hot_list ADD COLUMN signals_firing INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE hot_list ADD COLUMN key_level REAL",
    )
    for statement in statements:
        try:
            await db.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


async def expire_old_entries(db_path: str) -> int:
    """Mark expired entries and return the number updated."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            UPDATE hot_list
            SET status = 'expired'
            WHERE status = 'active' AND datetime(expires_at) <= datetime('now')
            """
        )
        await db.commit()
    count = int(cursor.rowcount or 0)
    if count > 0:
        logger.info("Hot list: expired %d coin(s)", count)
    return count


async def remove_from_hot_list(db_path: str, symbol: str, reason: str = "manual") -> bool:
    """Deactivate an entry without deleting its history."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "UPDATE hot_list SET status = ? WHERE symbol = ? AND status = 'active'",
            (reason, symbol.upper()),
        )
        await db.commit()
    return int(cursor.rowcount or 0) > 0


async def get_merged_watchlist(db_path: str, fixed_watchlist: list[str]) -> list[str]:
    """Return fixed crypto symbols followed by unique active hot-list symbols."""
    hot_coins = await get_active_hot_list(db_path)
    fixed_symbols = [symbol.upper() for symbol in fixed_watchlist]
    fixed_set = set(fixed_symbols)
    hot_symbols = [
        str(coin["symbol"]).upper()
        for coin in hot_coins
        if str(coin["symbol"]).upper() not in fixed_set
    ]
    return [*fixed_symbols, *hot_symbols]


async def record_discovery_run(
    db_path: str,
    *,
    candidates_scanned: int,
    candidates_safe: int,
    added_count: int,
    expired_count: int,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO discovery_runs (
                candidates_scanned,
                candidates_safe,
                added_count,
                expired_count
            )
            VALUES (?, ?, ?, ?)
            """,
            (candidates_scanned, candidates_safe, added_count, expired_count),
        )
        await db.commit()


async def get_latest_discovery_run(db_path: str) -> dict[str, object] | None:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT run_at, candidates_scanned, candidates_safe, added_count, expired_count
            FROM discovery_runs
            ORDER BY run_at DESC, id DESC
            LIMIT 1
            """
        )
        row = await cursor.fetchone()
    return dict(row) if row is not None else None
