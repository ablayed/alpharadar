from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import aiosqlite
import httpx
import pytest
import pytest_asyncio

from scripts.backfill_historical_outcomes import backfill_historical_outcomes
from src.reflection import historical_backfill
from src.reflection import tracker as tracker_module
from src.storage.database import Database


async def _insert_prediction(
    db_path: str,
    *,
    created_at: datetime,
    symbol: str,
    predicted_direction: str = "bullish",
    price_at_prediction: float = 100.0,
    outcome_recorded: int = 0,
    actual_price_1d: float | None = None,
    actual_price_3d: float | None = None,
    actual_price_5d: float | None = None,
    actual_price_7d: float | None = None,
    metadata: str | None = '{"xgb_prob": 0.8, "rf_prob": 0.7, "lr_prob": 0.6}',
) -> int:
    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            INSERT INTO predictions (
                created_at,
                symbol,
                asset_type,
                predicted_direction,
                confidence,
                reasoning,
                metadata,
                source,
                price_at_prediction,
                outcome_recorded,
                actual_price_1d,
                actual_price_3d,
                actual_price_5d,
                actual_price_7d
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at.strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                "crypto",
                predicted_direction,
                0.8,
                "test",
                metadata,
                "analyze",
                price_at_prediction,
                outcome_recorded,
                actual_price_1d,
                actual_price_3d,
                actual_price_5d,
                actual_price_7d,
            ),
        )
        await connection.commit()
        return int(cursor.lastrowid)


def _price_map(now: datetime) -> dict[tuple[str, str], float]:
    return {
        ("BTC", (now - timedelta(days=7)).date().isoformat()): 101.0,
        ("BTC", (now - timedelta(days=5)).date().isoformat()): 103.0,
        ("BTC", (now - timedelta(days=3)).date().isoformat()): 106.0,
        ("BTC", (now - timedelta(days=1)).date().isoformat()): 104.0,
        ("SOL", (now - timedelta(days=3)).date().isoformat()): 99.0,
        ("SOL", (now - timedelta(days=1)).date().isoformat()): 102.0,
        ("ADA", (now - timedelta(days=1)).date().isoformat()): 105.0,
    }


def _fixture_now() -> datetime:
    return datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def backfill_db():
    test_dir = Path(".test_tmp")
    test_dir.mkdir(exist_ok=True)
    db_path = test_dir / f"{uuid4().hex}.db"
    database = Database(str(db_path))
    await database.initialize()
    try:
        yield database
    finally:
        for _ in range(5):
            try:
                db_path.unlink(missing_ok=True)
                break
            except PermissionError:
                await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_historical_backfill_dry_run_makes_no_writes_and_preserves_ordering(backfill_db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    now = _fixture_now()
    btc_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=8), symbol="BTC")
    eth_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=8), symbol="ETH")
    sol_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=4), symbol="SOL")
    ada_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=2), symbol="ADA")
    xrp_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(hours=12), symbol="XRP")

    prices = _price_map(now)

    async def historical_price_fetcher(symbol: str, asset_type: str, target_at: datetime) -> float | None:
        del asset_type
        return prices.get((symbol, target_at.date().isoformat()))

    def _fail_if_called(*args: object, **kwargs: object) -> bool:
        raise AssertionError("MLEnsemble performance path must not be touched during backfill")

    monkeypatch.setattr(tracker_module, "_record_ml_performance_outcome", _fail_if_called)

    summary = await backfill_historical_outcomes(
        db_path=backfill_db.db_path,
        historical_price_fetcher=historical_price_fetcher,
        dry_run=True,
        now=now,
    )

    assert summary.to_dict() == {
        "dry_run": True,
        "passes": {
            "1d": {
                "scanned": 4,
                "resolved": 3,
                "skipped": {
                    "no_historical_price_found": 1,
                    "not_old_enough": 1,
                },
            },
            "3d": {
                "scanned": 2,
                "resolved": 2,
                "skipped": {
                    "dependency_pass_not_yet_satisfied": 1,
                    "not_old_enough": 2,
                },
            },
            "5d": {
                "scanned": 1,
                "resolved": 1,
                "skipped": {
                    "dependency_pass_not_yet_satisfied": 1,
                    "not_old_enough": 3,
                },
            },
            "7d": {
                "scanned": 1,
                "resolved": 1,
                "skipped": {
                    "dependency_pass_not_yet_satisfied": 1,
                    "not_old_enough": 3,
                },
            },
        },
    }

    async with aiosqlite.connect(backfill_db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT id,
                   outcome_recorded,
                   actual_price_1d,
                   actual_price_3d,
                   actual_price_5d,
                   actual_price_7d
            FROM predictions
            WHERE id IN (?, ?, ?, ?, ?)
            ORDER BY id
            """,
            (btc_id, eth_id, sol_id, ada_id, xrp_id),
        )
        rows = await cursor.fetchall()

    assert rows == [
        (btc_id, 0, None, None, None, None),
        (eth_id, 0, None, None, None, None),
        (sol_id, 0, None, None, None, None),
        (ada_id, 0, None, None, None, None),
        (xrp_id, 0, None, None, None, None),
    ]


@pytest.mark.asyncio
async def test_historical_backfill_live_writes_expected_values_and_is_idempotent(backfill_db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    now = _fixture_now()
    btc_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=8), symbol="BTC")
    await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=8), symbol="ETH")
    sol_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=4), symbol="SOL")
    ada_id = await _insert_prediction(backfill_db.db_path, created_at=now - timedelta(days=2), symbol="ADA")

    prices = _price_map(now)

    async def historical_price_fetcher(symbol: str, asset_type: str, target_at: datetime) -> float | None:
        del asset_type
        return prices.get((symbol, target_at.date().isoformat()))

    def _fail_if_called(*args: object, **kwargs: object) -> bool:
        raise AssertionError("MLEnsemble performance path must not be touched during backfill")

    monkeypatch.setattr(tracker_module, "_record_ml_performance_outcome", _fail_if_called)

    first_summary = await backfill_historical_outcomes(
        db_path=backfill_db.db_path,
        historical_price_fetcher=historical_price_fetcher,
        dry_run=False,
        now=now,
    )
    second_summary = await backfill_historical_outcomes(
        db_path=backfill_db.db_path,
        historical_price_fetcher=historical_price_fetcher,
        dry_run=False,
        now=now,
    )

    assert first_summary.to_dict()["passes"]["1d"]["resolved"] == 3
    assert first_summary.to_dict()["passes"]["3d"]["resolved"] == 2
    assert first_summary.to_dict()["passes"]["5d"]["resolved"] == 1
    assert first_summary.to_dict()["passes"]["7d"]["resolved"] == 1
    assert second_summary.to_dict()["passes"]["1d"]["resolved"] == 0
    assert second_summary.to_dict()["passes"]["3d"]["resolved"] == 0
    assert second_summary.to_dict()["passes"]["5d"]["resolved"] == 0
    assert second_summary.to_dict()["passes"]["7d"]["resolved"] == 0

    async with aiosqlite.connect(backfill_db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT outcome_recorded,
                   actual_price_1d,
                   actual_change_1d_pct,
                   is_correct_1d,
                   actual_direction,
                   actual_price_3d,
                   actual_change_3d_pct,
                   is_correct_3d,
                   actual_price_5d,
                   actual_change_5d_pct,
                   is_correct_5d,
                   actual_price_7d,
                   actual_change_7d_pct,
                   is_correct_7d
            FROM predictions
            WHERE id = ?
            """,
            (btc_id,),
        )
        btc_row = await cursor.fetchone()
        cursor = await connection.execute(
            """
            SELECT outcome_recorded,
                   actual_price_1d,
                   actual_change_1d_pct,
                   is_correct_1d,
                   actual_direction,
                   actual_price_3d,
                   actual_change_3d_pct,
                   is_correct_3d
            FROM predictions
            WHERE id = ?
            """,
            (sol_id,),
        )
        sol_row = await cursor.fetchone()
        cursor = await connection.execute(
            """
            SELECT outcome_recorded,
                   actual_price_1d,
                   actual_change_1d_pct,
                   is_correct_1d,
                   actual_price_3d,
                   actual_price_5d,
                   actual_price_7d
            FROM predictions
            WHERE id = ?
            """,
            (ada_id,),
        )
        ada_row = await cursor.fetchone()

    assert btc_row == (1, 101.0, 1.0, 1, "bullish", 103.0, 3.0, 1, 106.0, 6.0, 1, 104.0, 4.0, 1)
    assert sol_row == (1, 99.0, -1.0, 0, "bearish", 102.0, 2.0, 1)
    assert ada_row == (1, 105.0, 5.0, 1, None, None, None)


@pytest.mark.asyncio
async def test_repository_historical_price_fetcher_falls_back_after_finnhub_403(
    backfill_db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_at = _fixture_now() - timedelta(days=1)
    fetcher = historical_backfill.RepositoryHistoricalPriceFetcher(
        db_path=backfill_db.db_path,
        finnhub_api_key="test-key",
    )

    request = httpx.Request("GET", "https://finnhub.io/api/v1/stock/candle")
    response = httpx.Response(403, request=request)

    async def fake_finnhub(symbol: str, target_at_value: datetime, api_key: str) -> float | None:
        del symbol, target_at_value, api_key
        raise httpx.HTTPStatusError("Forbidden", request=request, response=response)

    def fake_yfinance(symbol: str, target_at_value: datetime) -> float | None:
        assert symbol == "AAPL"
        assert target_at_value == target_at
        return 111.0

    monkeypatch.setattr(historical_backfill, "_fetch_finnhub_stock_close", fake_finnhub)
    monkeypatch.setattr(historical_backfill, "_download_yfinance_stock_close", fake_yfinance)

    price = await fetcher("AAPL", "stock", target_at)

    assert price == 111.0


@pytest.mark.asyncio
async def test_coingecko_history_endpoint_parses_usd_price(
    backfill_db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_at = datetime(2026, 3, 23, 12, 0, tzinfo=UTC)
    captured_urls: list[str] = []

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

        async def get(self, url: str, params: dict[str, object] | None = None) -> httpx.Response:
            request = httpx.Request("GET", url, params=params)
            captured_urls.append(str(request.url))
            return httpx.Response(
                200,
                request=request,
                json={"market_data": {"current_price": {"usd": 123.45}}},
            )

    monkeypatch.setattr(historical_backfill.httpx, "AsyncClient", _FakeAsyncClient)

    price = await historical_backfill._fetch_coingecko_crypto_close("bitcoin", target_at)

    assert price == 123.45
    assert captured_urls == [
        "https://api.coingecko.com/api/v3/coins/bitcoin/history?date=23-03-2026&localization=false"
    ]


@pytest.mark.asyncio
async def test_coingecko_429_retries_are_bounded(
    backfill_db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_at = datetime(2026, 3, 23, 12, 0, tzinfo=UTC)
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

        async def get(self, url: str, params: dict[str, object] | None = None) -> httpx.Response:
            del params
            attempts["count"] += 1
            request = httpx.Request("GET", url)
            return httpx.Response(
                429,
                request=request,
                headers={"Retry-After": "0"},
                json={
                    "status": {
                        "error_code": 429,
                        "error_message": "rate limited",
                    }
                },
            )

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(historical_backfill.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(historical_backfill.asyncio, "sleep", fake_sleep)

    price = await historical_backfill._fetch_coingecko_crypto_close("bitcoin", target_at)

    assert price is None
    assert attempts["count"] == 3
    assert sleep_calls == [0.0, 0.0]


@pytest.mark.asyncio
async def test_render_symbol_is_not_auto_populated_with_guessed_coingecko_id(
    backfill_db: Database,
) -> None:
    async with aiosqlite.connect(backfill_db.db_path) as connection:
        await connection.execute(
            """
            INSERT INTO watchlist (
                symbol,
                asset_type,
                display_name,
                is_active,
                coingecko_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("RENDER", "crypto", "Render", 1, None),
        )
        await connection.commit()

    fetcher = historical_backfill.RepositoryHistoricalPriceFetcher(db_path=backfill_db.db_path)
    price = await fetcher("RENDER", "crypto", _fixture_now())

    assert price is None

    async with aiosqlite.connect(backfill_db.db_path) as connection:
        cursor = await connection.execute(
            "SELECT coingecko_id FROM watchlist WHERE symbol = ?",
            ("RENDER",),
        )
        row = await cursor.fetchone()

    assert row == (None,)
