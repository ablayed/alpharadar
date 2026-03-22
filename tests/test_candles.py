from __future__ import annotations

import httpx

from src.ta import candles


class FailingFinnhubClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

    async def __aenter__(self) -> FailingFinnhubClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type, exc, tb

    async def get(self, *args: object, **kwargs: object) -> httpx.Response:
        request = httpx.Request("GET", "https://finnhub.io/api/v1/stock/candle")
        response = httpx.Response(
            403,
            request=request,
            json={"error": "You don't have access to this resource."},
        )
        raise httpx.HTTPStatusError("403 forbidden", request=request, response=response)


async def test_fetch_stock_candles_falls_back_to_yfinance(monkeypatch) -> None:
    cached_payloads: list[tuple[str, object, int]] = []
    fallback_payload = {
        "s": "ok",
        "c": [float(100 + idx) for idx in range(80)],
        "h": [float(101 + idx) for idx in range(80)],
        "l": [float(99 + idx) for idx in range(80)],
        "v": [float(1_000 + idx) for idx in range(80)],
    }

    async def fake_get_cached_json(db_path: str | None, cache_key: str) -> object | None:
        _ = db_path, cache_key
        return None

    async def fake_set_cached_json(
        db_path: str | None,
        cache_key: str,
        payload: object,
        *,
        ttl_seconds: int,
    ) -> None:
        _ = db_path
        cached_payloads.append((cache_key, payload, ttl_seconds))

    async def fake_yfinance_stock_candles(symbol: str, days: int) -> dict[str, object] | None:
        assert symbol == "AAPL"
        assert days == 365
        return fallback_payload

    monkeypatch.setattr(candles, "_get_cached_json", fake_get_cached_json)
    monkeypatch.setattr(candles, "_set_cached_json", fake_set_cached_json)
    monkeypatch.setattr(candles, "_fetch_yfinance_stock_candles", fake_yfinance_stock_candles)
    monkeypatch.setattr(candles.httpx, "AsyncClient", FailingFinnhubClient)

    parsed = await candles.fetch_stock_candles("AAPL", "secret", db_path="data/alpharadar.db")

    assert parsed is not None
    close, high, low, volume = parsed
    assert len(close) == 80
    assert len(high) == 80
    assert len(low) == 80
    assert len(volume) == 80
    assert cached_payloads == [("ta:finnhub:candles:AAPL:365", fallback_payload, 21_600)]
