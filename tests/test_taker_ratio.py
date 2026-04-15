from __future__ import annotations

import pytest

from src.data import taker_ratio


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *, payload, timeout: float, follow_redirects: bool):
        self._payload = payload
        self.timeout = timeout
        self.follow_redirects = follow_redirects

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, params: dict[str, object]):
        assert url == taker_ratio.BINANCE_AGG_TRADES_URL
        assert params["symbol"] == "BTCUSDT"
        assert params["limit"] == 500
        return _FakeResponse(self._payload)


@pytest.mark.asyncio
async def test_fetch_taker_buy_ratio_parses_buyer_initiated_volume(monkeypatch) -> None:
    trades = [
        {"q": "2", "m": False},
        {"q": "3", "m": True},
        {"q": "5", "m": False},
    ]
    monkeypatch.setattr(
        taker_ratio.httpx,
        "AsyncClient",
        lambda timeout, follow_redirects: _FakeAsyncClient(
            payload=trades,
            timeout=timeout,
            follow_redirects=follow_redirects,
        ),
    )

    ratio = await taker_ratio.fetch_taker_buy_ratio("btcusdt")

    assert ratio == 0.7


@pytest.mark.asyncio
async def test_fetch_taker_ratios_batch_collects_results(monkeypatch) -> None:
    async def fake_fetch(symbol: str, limit: int = 500) -> float | None:
        assert limit == 500
        return {
            "BTCUSDT": 0.61,
            "ETHUSDT": 0.47,
            "DOGEUSDT": None,
        }[symbol]

    monkeypatch.setattr(taker_ratio, "fetch_taker_buy_ratio", fake_fetch)

    result = await taker_ratio.fetch_taker_ratios_batch(["BTCUSDT", "ETHUSDT", "DOGEUSDT"])

    assert result == {
        "BTCUSDT": 0.61,
        "ETHUSDT": 0.47,
        "DOGEUSDT": None,
    }
