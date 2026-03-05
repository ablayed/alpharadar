from __future__ import annotations

import httpx
import pytest

from src.data.defillama_client import DeFiLlamaClient


@pytest.mark.asyncio
async def test_total_tvl_change_and_movers(db) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v2/historicalChainTvl"):
            return httpx.Response(
                200,
                json=[
                    {"date": 1_640_995_200, "tvl": 180_000_000_000},
                    {"date": 1_641_081_600, "tvl": 189_000_000_000},
                ],
            )
        if request.url.path.endswith("/protocols"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "1",
                        "name": "Aave",
                        "category": "Lending",
                        "chains": ["Ethereum"],
                        "tvl": 5_200_000_000,
                        "change_1d": 4.3,
                        "change_7d": -1.2,
                    },
                    {
                        "id": "2",
                        "name": "Curve",
                        "category": "DEX",
                        "chains": ["Ethereum"],
                        "tvl": 2_800_000_000,
                        "change_1d": -6.0,
                        "change_7d": -2.0,
                    },
                ],
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = DeFiLlamaClient(db=db, transport=httpx.MockTransport(handler))
    await client.initialize()
    tvl, change_pct = await client.get_total_tvl()
    movers = await client.get_top_movers()
    await client.close()

    assert tvl == pytest.approx(189_000_000_000)
    assert change_pct == pytest.approx(5.0)
    assert movers[0].name == "Curve"
    assert movers[0].change_1d_pct == pytest.approx(-6.0)

