from __future__ import annotations

import pytest

from src.data.aggregator import DataAggregator
from src.storage.database import WatchlistItem


class _FakeDB:
    async def get_watchlist(self, asset_type: str | None = None) -> list[WatchlistItem]:
        assert asset_type == "crypto"
        return [
            WatchlistItem(
                symbol="RNDR",
                asset_type="crypto",
                display_name="Render",
                coingecko_id="render-token",
            )
        ]


@pytest.mark.asyncio
async def test_coin_id_map_includes_render_alias() -> None:
    aggregator = object.__new__(DataAggregator)
    aggregator.db = _FakeDB()

    coin_map = await DataAggregator._coin_id_map(aggregator)

    assert coin_map["RNDR"] == "render-token"
    assert coin_map["RENDER"] == "render-token"
