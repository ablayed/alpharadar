from __future__ import annotations

import httpx
import pytest

from src.data.macro import _classify_dxy, fetch_fear_greed


@pytest.mark.asyncio
async def test_fetch_fear_greed_uses_contrarian_signal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.alternative.me"
        return httpx.Response(
            200,
            json={"data": [{"value": "18", "value_classification": "Fear"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data = await fetch_fear_greed(client)

    assert data is not None
    assert data.signal == "bullish"
    assert data.value == 18
    assert "contrarian" in data.description


def test_classify_dxy_marks_weekly_dollar_drop_as_bullish() -> None:
    data = _classify_dxy(102.4, -1.3, -0.6)

    assert data.signal == "bullish"
    assert data.current == 102.4
    assert data.change_7d_pct == -1.3
    assert "Dollar down -1.3% this week" in data.description
