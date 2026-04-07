from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.agents.providers import LLMRole
from src.briefing.models import BriefingData
from src.data.aggregator import DataAggregator
from src.data.base import AssetPrice, AssetType, DataFreshness, NewsItem
from src.storage.database import WatchlistItem
from src.ta.signals import TASignal


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


class _FakePlaybookLLM:
    def __init__(self) -> None:
        self.user_prompt = ""
        self.role: LLMRole | None = None

    def mark_usage(self) -> int:
        return 0

    def usage_since(self, marker: int) -> list[object]:
        assert marker == 0
        return []

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        role: LLMRole = LLMRole.GENERAL,
    ) -> str:
        del system_prompt, temperature, max_tokens
        self.user_prompt = user_prompt
        self.role = role
        return "playbook output"


def _sample_playbook_briefing() -> BriefingData:
    long_headline = (
        "NVIDIA secures a major enterprise AI expansion deal while traders debate whether the move has "
        "already priced in the near-term upside and how quickly margin gains can follow through across "
        "the next few sessions."
    )
    return BriefingData(
        generated_at=datetime(2026, 4, 7, 14, 30, tzinfo=UTC),
        is_weekend=False,
        market_is_open=True,
        market_session="regular",
        index_prices=[],
        stock_prices=[
            AssetPrice(
                symbol="NVDA",
                asset_type=AssetType.STOCK,
                current_price=142.3,
                change_24h_pct=3.2,
                change_7d_pct=8.5,
                high_24h=None,
                low_24h=None,
                market_cap=None,
                volume_24h=None,
                previous_close=None,
                freshness=DataFreshness.REALTIME,
                source="test",
            )
        ],
        crypto_prices=[],
        news_items=[
            NewsItem(
                headline=long_headline,
                summary="",
                source="Test",
                url="https://example.com",
                published_at="2026-04-07T14:00:00+00:00",
                related_symbols=["NVDA"],
                sentiment_label="positive",
                sentiment_score=0.81,
                sentiment_confidence=0.9,
            )
        ],
        earnings_today=[],
        sec_filings=[],
        total_tvl=None,
        tvl_change_24h_pct=None,
        defi_movers=[],
        trending_crypto=[],
        coingecko_usage=(0, 10_000),
        ta_signals={
            "NVDA": TASignal(
                symbol="NVDA",
                timeframe="daily",
                current_price=142.3,
                rsi_value=58.0,
                rsi_signal="neutral",
                macd_line=1.2,
                macd_signal_line=0.9,
                macd_histogram=0.3,
                macd_signal="bullish_cross",
                bb_upper=148.6,
                bb_lower=135.4,
                bb_position="near_upper",
                bb_squeeze=False,
                atr_value=4.2,
                atr_pct=2.95,
                volatility="normal",
                ema_20=139.8,
                ema_50=132.1,
                trend="strong_up",
                overall_signal="buy",
                confidence=0.74,
                summary="NVDA: MACD bullish cross, trend strong up. Signal: buy (74%).",
            )
        },
    )


@pytest.mark.asyncio
async def test_coin_id_map_includes_render_alias() -> None:
    aggregator = object.__new__(DataAggregator)
    aggregator.db = _FakeDB()

    coin_map = await DataAggregator._coin_id_map(aggregator)

    assert coin_map["RNDR"] == "render-token"
    assert coin_map["RENDER"] == "render-token"


@pytest.mark.asyncio
async def test_generate_playbook_uses_compact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    aggregator = object.__new__(DataAggregator)
    llm = _FakePlaybookLLM()

    async def fake_ensure_llm_client() -> _FakePlaybookLLM:
        return llm

    async def fake_log_playbook_predictions(playbook_text: str, notable_assets: list[dict[str, object]]) -> None:
        del playbook_text, notable_assets

    monkeypatch.setattr(aggregator, "_ensure_llm_client", fake_ensure_llm_client)
    monkeypatch.setattr(aggregator, "_log_playbook_predictions", fake_log_playbook_predictions)

    playbook_text, llm_calls, _duration, llm_usage = await aggregator.generate_playbook(_sample_playbook_briefing())

    assert playbook_text == "playbook output"
    assert llm_calls == 1
    assert llm_usage == []
    assert llm.role == LLMRole.PLAYBOOK
    assert "Headline: NVIDIA secures a major enterprise AI expansion deal" in llm.user_prompt
    assert "..." in llm.user_prompt
    assert "Levels: support ~$135.40, EMA20 ~$139.80, resistance ~$148.60" in llm.user_prompt
    assert "--- Technical Analysis ---" not in llm.user_prompt
    assert "--- Correlation Context ---" not in llm.user_prompt
    assert "--- Composite Score ---" not in llm.user_prompt
