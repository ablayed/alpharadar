from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.agents.nodes.mover_selector import mover_selector_node
from src.agents.state import AgentAnalysis
from src.briefing.models import BriefingData
from src.data.base import (
    AssetPrice,
    AssetType,
    DataFreshness,
    DeFiProtocol,
    EarningsEvent,
    NewsItem,
    SECFiling,
)


def _make_asset(
    symbol: str,
    asset_type: AssetType,
    price: float,
    change_24h: float,
    change_7d: float | None = None,
) -> AssetPrice:
    return AssetPrice(
        symbol=symbol,
        asset_type=asset_type,
        current_price=price,
        change_24h_pct=change_24h,
        change_7d_pct=change_7d,
        high_24h=None,
        low_24h=None,
        market_cap=None,
        volume_24h=None,
        previous_close=None,
        freshness=DataFreshness.REALTIME,
        source="test",
    )


def _make_news(
    *,
    headline: str,
    symbol: str,
    score: float,
) -> NewsItem:
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return NewsItem(
        headline=headline,
        summary="",
        source="Test",
        url="https://example.com",
        published_at="2026-03-05T06:00:00+00:00",
        related_symbols=[symbol],
        sentiment_label=label,
        sentiment_score=score,
        sentiment_confidence=0.9,
    )


def _sample_briefing_data() -> BriefingData:
    return BriefingData(
        generated_at=datetime(2026, 3, 5, 7, 0, tzinfo=UTC),
        is_weekend=False,
        market_is_open=True,
        market_session="regular",
        index_prices=[
            _make_asset("SPY", AssetType.STOCK, 584.23, 0.3),
            _make_asset("QQQ", AssetType.STOCK, 421.15, 1.5),
        ],
        stock_prices=[
            _make_asset("NVDA", AssetType.STOCK, 142.3, 3.2, 8.5),
            _make_asset("MSFT", AssetType.STOCK, 410.2, 3.2, 4.1),
        ],
        crypto_prices=[
            _make_asset("BTC", AssetType.CRYPTO, 68420.0, 2.1, 5.5),
            _make_asset("SOL", AssetType.CRYPTO, 168.4, 5.2, 12.1),
        ],
        news_items=[
            _make_news(headline="NVIDIA faces mixed reaction", symbol="NVDA", score=-0.7),
            _make_news(headline="Microsoft secures new enterprise deal", symbol="MSFT", score=0.8),
        ],
        earnings_today=[
            EarningsEvent(
                symbol="NVDA",
                date="2026-03-05",
                hour="amc",
                eps_estimate=0.85,
                revenue_estimate=None,
            )
        ],
        sec_filings=[
            SECFiling(
                company_name="Microsoft",
                ticker="MSFT",
                form_type="8-K",
                filing_date="2026-03-05",
                description="Entry into a Material Agreement",
                url="https://example.com/8k",
            )
        ],
        total_tvl=142_800_000_000,
        tvl_change_24h_pct=3.6,
        defi_movers=[
            DeFiProtocol(
                name="Aave",
                tvl=5_200_000_000,
                change_1d_pct=4.3,
                change_7d_pct=-1.2,
                category="Lending",
                chain="Ethereum",
            )
        ],
        trending_crypto=[{"symbol": "PEPE"}],
        coingecko_usage=(342, 10_000),
        failed_sections=[],
    )


class FakeLLM:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls = 0

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> object:
        del system_prompt, user_prompt, temperature, max_tokens
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return {"raw": "missing"}


@pytest.mark.asyncio
async def test_mover_selector_prefers_largest_move_then_filing() -> None:
    state = {
        "stock_prices": [
            {"symbol": "NVDA", "change_24h_pct": 3.2},
            {"symbol": "MSFT", "change_24h_pct": 3.2},
        ],
        "crypto_prices": [
            {"symbol": "SOL", "change_24h_pct": 2.5},
        ],
        "sec_filings": [
            {"ticker": "MSFT", "form_type": "8-K"},
        ],
        "news_items": [
            {"related_symbols": ["NVDA"], "sentiment_score": -0.8},
        ],
        "is_weekend": False,
    }

    result = await mover_selector_node(state)

    assert result["selected_movers"] == [
        {"symbol": "MSFT", "asset_type": "stock", "reason": "SEC 8-K filing"},
        {"symbol": "NVDA", "asset_type": "stock", "reason": "Strong sentiment divergence"},
    ]


@pytest.mark.asyncio
async def test_mover_selector_forced_symbols_override_default_logic() -> None:
    state = {
        "stock_prices": [{"symbol": "NVDA", "change_24h_pct": 0.2}],
        "crypto_prices": [{"symbol": "BTC", "change_24h_pct": 0.1}],
        "sec_filings": [],
        "news_items": [],
        "is_weekend": True,
        "forced_symbols": ["nvda"],
    }

    result = await mover_selector_node(state)

    assert result["selected_movers"] == [
        {"symbol": "NVDA", "asset_type": "stock", "reason": "Requested via /analyze"}
    ]


@pytest.mark.asyncio
async def test_agent_pipeline_builds_analysis_from_mock_llm() -> None:
    pytest.importorskip("langgraph")
    from src.agents.graph import AgentPipeline

    llm = FakeLLM(
        [
            {
                "summary": "Stocks outperformed with NVDA leading.",
                "signals": ["NVDA up 3.2%", "MSFT filed an 8-K"],
                "asset_highlights": {"NVDA": "Momentum leader"},
            },
            {
                "summary": "Sentiment is mixed but improving for large-cap AI.",
                "signals": ["MSFT headlines positive"],
                "dominant_narrative": "AI capex",
                "contrarian_signal": "NVDA rose despite negative headlines",
            },
            {
                "summary": "Momentum remains constructive.",
                "signals": ["SOL leads crypto"],
                "momentum_leaders": ["NVDA", "SOL"],
                "momentum_laggards": ["BTC"],
                "regime": "mixed",
            },
            {
                "thesis": "Demand remains strong and price action confirms it.",
                "evidence": ["+3.2% move", "positive AI demand backdrop"],
                "risks_acknowledged": ["valuation"],
                "confidence": 0.72,
                "catalyst": "follow-through buying",
            },
            {
                "thesis": "Valuation is stretched relative to near-term execution risk.",
                "evidence": ["negative headlines", "rich multiple"],
                "bull_weaknesses": ["move may be overextended"],
                "confidence": 0.58,
                "risk_trigger": "soft commentary",
            },
            {
                "prevailing_view": "bullish",
                "confidence": 0.68,
                "summary": "The bull case has stronger momentum support, but bears still have a valuation argument.",
                "bull_in_brief": "Momentum and demand support upside.",
                "bear_in_brief": "Valuation remains the main pushback.",
                "key_risk": "A softer demand read-through.",
                "what_to_watch": "Semiconductor follow-through over the next few sessions.",
            },
        ]
    )

    pipeline = AgentPipeline(gemini_client=llm)  # type: ignore[arg-type]
    analysis = await pipeline.run(_sample_briefing_data(), forced_symbols=["NVDA"])

    assert isinstance(analysis, AgentAnalysis)
    assert analysis.total_llm_calls == 6
    assert len(analysis.analyst_reports) == 3
    assert len(analysis.debate_verdicts) == 1
    assert analysis.debate_verdicts[0].symbol == "NVDA"
    assert analysis.debate_verdicts[0].selection_reason == "Requested via /analyze"


@pytest.mark.asyncio
async def test_agent_pipeline_handles_raw_llm_payloads_without_crashing() -> None:
    pytest.importorskip("langgraph")
    from src.agents.graph import AgentPipeline

    llm = FakeLLM([{"raw": "garbage"} for _ in range(6)])
    pipeline = AgentPipeline(gemini_client=llm)  # type: ignore[arg-type]

    analysis = await pipeline.run(_sample_briefing_data(), forced_symbols=["NVDA"])

    assert analysis.total_llm_calls == 6
    assert len(analysis.analyst_reports) == 3
    assert len(analysis.debate_verdicts) == 1
    assert analysis.debate_verdicts[0].prevailing_view == "neutral"
    assert analysis.debate_verdicts[0].selection_reason == "Requested via /analyze"
