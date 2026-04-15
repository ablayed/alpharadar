from __future__ import annotations

from datetime import UTC, datetime

from src.agents.state import AgentAnalysis, DebateVerdict, LLMUsageSummary
from src.data.derivatives import CMEGap
from src.data.macro import BTCDominanceData, DXYData, FearGreedData
from src.briefing.formatter import (
    MAX_TELEGRAM_LENGTH,
    BriefingFormatter,
    _format_event_annotation,
    _send_briefing_parts,
    _sentiment_indicator,
    _sentiment_summary,
)
from src.briefing.models import BriefingData
from src.correlation.models import MarketRegime, RegimeState
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
    label: str | None,
    score: float | None,
    confidence: float | None,
    event_type: str | None = None,
    event_confidence: float | None = None,
    expected_car_pct: float | None = None,
    expected_car_window: str | None = None,
    remaining_drift_pct: float | None = None,
) -> NewsItem:
    return NewsItem(
        headline=headline,
        summary="",
        source="Test",
        url="https://example.com",
        published_at="2026-03-05T06:00:00+00:00",
        related_symbols=[symbol],
        sentiment_label=label,
        sentiment_score=score,
        sentiment_confidence=confidence,
        event_type=event_type,
        event_confidence=event_confidence,
        expected_car_pct=expected_car_pct,
        expected_car_window=expected_car_window,
        remaining_drift_pct=remaining_drift_pct,
    )


def _sample_data(is_weekend: bool) -> BriefingData:
    return BriefingData(
        generated_at=datetime(2026, 3, 5, 7, 0, tzinfo=UTC),
        is_weekend=is_weekend,
        market_is_open=True,
        market_session="regular",
        index_prices=[
            _make_asset("SPY", AssetType.STOCK, 584.23, 0.32),
            _make_asset("QQQ", AssetType.STOCK, 421.15, 1.51),
            _make_asset("VIX", AssetType.STOCK, 31.2, -9.5),
        ],
        stock_prices=[
            _make_asset("NVDA", AssetType.STOCK, 142.3, 3.2, 8.4),
            _make_asset("TSLA", AssetType.STOCK, 178.5, -1.8, -6.2),
            _make_asset("AAPL", AssetType.STOCK, 198.2, 0.1, 0.3),
        ],
        crypto_prices=[
            _make_asset("BTC", AssetType.CRYPTO, 68420.0, 2.1, 2.1),
            _make_asset("ETH", AssetType.CRYPTO, 3180.0, 1.8, 4.5),
            _make_asset("SOL", AssetType.CRYPTO, 168.4, 5.2, 12.1),
            _make_asset("FET", AssetType.CRYPTO, 1.82, -3.1, -8.4),
        ],
        news_items=[
            _make_news(
                headline="NVIDIA datacenter revenue beat & guidance raise",
                symbol="NVDA",
                label="positive",
                score=0.82,
                confidence=0.91,
                event_type="guidance_raise",
                event_confidence=0.82,
                expected_car_pct=2.0,
                expected_car_window="[-1, +3] days",
                remaining_drift_pct=1.4,
            ),
            _make_news(
                headline="Vehicle recall pressures near-term margins",
                symbol="TSLA",
                label="negative",
                score=-0.64,
                confidence=0.88,
            ),
            _make_news(
                headline="Apple scheduled to report earnings next Thursday",
                symbol="AAPL",
                label="neutral",
                score=0.02,
                confidence=0.61,
            ),
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
        trending_crypto=[
            {"symbol": "PEPE"},
            {"symbol": "WIF"},
            {"symbol": "TAO"},
            {"symbol": "DOGE"},
            {"symbol": "RNDR"},
        ],
        coingecko_usage=(342, 10_000),
        fear_greed=FearGreedData(
            value=18,
            classification="Fear",
            signal="bullish",
            description="Fear remains elevated",
        ),
        dxy=DXYData(
            current=102.4,
            change_7d_pct=-1.4,
            btc_correlation_30d=-0.62,
            signal="bullish",
            description="Dollar weakening",
        ),
        cme_gap=CMEGap(
            gap_low=66_000.0,
            gap_high=67_200.0,
            gap_direction="up",
            distance_pct=2.7,
            filled=False,
            description="Gap below",
        ),
        btc_dominance=BTCDominanceData(
            dominance_pct=54.8,
            change_7d=0.0,
            season="neutral",
            description="Rotation mixed",
        ),
        failed_sections=[],
    )


def _sample_agent_analysis() -> AgentAnalysis:
    return AgentAnalysis(
        debate_verdicts=[
            DebateVerdict(
                symbol="NVDA",
                display_name="NVDA",
                prevailing_view="bullish",
                confidence=0.72,
                summary="Datacenter demand remains firm and the latest move is backed by strong sentiment.",
                bull_thesis="AI infrastructure demand still supports upside momentum.",
                bear_thesis="Valuation leaves little room for a softer guide.",
                key_risk="Any negative revision to AI capex guidance.",
                action_context="Watch for demand commentary and follow-through in semis.",
                selection_reason="+3.2% move",
            )
        ],
        llm_usage=[
            LLMUsageSummary(
                role="general",
                provider="openrouter",
                model="deepseek/deepseek-v3.2",
                model_display="DeepSeek V3.2",
                calls=3,
            ),
            LLMUsageSummary(
                role="bull",
                provider="openrouter",
                model="deepseek/deepseek-r1-0528",
                model_display="DeepSeek R1 0528",
                calls=1,
            ),
            LLMUsageSummary(
                role="bear",
                provider="openrouter",
                model="deepseek/deepseek-r1-0528",
                model_display="DeepSeek R1 0528",
                calls=1,
            ),
            LLMUsageSummary(
                role="judge",
                provider="openrouter",
                model="google/gemini-2.5-pro-preview",
                model_display="Gemini 2.5 Pro",
                calls=1,
            ),
        ],
        total_llm_calls=6,
        pipeline_duration_seconds=12.3,
    )


def test_format_daily_contains_context_annotations_and_no_filler() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=False)
    text = formatter.format_daily(data)

    assert "<b>MARKET SNAPSHOT</b>" in text
    assert "<b>STOCKS</b>" in text
    assert "<b>CRYPTO</b>" in text
    assert "<b>CATALYSTS</b>" in text
    assert "<b>MACRO CONTEXT</b>" in text
    assert "SPY: $584.23 +0.3% (S&amp;P 500 proxy)" in text
    assert "QQQ: $421.15 🟢 +1.5% (NASDAQ proxy)" in text
    assert "Fear &amp; Greed: 18 (Fear)" in text
    assert "Dollar (DXY): 102.40 (-1.4% this week)" in text
    assert "BTC Dominance: 54.8% neutral" in text
    assert "CME Gap: $66,000-$67,200 unfilled (2.7% away)" in text
    assert "<i>VIX at 31 signals high fear  DeFi capital inflows accelerating</i>" in text
    assert "🟢 <b>NVDA</b> $142.30 jumping +3.2%" in text
    assert "(+0.82) NVIDIA datacenter revenue beat &amp; guidance raise" in text
    assert "    [guidance_raise]" in text
    assert "Hist. avg: +2.0% over [-1, +3] days  remaining 1.4% upside expected" in text
    assert "🔴 <b>TSLA</b> $178.50 dipping -1.8%" in text
    assert "(-0.64) Vehicle recall pressures near-term margins" in text
    assert "<b>AAPL</b> $198.20 +0.1%" in text
    assert "Apple scheduled to report earnings next Thursday" in text
    assert "No significant news" not in text
    assert "🟢 <b>SOL</b> $168.40 rallying +5.2% (7d: +12.1%  solid weekly gains)" in text
    assert "🔴 <b>FET</b> $1.82 dipping -3.1% (7d: -8.4%  weak weekly trend)" in text
    assert "Sentiment: 📈 1 pos  📉 1 neg  1 neu" in text
    assert "📋 8-K: MSFT  Entry into a Material Agreement (filed Mar 5)" in text
    assert "CG budget: 342/10,000  Next briefing: Tomorrow 7:00 AM" in text


def test_format_weekend_shows_weekly_sections_instead_of_daily_stock_blocks() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=True)
    text = formatter.format_weekend(data)

    assert "<b>STOCKS</b>" not in text
    assert "<b>CATALYSTS</b>" not in text
    assert "<b>WEEK IN REVIEW</b>" in text
    assert "best performer" in text
    assert "worst performer" in text
    assert "<b>CRYPTO WEEK</b>" in text
    assert "<b>DEFI MOVERS</b>" in text
    assert "CG budget: 342/10,000  Next briefing: Tomorrow 7:00 AM" in text


def test_format_daily_includes_ai_deep_dive_when_present() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=False)
    data.agent_analysis = _sample_agent_analysis()

    text = formatter.format_daily(data)

    assert "<b>AI DEEP DIVE</b>" in text
    assert "NVDA" in text
    assert "Bullish (72%)" in text
    assert "Bulls say:" not in text
    assert "Bears say:" not in text
    assert "<b>TA:</b>" not in text
    assert "<b>Composite:</b>" not in text
    assert "<b>Regime:</b>" not in text
    assert "<b>Risk:</b>" in text
    assert "Watch:" in text
    assert "Pipeline: 6 LLM calls in 12.3s" in text
    assert "General: OpenRouter / DeepSeek V3.2 (3 calls)" in text
    assert "Judge: OpenRouter / Gemini 2.5 Pro (1 call)" in text


def test_format_daily_simplifies_market_regime_section() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=False)
    data.regime_state = RegimeState(
        regime=MarketRegime.RISK_OFF,
        confidence=0.85,
        description="Risk-off: VIX elevated, broad selloff across stocks and crypto",
        btc_qqq_corr=0.72,
        vix_level="high",
        divergence_speed=-0.15,
        implication_stocks="Defensive positioning expected",
        implication_crypto="High contagion risk from equities",
    )

    text = formatter.format_daily(data)

    assert "<b>MARKET MOOD</b>" in text
    assert "Fearful" in text
    assert "investors are selling across the board" in text
    assert "Crypto is at risk of following stocks lower." in text
    assert "confidence" not in text
    assert "BTC-QQQ corr" not in text
    assert "MARKET REGIME" not in text


def test_format_hot_list_section_splits_cex_and_dex_entries() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=False)
    data.hot_list = [
        {
            "symbol": "DOT",
            "signal_score": 55,
            "direction": "long_setup",
            "source": "trending_coingecko",
        },
        {
            "symbol": "GRIFF",
            "signal_score": 72,
            "direction": "long_setup",
            "source": "dexscreener",
            "chain_id": "solana",
        },
    ]

    text = formatter.format_daily(data)

    assert "<b>HOT LIST</b> (2 coins)" in text
    assert "CEX: DOT 55pts 🟢" in text
    assert "DEX: GRIFF 72pts 🌐 Solana" in text


def test_sentiment_indicator_thresholds() -> None:
    positive = _make_news(
        headline="Positive",
        symbol="NVDA",
        label="positive",
        score=0.55,
        confidence=0.82,
    )
    mixed = _make_news(
        headline="Mixed",
        symbol="AAPL",
        label="positive",
        score=0.18,
        confidence=0.54,
    )
    neutral = _make_news(
        headline="Neutral",
        symbol="MSFT",
        label="neutral",
        score=0.03,
        confidence=0.61,
    )
    negative = _make_news(
        headline="Negative",
        symbol="TSLA",
        label="negative",
        score=-0.51,
        confidence=0.9,
    )

    assert _sentiment_indicator(positive) == "📈 "
    assert _sentiment_indicator(mixed) == "📈 "
    assert _sentiment_indicator(neutral) == ""
    assert _sentiment_indicator(negative) == "📉 "


def test_sentiment_summary_counts_missing_data_as_neutral() -> None:
    news_items = [
        _make_news(headline="One", symbol="NVDA", label="positive", score=0.8, confidence=0.9),
        _make_news(headline="Two", symbol="TSLA", label="negative", score=-0.6, confidence=0.9),
        _make_news(headline="Three", symbol="AAPL", label=None, score=None, confidence=None),
    ]

    assert _sentiment_summary(news_items) == "Sentiment: 📈 1 pos  📉 1 neg  1 neu"


def test_send_briefing_parts_splits_at_section_boundaries() -> None:
    long_section = "\n".join(f"Line {idx} {'x' * 80}" for idx in range(80))
    message = (
        "Header\n\n"
        f"<b>STOCKS</b>\n{long_section}\n\n"
        f"<b>CRYPTO</b>\n{long_section}\n\n"
        "<i>Footer</i>"
    )

    parts = _send_briefing_parts(message)

    assert len(parts) >= 2
    assert all(len(part) <= MAX_TELEGRAM_LENGTH for part in parts)
    assert any("<b>CRYPTO</b>" in part for part in parts)


def test_format_event_annotation_skips_general_or_low_confidence_events() -> None:
    general = _make_news(
        headline="General update",
        symbol="AAPL",
        label="neutral",
        score=0.0,
        confidence=0.6,
        event_type="general_news",
        event_confidence=0.9,
        expected_car_pct=0.0,
        expected_car_window="N/A",
        remaining_drift_pct=None,
    )
    low_confidence = _make_news(
        headline="Possible partnership",
        symbol="MSFT",
        label="positive",
        score=0.2,
        confidence=0.7,
        event_type="partnership",
        event_confidence=0.4,
        expected_car_pct=1.0,
        expected_car_window="[-1, +3] days",
        remaining_drift_pct=0.8,
    )

    assert _format_event_annotation(general) == ""
    assert _format_event_annotation(low_confidence) == ""


def test_format_agent_verdict_keeps_full_text_for_analyze() -> None:
    formatter = BriefingFormatter()
    analysis = AgentAnalysis(
        debate_verdicts=[
            DebateVerdict(
                symbol="NVDA",
                display_name="NVDA",
                prevailing_view="bullish",
                confidence=0.74,
                summary=" ".join(["Detailed"] * 120),
                bull_thesis=" ".join(["Bull"] * 90),
                bear_thesis=" ".join(["Bear"] * 90),
                key_risk=" ".join(["Risk"] * 70),
                action_context=" ".join(["Watch"] * 70),
            )
        ],
        total_llm_calls=6,
        pipeline_duration_seconds=3.2,
    )

    text = formatter.format_agent_verdict("NVDA", analysis)

    assert text is not None
    assert "Detailed Detailed Detailed" in text
    assert "Bull Bull Bull" in text
    assert "Bear Bear Bear" in text
    assert "Risk Risk Risk" in text
    assert "Watch Watch Watch" in text
