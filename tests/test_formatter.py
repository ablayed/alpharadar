from __future__ import annotations

from datetime import UTC, datetime

from src.briefing.formatter import (
    MAX_TELEGRAM_LENGTH,
    BriefingFormatter,
    _send_briefing_parts,
    _sentiment_indicator,
    _sentiment_summary,
)
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
    label: str | None,
    score: float | None,
    confidence: float | None,
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
            _make_asset("NVDA", AssetType.STOCK, 142.3, 3.2),
            _make_asset("TSLA", AssetType.STOCK, 178.5, -1.8),
            _make_asset("AAPL", AssetType.STOCK, 198.2, 0.1),
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
        failed_sections=[],
    )


def test_format_daily_contains_context_annotations_and_no_filler() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=False)
    text = formatter.format_daily(data)

    assert "<b>MARKET SNAPSHOT</b>" in text
    assert "<b>STOCKS</b>" in text
    assert "<b>CRYPTO</b>" in text
    assert "<b>CATALYSTS</b>" in text
    assert "SPY: $584.23 +0.3% (S&amp;P 500 proxy)" in text
    assert "QQQ: $421.15 🟢 +1.5% (NASDAQ proxy)" in text
    assert "<i>VIX at 31 signals high fear  DeFi capital inflows accelerating</i>" in text
    assert "🟢 <b>NVDA</b> $142.30 jumping +3.2%" in text
    assert "    NVIDIA datacenter revenue beat &amp; guidance raise" in text
    assert "🔴 <b>TSLA</b> $178.50 dipping -1.8%" in text
    assert "<b>AAPL</b> $198.20 +0.1%" in text
    assert "No significant news" not in text
    assert "🟢 <b>SOL</b> $168.40 rallying +5.2% (7d: +12.1%  solid weekly gains)" in text
    assert "🔴 <b>FET</b> $1.82 dipping -3.1% (7d: -8.4%  weak weekly trend)" in text
    assert "Sentiment: 📈 1 pos  📉 1 neg  1 neu" in text
    assert "📋 8-K: MSFT  Entry into a Material Agreement (filed Mar 5)" in text
    assert "CG budget: 342/10,000  Next briefing: Tomorrow 7:00 AM" in text


def test_format_weekend_excludes_stock_and_catalysts() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=True)
    text = formatter.format_weekend(data)

    assert "<b>STOCKS</b>" not in text
    assert "<b>CATALYSTS</b>" not in text
    assert "<b>DEFI MOVERS</b>" in text
    assert "CG budget: 342/10,000  Next briefing: Tomorrow 7:00 AM" in text


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
    assert _sentiment_indicator(mixed) == "📊 "
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
