from __future__ import annotations

from datetime import UTC, datetime

from src.briefing.formatter import BriefingFormatter
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


def _sample_data(is_weekend: bool) -> BriefingData:
    return BriefingData(
        generated_at=datetime(2026, 3, 5, 7, 0, tzinfo=UTC),
        is_weekend=is_weekend,
        market_is_open=True,
        market_session="regular",
        index_prices=[
            _make_asset("S&P 500", AssetType.STOCK, 5842.3, 0.32),
            _make_asset("NASDAQ", AssetType.STOCK, 18421.5, 0.51),
            _make_asset("VIX", AssetType.STOCK, 14.2, 1.8),
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
            NewsItem(
                headline="Datacenter revenue beat & guidance raise",
                summary="",
                source="Test",
                url="https://example.com",
                published_at="2026-03-05T06:00:00+00:00",
                related_symbols=["NVDA"],
            )
        ],
        earnings_today=[EarningsEvent(symbol="NVDA", date="2026-03-05", hour="amc", eps_estimate=0.85, revenue_estimate=None)],
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
        tvl_change_24h_pct=0.6,
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


def test_format_daily_contains_sections_and_html_escape() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=False)
    text = formatter.format_daily(data)

    assert "<b>MARKET SNAPSHOT</b>" in text
    assert "<b>STOCKS</b>" in text
    assert "<b>CRYPTO</b>" in text
    assert "<b>CATALYSTS</b>" in text
    assert "&amp;" in text
    assert len(text) <= 4096


def test_format_weekend_excludes_stock_and_catalysts() -> None:
    formatter = BriefingFormatter()
    data = _sample_data(is_weekend=True)
    text = formatter.format_weekend(data)

    assert "<b>STOCKS</b>" not in text
    assert "<b>CATALYSTS</b>" not in text
    assert "<b>DEFI MOVERS</b>" in text
    assert len(text) <= 4096

