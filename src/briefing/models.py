from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.agents.state import AgentAnalysis
from src.data.base import AssetPrice, DeFiProtocol, EarningsEvent, NewsItem, SECFiling


@dataclass(slots=True)
class BriefingData:
    """Complete data package passed to formatter."""

    generated_at: datetime
    is_weekend: bool
    market_is_open: bool
    market_session: str | None

    index_prices: list[AssetPrice]

    stock_prices: list[AssetPrice]
    crypto_prices: list[AssetPrice]

    news_items: list[NewsItem]
    earnings_today: list[EarningsEvent]
    sec_filings: list[SECFiling]

    total_tvl: float | None
    tvl_change_24h_pct: float | None
    defi_movers: list[DeFiProtocol]

    trending_crypto: list[dict[str, object]]

    coingecko_usage: tuple[int, int]

    agent_analysis: AgentAnalysis | None = None
    failed_sections: list[str] = field(default_factory=list)
