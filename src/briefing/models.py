from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.agents.state import AgentAnalysis
from src.correlation.models import LeadLagResult, PairCorrelation, RegimeState
from src.data.base import AssetPrice, DeFiProtocol, EarningsEvent, NewsItem, SECFiling
from src.data.derivatives import CMEGap, FundingData, OpenInterestData
from src.data.macro import BTCDominanceData, DXYData, FearGreedData
from src.events.models import DetectedEvent
from src.signals.models import CompositeScore
from src.ta.signals import TASignal


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
    hot_list: list[dict[str, object]] = field(default_factory=list)

    event_data: list[DetectedEvent] = field(default_factory=list)
    ta_signals: dict[str, TASignal] = field(default_factory=dict)
    correlation_pairs: dict[str, PairCorrelation] = field(default_factory=dict)
    lead_lag_pairs: dict[str, LeadLagResult] = field(default_factory=dict)
    correlation_top_movers: list[PairCorrelation] = field(default_factory=list)
    regime_state: RegimeState | None = None
    composite_scores: dict[str, CompositeScore] = field(default_factory=dict)
    agent_analysis: AgentAnalysis | None = None
    funding: dict[str, FundingData] = field(default_factory=dict)
    oi: dict[str, OpenInterestData] = field(default_factory=dict)
    fear_greed: FearGreedData | None = None
    dxy: DXYData | None = None
    cme_gap: CMEGap | None = None
    btc_dominance: BTCDominanceData | None = None
    failed_sections: list[str] = field(default_factory=list)
