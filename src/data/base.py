from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"


class DataFreshness(str, Enum):
    REALTIME = "realtime"
    DELAYED_15M = "delayed_15m"
    HOURLY = "hourly"
    DAILY = "daily"


@dataclass(slots=True)
class AssetPrice:
    symbol: str
    asset_type: AssetType
    current_price: float
    change_24h_pct: float
    change_7d_pct: float | None
    high_24h: float | None
    low_24h: float | None
    market_cap: float | None
    volume_24h: float | None
    previous_close: float | None
    freshness: DataFreshness
    source: str


@dataclass(slots=True)
class NewsItem:
    headline: str
    summary: str
    source: str
    url: str
    published_at: str
    related_symbols: list[str]
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    sentiment_confidence: float | None = None
    event_type: str | None = None
    event_confidence: float | None = None
    expected_car_pct: float | None = None
    expected_car_window: str | None = None
    remaining_drift_pct: float | None = None


@dataclass(slots=True)
class EarningsEvent:
    symbol: str
    date: str
    hour: str
    eps_estimate: float | None
    revenue_estimate: float | None


@dataclass(slots=True)
class SECFiling:
    company_name: str
    ticker: str
    form_type: str
    filing_date: str
    description: str
    url: str


@dataclass(slots=True)
class DeFiProtocol:
    name: str
    tvl: float
    change_1d_pct: float
    change_7d_pct: float
    category: str
    chain: str | None


class BaseDataProvider(ABC):
    """All data providers implement this interface."""

    # Phase 5: New providers - ArkhamClient, LunarCrushClient, DEXScreenerClient

    @abstractmethod
    async def initialize(self) -> None:
        """Set up HTTP client, validate API key, etc."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and authenticated."""
