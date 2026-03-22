from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(slots=True)
class CorrelationSnapshot:
    """A point-in-time correlation state for all asset pairs."""

    timestamp: str
    window_size: int
    symbols: list[str]
    matrix: np.ndarray
    btc_spy: float | None = None
    btc_qqq: float | None = None
    btc_eth: float | None = None
    eth_sol: float | None = None


@dataclass(slots=True)
class PairCorrelation:
    """Correlation data for a specific pair across multiple windows."""

    symbol_a: str
    symbol_b: str
    corr_20d: float
    corr_60d: float
    corr_90d: float
    divergence: float
    z_score: float
    regime_signal: str


class MarketRegime(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    DIVERGENT = "divergent"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RegimeState:
    regime: MarketRegime
    confidence: float
    description: str
    btc_qqq_corr: float | None
    vix_level: str
    divergence_speed: float | None
    implication_stocks: str
    implication_crypto: str


@dataclass(slots=True)
class LeadLagResult:
    symbol_a: str
    symbol_b: str
    peak_lag: int
    peak_correlation: float
    interpretation: str
