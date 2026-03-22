from __future__ import annotations

import logging
from datetime import UTC, datetime
from itertools import combinations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.correlation.models import CorrelationSnapshot, PairCorrelation

logger = logging.getLogger(__name__)


class CorrelationEngine:
    """Compute and track rolling cross-asset correlations."""

    WINDOWS = [20, 30, 60, 90]

    def __init__(self) -> None:
        self._returns: dict[str, np.ndarray] = {}
        self._symbols: list[str] = []
        self.last_updated_at: str | None = None

    def update_prices(self, price_data: dict[str, np.ndarray]) -> None:
        self._returns = {}
        for symbol, prices in price_data.items():
            price_array = np.asarray(prices, dtype=np.float64)
            finite_prices = price_array[np.isfinite(price_array) & (price_array > 0)]
            if len(finite_prices) < 2:
                continue
            self._returns[symbol.upper()] = np.diff(np.log(finite_prices))
        self._symbols = sorted(self._returns.keys())
        self.last_updated_at = datetime.now(UTC).isoformat()

    def compute_snapshot(self, window: int = 30) -> CorrelationSnapshot | None:
        if len(self._returns) < 2:
            return None

        symbols = [symbol for symbol in self._symbols if symbol in self._returns]
        if len(symbols) < 2:
            return None

        min_len = min(len(self._returns[symbol]) for symbol in symbols)
        if min_len < window:
            logger.warning("Not enough data for %d-day correlation (have %d)", window, min_len)
            return None

        returns_matrix = np.stack([self._returns[symbol][-window:] for symbol in symbols])
        matrix = np.nan_to_num(np.corrcoef(returns_matrix), nan=0.0)
        index_by_symbol = {symbol: idx for idx, symbol in enumerate(symbols)}

        def pair_value(symbol_a: str, symbol_b: str) -> float | None:
            idx_a = index_by_symbol.get(symbol_a)
            idx_b = index_by_symbol.get(symbol_b)
            if idx_a is None or idx_b is None:
                return None
            return float(matrix[idx_a, idx_b])

        return CorrelationSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            window_size=window,
            symbols=symbols,
            matrix=matrix,
            btc_spy=pair_value("BTC", "SPY"),
            btc_qqq=pair_value("BTC", "QQQ"),
            btc_eth=pair_value("BTC", "ETH"),
            eth_sol=pair_value("ETH", "SOL"),
        )

    def compute_all_windows(self) -> dict[int, CorrelationSnapshot]:
        snapshots: dict[int, CorrelationSnapshot] = {}
        for window in self.WINDOWS:
            snapshot = self.compute_snapshot(window)
            if snapshot is not None:
                snapshots[window] = snapshot
        return snapshots

    def get_pair_analysis(self, sym_a: str, sym_b: str) -> PairCorrelation | None:
        symbol_a = sym_a.upper()
        symbol_b = sym_b.upper()
        aligned = self._aligned_pair(symbol_a, symbol_b)
        if aligned is None:
            return None

        returns_a, returns_b = aligned
        corr_20 = self._latest_corr(returns_a, returns_b, 20)
        corr_60 = self._latest_corr(returns_a, returns_b, 60)
        corr_90 = self._latest_corr(returns_a, returns_b, 90)
        if corr_20 is None or corr_60 is None or corr_90 is None:
            return None

        divergence = corr_20 - corr_90
        rolling_20 = self._rolling_corr_series(returns_a, returns_b, 20)
        history = rolling_20[:-1][-60:] if len(rolling_20) > 1 else np.array([], dtype=np.float64)
        if len(history) >= 5:
            baseline = float(np.mean(history))
            std = float(np.std(history))
            z_score = (corr_20 - baseline) / max(std, 0.05)
        else:
            z_score = (corr_20 - corr_60) / 0.15

        if abs(z_score) > 2.0:
            regime_signal = "anomalous"
        elif divergence > 0.2:
            regime_signal = "coupling"
        elif divergence < -0.2:
            regime_signal = "decoupling"
        else:
            regime_signal = "stable"

        return PairCorrelation(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            corr_20d=corr_20,
            corr_60d=corr_60,
            corr_90d=corr_90,
            divergence=divergence,
            z_score=z_score,
            regime_signal=regime_signal,
        )

    def get_top_movers(self, n: int = 5) -> list[PairCorrelation]:
        pairs: list[PairCorrelation] = []
        for symbol_a, symbol_b in combinations(self._symbols, 2):
            pair = self.get_pair_analysis(symbol_a, symbol_b)
            if pair is not None:
                pairs.append(pair)
        pairs.sort(key=lambda item: abs(item.divergence), reverse=True)
        return pairs[:n]

    def get_returns(self, symbol: str) -> np.ndarray | None:
        return self._returns.get(symbol.upper())

    def _aligned_pair(self, symbol_a: str, symbol_b: str) -> tuple[np.ndarray, np.ndarray] | None:
        returns_a = self._returns.get(symbol_a)
        returns_b = self._returns.get(symbol_b)
        if returns_a is None or returns_b is None:
            return None
        size = min(len(returns_a), len(returns_b))
        if size < 20:
            return None
        return returns_a[-size:], returns_b[-size:]

    @staticmethod
    def _latest_corr(returns_a: np.ndarray, returns_b: np.ndarray, window: int) -> float | None:
        if min(len(returns_a), len(returns_b)) < window:
            return None
        corr = np.corrcoef(returns_a[-window:], returns_b[-window:])[0, 1]
        return float(np.nan_to_num(corr))

    @staticmethod
    def _rolling_corr_series(returns_a: np.ndarray, returns_b: np.ndarray, window: int) -> np.ndarray:
        size = min(len(returns_a), len(returns_b))
        if size < window:
            return np.array([], dtype=np.float64)

        aligned_a = returns_a[-size:]
        aligned_b = returns_b[-size:]
        windows_a = sliding_window_view(aligned_a, window_shape=window)
        windows_b = sliding_window_view(aligned_b, window_shape=window)

        centered_a = windows_a - windows_a.mean(axis=1, keepdims=True)
        centered_b = windows_b - windows_b.mean(axis=1, keepdims=True)
        numerator = np.sum(centered_a * centered_b, axis=1)
        denominator = np.sqrt(np.sum(centered_a**2, axis=1) * np.sum(centered_b**2, axis=1))
        correlation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > 0,
        )
        return np.nan_to_num(correlation, nan=0.0)
