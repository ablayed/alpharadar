from __future__ import annotations

import numpy as np

from src.correlation.models import LeadLagResult


def compute_lead_lag(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    symbol_a: str,
    symbol_b: str,
    max_lag: int = 5,
) -> LeadLagResult:
    size = min(len(returns_a), len(returns_b))
    if size < 30:
        return LeadLagResult(symbol_a.upper(), symbol_b.upper(), 0, 0.0, "Insufficient data")

    aligned_a = np.asarray(returns_a[-size:], dtype=np.float64)
    aligned_b = np.asarray(returns_b[-size:], dtype=np.float64)

    best_lag = 0
    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            current_a = aligned_a
            current_b = aligned_b
        elif lag > 0:
            current_a = aligned_a[:-lag]
            current_b = aligned_b[lag:]
        else:
            current_a = aligned_a[-lag:]
            current_b = aligned_b[:lag]

        if len(current_a) < 3 or len(current_b) < 3:
            continue

        corr = _safe_corr(current_a, current_b)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    symbol_a = symbol_a.upper()
    symbol_b = symbol_b.upper()
    if best_lag == 0:
        interpretation = f"{symbol_a} and {symbol_b} move simultaneously (lag 0)"
    elif best_lag > 0:
        interpretation = f"{symbol_a} leads {symbol_b} by {best_lag} day(s) (corr: {best_corr:.2f})"
    else:
        interpretation = f"{symbol_b} leads {symbol_a} by {abs(best_lag)} day(s) (corr: {best_corr:.2f})"

    return LeadLagResult(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        peak_lag=best_lag,
        peak_correlation=best_corr,
        interpretation=interpretation,
    )


def _safe_corr(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if np.std(values_a) == 0 or np.std(values_b) == 0:
        return 0.0
    return float(np.nan_to_num(np.corrcoef(values_a, values_b)[0, 1]))
