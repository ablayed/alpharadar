from __future__ import annotations

import numpy as np

from src.correlation import CorrelationEngine, MarketRegime, compute_lead_lag, detect_regime


def test_correlation_engine_computes_pair_analysis_and_top_movers() -> None:
    base = np.linspace(100.0, 140.0, 120)
    btc = base * 1.8
    eth = base * 1.2
    sol = np.concatenate([base[:60] * 0.95, (base[60:] * 1.7) + np.linspace(0.0, 40.0, 60)])

    engine = CorrelationEngine()
    engine.update_prices(
        {
            "QQQ": base,
            "BTC": btc,
            "ETH": eth,
            "SOL": sol,
        }
    )

    snapshot = engine.compute_snapshot(30)
    assert snapshot is not None
    assert snapshot.window_size == 30
    assert snapshot.btc_qqq is not None

    pair = engine.get_pair_analysis("BTC", "QQQ")
    assert pair is not None
    assert pair.symbol_a == "BTC"
    assert pair.symbol_b == "QQQ"
    assert -1.0 <= pair.corr_20d <= 1.0
    assert -1.0 <= pair.corr_60d <= 1.0
    assert -1.0 <= pair.corr_90d <= 1.0

    movers = engine.get_top_movers(2)
    assert len(movers) == 2
    assert abs(movers[0].divergence) >= abs(movers[1].divergence)


def test_detect_regime_classifies_divergent_and_risk_off() -> None:
    divergent = detect_regime(
        btc_qqq_corr_20d=0.15,
        btc_qqq_corr_60d=0.55,
        vix_proxy_change=1.0,
        spy_change_pct=-0.8,
        btc_change_pct=2.3,
    )
    assert divergent.regime == MarketRegime.DIVERGENT

    risk_off = detect_regime(
        btc_qqq_corr_20d=0.75,
        btc_qqq_corr_60d=0.70,
        vix_proxy_change=6.0,
        spy_change_pct=-1.6,
        btc_change_pct=-3.4,
    )
    assert risk_off.regime == MarketRegime.RISK_OFF


def test_compute_lead_lag_detects_leading_series() -> None:
    seed = np.sin(np.linspace(0.0, 6.0, 80))
    lagged = np.concatenate([np.zeros(2), seed[:-2]])

    result = compute_lead_lag(seed, lagged, "QQQ", "BTC", max_lag=4)

    assert result.peak_lag > 0
    assert "QQQ leads BTC" in result.interpretation
