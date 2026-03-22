from __future__ import annotations

from src.correlation.models import MarketRegime, RegimeState


def detect_regime(
    btc_qqq_corr_20d: float | None,
    btc_qqq_corr_60d: float | None,
    vix_proxy_change: float,
    spy_change_pct: float,
    btc_change_pct: float,
) -> RegimeState:
    if vix_proxy_change > 5:
        vix_level = "high"
    elif vix_proxy_change > 2:
        vix_level = "elevated"
    else:
        vix_level = "low"

    divergence = None
    if btc_qqq_corr_20d is not None and btc_qqq_corr_60d is not None:
        divergence = btc_qqq_corr_20d - btc_qqq_corr_60d

    corr = btc_qqq_corr_20d

    if vix_level == "high" and spy_change_pct < -1.0 and btc_change_pct < -2.0:
        return RegimeState(
            regime=MarketRegime.RISK_OFF,
            confidence=0.85,
            description="Risk-off: VIX elevated, broad selloff across stocks and crypto",
            btc_qqq_corr=corr,
            vix_level=vix_level,
            divergence_speed=divergence,
            implication_stocks="Defensive positioning; expect continued selling pressure",
            implication_crypto="High contagion risk; crypto often amplifies equity drawdowns in risk-off",
        )

    if spy_change_pct < -0.5 and btc_change_pct > 1.0:
        return RegimeState(
            regime=MarketRegime.DIVERGENT,
            confidence=0.75,
            description="Crypto diverging from equities with BTC showing relative strength",
            btc_qqq_corr=corr,
            vix_level=vix_level,
            divergence_speed=divergence,
            implication_stocks="Equity weakness may be sector-specific rather than systemic",
            implication_crypto="BTC resilience suggests crypto is being driven by independent flows",
        )

    if divergence is not None and abs(divergence) > 0.3:
        direction = "coupling" if divergence > 0 else "decoupling"
        return RegimeState(
            regime=MarketRegime.VOLATILE,
            confidence=0.70,
            description=f"Rapid correlation {direction}; regime transition underway",
            btc_qqq_corr=corr,
            vix_level=vix_level,
            divergence_speed=divergence,
            implication_stocks="Expect elevated volatility and weaker follow-through",
            implication_crypto=(
                "Crypto is tracking equities closely"
                if divergence > 0
                else "Crypto is decoupling and reacting to independent drivers"
            ),
        )

    if vix_level == "low" and spy_change_pct > 0 and btc_change_pct > 0:
        return RegimeState(
            regime=MarketRegime.RISK_ON,
            confidence=0.80,
            description="Risk-on: low volatility and broad asset appreciation",
            btc_qqq_corr=corr,
            vix_level=vix_level,
            divergence_speed=divergence,
            implication_stocks="Favorable backdrop for growth and beta exposure",
            implication_crypto="Positive momentum; altcoins tend to outperform in broad risk-on tape",
        )

    return RegimeState(
        regime=MarketRegime.UNKNOWN,
        confidence=0.40,
        description="Mixed signals; no clear regime classification",
        btc_qqq_corr=corr,
        vix_level=vix_level,
        divergence_speed=divergence,
        implication_stocks="Monitor for clearer trend confirmation before leaning directional",
        implication_crypto="No strong cross-asset signal; stay selective on positioning",
    )
