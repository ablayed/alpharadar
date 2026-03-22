from __future__ import annotations

import time

from src.signals.models import CompositeScore, SignalInput

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on": {
        "sentiment": 0.15,
        "ta": 0.30,
        "event": 0.20,
        "correlation": 0.10,
        "agent": 0.25,
    },
    "risk_off": {
        "sentiment": 0.25,
        "ta": 0.15,
        "event": 0.25,
        "correlation": 0.20,
        "agent": 0.15,
    },
    "volatile": {
        "sentiment": 0.20,
        "ta": 0.20,
        "event": 0.20,
        "correlation": 0.25,
        "agent": 0.15,
    },
    "divergent": {
        "sentiment": 0.15,
        "ta": 0.25,
        "event": 0.15,
        "correlation": 0.30,
        "agent": 0.15,
    },
    "default": {
        "sentiment": 0.15,
        "ta": 0.25,
        "event": 0.20,
        "correlation": 0.15,
        "agent": 0.25,
    },
}


def compute_composite(
    symbol: str,
    signals: list[SignalInput],
    regime: str = "default",
) -> CompositeScore:
    now = time.time()
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["default"])
    weighted_values: dict[str, float] = {}
    total_weight = 0.0

    for signal in signals:
        age_hours = max(0.0, (now - signal.timestamp) / 3600)
        decay = 0.5 ** (age_hours / signal.half_life_hours)
        decayed_value = signal.value * decay * signal.confidence
        weight = weights.get(signal.name, 0.15)
        weighted_values[signal.name] = decayed_value * weight
        total_weight += weight

    raw_score = (sum(weighted_values.values()) / total_weight) if total_weight > 0 else 0.0
    score = max(-1.0, min(1.0, raw_score))

    directions = [1 if value > 0.05 else (-1 if value < -0.05 else 0) for value in weighted_values.values()]
    if directions:
        dominant = max(set(directions), key=directions.count)
        agreement = directions.count(dominant) / len(directions)
    else:
        agreement = 0.0

    if score > 0.4:
        direction = "strong_buy"
    elif score > 0.15:
        direction = "buy"
    elif score < -0.4:
        direction = "strong_sell"
    elif score < -0.15:
        direction = "sell"
    else:
        direction = "neutral"

    confidence = min(0.95, agreement * 0.7 + len(signals) * 0.05)
    leading = sorted(weighted_values.items(), key=lambda item: abs(item[1]), reverse=True)[:2]
    drivers = []
    for name, value in leading:
        bias = "bullish" if value > 0 else "bearish" if value < 0 else "neutral"
        drivers.append(f"{name} ({bias})")
    summary = (
        f"{symbol.upper()}: Composite {direction.replace('_', ' ')} ({score:+.2f}). "
        f"Driven by {' and '.join(drivers) if drivers else 'limited signal input'}. "
        f"Signal agreement: {agreement:.0%}. Regime: {regime}."
    )

    return CompositeScore(
        symbol=symbol.upper(),
        score=score,
        confidence=confidence,
        direction=direction,
        signal_breakdown=weighted_values,
        agreement=agreement,
        regime_context=regime,
        summary=summary,
    )
