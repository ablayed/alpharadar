from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SignalInput:
    """A single normalized signal for composite scoring."""

    name: str
    value: float
    confidence: float
    timestamp: float
    half_life_hours: float


@dataclass(slots=True)
class CompositeScore:
    """The fused signal score for one asset."""

    symbol: str
    score: float
    confidence: float
    direction: str
    signal_breakdown: dict[str, float] = field(default_factory=dict)
    agreement: float = 0.0
    regime_context: str = "default"
    summary: str = ""
