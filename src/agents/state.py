"""
Shared types for the LangGraph multi-agent market analysis pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


@dataclass(slots=True)
class AnalystReport:
    """Output from a single analyst agent."""

    analyst_name: str
    summary: str
    signals: list[str] = field(default_factory=list)
    asset_highlights: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DebateArgument:
    """A single argument in the Bull/Bear debate."""

    side: str
    thesis: str
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass(slots=True)
class DebateVerdict:
    """The judge's synthesis of the Bull/Bear debate."""

    symbol: str
    display_name: str
    prevailing_view: str
    confidence: float
    summary: str
    bull_thesis: str
    bear_thesis: str
    key_risk: str
    action_context: str
    selection_reason: str = ""
    ta_summary: str = ""
    composite_summary: str = ""
    regime_context: str = ""


@dataclass(slots=True)
class AgentAnalysis:
    """Complete output of the agent pipeline, attached to BriefingData."""

    analyst_reports: list[AnalystReport] = field(default_factory=list)
    debate_verdicts: list[DebateVerdict] = field(default_factory=list)
    total_llm_calls: int = 0
    pipeline_duration_seconds: float = 0.0


class PipelineState(TypedDict, total=False):
    """LangGraph state passed between pipeline nodes."""

    index_prices: list[dict[str, object]]
    stock_prices: list[dict[str, object]]
    crypto_prices: list[dict[str, object]]
    news_items: list[dict[str, object]]
    event_data: list[dict[str, object]]
    sec_filings: list[dict[str, object]]
    defi_movers: list[dict[str, object]]
    total_tvl: float | None
    tvl_change_pct: float | None
    market_is_open: bool
    is_weekend: bool
    forced_symbols: list[str]
    ta_signals: dict[str, dict[str, object]]
    correlation_pairs: dict[str, dict[str, object]]
    lead_lag_pairs: dict[str, dict[str, object]]
    composite_scores: dict[str, dict[str, object]]
    regime_state: dict[str, object] | None
    reflection_lessons: str

    market_report: dict[str, object] | None
    sentiment_report: dict[str, object] | None
    technical_report: dict[str, object] | None

    selected_movers: list[dict[str, object]]
    bull_arguments: list[dict[str, object]]
    bear_arguments: list[dict[str, object]]
    debate_verdicts: list[dict[str, object]]

    llm_call_count: int
    errors: list[str]
