"""
LangGraph definition for the AlphaRadar multi-agent analysis pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

from src.agents.llm import GeminiClient
from src.agents.nodes.bear_researcher import bear_researcher_node
from src.agents.nodes.bull_researcher import bull_researcher_node
from src.agents.nodes.debate_judge import debate_judge_node
from src.agents.nodes.market_analyst import market_analyst_node
from src.agents.nodes.mover_selector import mover_selector_node
from src.agents.nodes.sentiment_analyst import sentiment_analyst_node
from src.agents.nodes.technical_analyst import technical_analyst_node
from src.agents.state import AgentAnalysis, AnalystReport, DebateVerdict, PipelineState
from src.events.models import DetectedEvent, EventType

logger = logging.getLogger(__name__)

TASignalFetcher = Callable[[str], Awaitable[dict[str, object] | None]]
LessonsProvider = Callable[[], Awaitable[str]]


class AgentPipeline:
    """Orchestrates the LangGraph multi-agent financial analysis flow."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        *,
        ta_signal_fetcher: TASignalFetcher | None = None,
        lessons_provider: LessonsProvider | None = None,
    ) -> None:
        self._llm = gemini_client
        self._ta_signal_fetcher = ta_signal_fetcher
        self._lessons_provider = lessons_provider
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        llm = self._llm

        async def market_node(state: PipelineState) -> dict[str, object]:
            return await market_analyst_node(state, llm)

        async def sentiment_node(state: PipelineState) -> dict[str, object]:
            return await sentiment_analyst_node(state, llm)

        async def technical_node(state: PipelineState) -> dict[str, object]:
            return await technical_analyst_node(state, llm)

        async def mover_node(state: PipelineState) -> dict[str, object]:
            return await mover_selector_node(state)

        async def ta_node(state: PipelineState) -> dict[str, object]:
            return await self._build_ta_context(state)

        async def bull_node(state: PipelineState) -> dict[str, object]:
            return await bull_researcher_node(state, llm)

        async def bear_node(state: PipelineState) -> dict[str, object]:
            return await bear_researcher_node(state, llm)

        async def judge_node(state: PipelineState) -> dict[str, object]:
            return await debate_judge_node(state, llm)

        graph = StateGraph(PipelineState)
        graph.add_node("market_analyst", market_node)
        graph.add_node("sentiment_analyst", sentiment_node)
        graph.add_node("technical_analyst", technical_node)
        graph.add_node("mover_selector", mover_node)
        graph.add_node("ta_context", ta_node)
        graph.add_node("bull_researcher", bull_node)
        graph.add_node("bear_researcher", bear_node)
        graph.add_node("debate_judge", judge_node)

        graph.add_edge(START, "market_analyst")
        graph.add_edge("market_analyst", "sentiment_analyst")
        graph.add_edge("sentiment_analyst", "technical_analyst")
        graph.add_edge("technical_analyst", "mover_selector")
        graph.add_edge("mover_selector", "ta_context")
        graph.add_edge("ta_context", "bull_researcher")
        graph.add_edge("bull_researcher", "bear_researcher")
        graph.add_edge("bear_researcher", "debate_judge")
        graph.add_edge("debate_judge", END)
        return graph.compile()

    async def run(
        self,
        briefing_data: object,
        *,
        forced_symbols: Sequence[str] | None = None,
    ) -> AgentAnalysis:
        """Execute the pipeline and convert the resulting state into domain types."""
        started_at = time.perf_counter()
        initial_state = self._serialize_briefing_data(briefing_data)
        initial_state["forced_symbols"] = [symbol.upper() for symbol in forced_symbols or []]
        initial_state["ta_signals"] = {}
        initial_state["reflection_lessons"] = await self._load_reflection_lessons()
        initial_state["llm_call_count"] = 0
        initial_state["errors"] = []
        initial_state["selected_movers"] = []
        initial_state["bull_arguments"] = []
        initial_state["bear_arguments"] = []
        initial_state["debate_verdicts"] = []

        logger.info("Starting agent pipeline")
        final_state = await self._graph.ainvoke(initial_state)
        duration = round(time.perf_counter() - started_at, 1)
        llm_calls = int(final_state.get("llm_call_count", 0))
        logger.info("Agent pipeline complete: %s calls in %.1fs", llm_calls, duration)

        analysis = AgentAnalysis(
            total_llm_calls=llm_calls,
            pipeline_duration_seconds=duration,
        )

        for report_key, analyst_name in (
            ("market_report", "market"),
            ("sentiment_report", "sentiment"),
            ("technical_report", "technical"),
        ):
            report = final_state.get(report_key)
            if not isinstance(report, dict):
                continue
            analysis.analyst_reports.append(
                AnalystReport(
                    analyst_name=analyst_name,
                    summary=_coerce_text(report.get("summary")),
                    signals=_coerce_string_list(report.get("signals")),
                    asset_highlights=_coerce_str_dict(report.get("asset_highlights")),
                )
            )

        selected_by_symbol = {
            str(item.get("symbol", "")).upper(): item
            for item in _coerce_dict_list(final_state.get("selected_movers"))
        }
        ta_signals = {
            str(symbol).upper(): payload
            for symbol, payload in _coerce_mapping(final_state.get("ta_signals")).items()
            if isinstance(payload, dict)
        }
        for verdict in _coerce_dict_list(final_state.get("debate_verdicts")):
            symbol = _coerce_text(verdict.get("symbol")) or "?"
            selected = selected_by_symbol.get(symbol.upper(), {})
            ta_signal = ta_signals.get(symbol.upper(), {})
            analysis.debate_verdicts.append(
                DebateVerdict(
                    symbol=symbol,
                    display_name=_coerce_text(verdict.get("display_name")) or symbol,
                    prevailing_view=_normalize_view(_coerce_text(verdict.get("prevailing_view"))),
                    confidence=_coerce_confidence(verdict.get("confidence")),
                    summary=_coerce_text(verdict.get("summary")),
                    bull_thesis=_coerce_text(verdict.get("bull_in_brief")),
                    bear_thesis=_coerce_text(verdict.get("bear_in_brief")),
                    key_risk=_coerce_text(verdict.get("key_risk")),
                    action_context=_coerce_text(verdict.get("what_to_watch")),
                    selection_reason=_coerce_text(selected.get("reason")),
                    ta_summary=_coerce_text(ta_signal.get("summary")),
                )
            )

        return analysis

    async def _build_ta_context(self, state: Mapping[str, object]) -> dict[str, object]:
        if self._ta_signal_fetcher is None:
            return {"ta_signals": {}}

        movers = _coerce_dict_list(state.get("selected_movers"))
        symbols = list(
            dict.fromkeys(
                str(mover.get("symbol", "")).upper()
                for mover in movers
                if str(mover.get("symbol", "")).strip()
            )
        )
        if not symbols:
            return {"ta_signals": {}}

        results = await asyncio.gather(
            *(self._ta_signal_fetcher(symbol) for symbol in symbols),
            return_exceptions=True,
        )
        ta_signals: dict[str, dict[str, object]] = {}
        for symbol, result in zip(symbols, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("TA context failed for %s: %s", symbol, result)
                continue
            if isinstance(result, dict):
                ta_signals[symbol] = result
        return {"ta_signals": ta_signals}

    async def _load_reflection_lessons(self) -> str:
        if self._lessons_provider is None:
            return ""
        try:
            return await self._lessons_provider()
        except Exception:
            logger.exception("Failed to load reflection lessons for agent prompts")
            return ""

    def _serialize_briefing_data(self, data: object) -> PipelineState:
        """Convert BriefingData into plain dictionaries for graph execution."""
        return {
            "index_prices": self._safe_serialize(getattr(data, "index_prices", [])),
            "stock_prices": self._safe_serialize(getattr(data, "stock_prices", [])),
            "crypto_prices": self._safe_serialize(getattr(data, "crypto_prices", [])),
            "news_items": self._safe_serialize(getattr(data, "news_items", [])),
            "event_data": self._serialize_event_data(getattr(data, "event_data", [])),
            "sec_filings": self._safe_serialize(getattr(data, "sec_filings", [])),
            "defi_movers": self._safe_serialize(getattr(data, "defi_movers", [])),
            "total_tvl": _coerce_optional_float(getattr(data, "total_tvl", None)),
            "tvl_change_pct": _coerce_optional_float(getattr(data, "tvl_change_24h_pct", None)),
            "market_is_open": bool(getattr(data, "market_is_open", False)),
            "is_weekend": bool(getattr(data, "is_weekend", False)),
        }

    @staticmethod
    def _serialize_event_data(items: object) -> list[dict[str, object]]:
        if not isinstance(items, Iterable) or isinstance(items, str | bytes):
            return []

        serialized: list[dict[str, object]] = []
        for item in items:
            if isinstance(item, DetectedEvent):
                payload = asdict(item)
                event_type = payload.get("event_type")
                if isinstance(event_type, EventType):
                    payload["event_type"] = event_type.value
                serialized.append(payload)
                continue
            if isinstance(item, dict):
                serialized.append(dict(item))
        return serialized

    @staticmethod
    def _safe_serialize(items: object) -> list[dict[str, object]]:
        if not isinstance(items, Iterable) or isinstance(items, str | bytes):
            return []

        serialized: list[dict[str, object]] = []
        for item in items:
            try:
                payload = asdict(item)
            except TypeError:
                if isinstance(item, dict):
                    payload = dict(item)
                else:
                    payload = {"value": str(item)}
            serialized.append(payload)
        return serialized


def _coerce_dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _coerce_mapping(value: object) -> Mapping[object, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        item_text = _coerce_text(item)
        if key_text and item_text:
            result[key_text] = item_text
    return result


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _coerce_confidence(value: object) -> float:
    if isinstance(value, int | float):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return 0.5
    else:
        return 0.5
    return max(0.0, min(1.0, numeric))


def _normalize_view(value: str) -> str:
    if value in {"bullish", "bearish", "neutral"}:
        return value
    return "neutral"
