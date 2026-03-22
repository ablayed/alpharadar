"""
Synthesizes bull and bear arguments into a balanced verdict.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """You are an expert financial debate judge evaluating bull vs bear cases.

CRITICAL RULES:
1. You MUST assign different confidence levels based on evidence quality.
   - 90%+ = overwhelming evidence on one side (rare)
   - 75-89% = strong evidence with minor counterarguments
   - 60-74% = moderate lean, both sides have merit
   - 50-59% = very slight lean, evidence nearly balanced

2. You MUST NOT default to 70% for everything. Evaluate EACH debate independently.

3. If one side has concrete data such as CAR calibration, price targets, or specific
   catalysts and the other side has vague concerns, the data-driven side should win
   with HIGH confidence.

4. If both sides are vague, confidence should be LOW (50-60%).

5. Your summary must explain WHY you lean one way. Cite the specific evidence that
   tipped the balance. Do not use generic phrases like "both sides have merit."

6. Include SPECIFIC numbers in your verdict: price levels, percentage targets, and
   timeframes from the bull and bear arguments.

Respond in JSON:
{
  "prevailing_view": "bullish | bearish | neutral",
  "confidence": 0.78,
  "summary": "3-4 sentences explaining your verdict with specific numbers and reasoning. What tipped the balance? What is the key evidence?",
  "bull_in_brief": "One sentence with the specific bullish price target or catalyst",
  "bear_in_brief": "One sentence with the specific bearish risk or downside target",
  "key_risk": "The single most important risk, with a specific trigger level or event",
  "what_to_watch": "One specific, measurable thing to monitor in the next 1-5 days"
}

Remember: neutral is ONLY for genuinely split evidence. Most debates have a lean - find it."""


async def debate_judge_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Judge the debate and return one verdict per mover."""
    movers = _dict_list(state.get("selected_movers"))
    bull_args = _dict_list(state.get("bull_arguments"))
    bear_args = _dict_list(state.get("bear_arguments"))
    event_data = _dict_list(state.get("event_data"))
    ta_signals = _mapping_dict(state.get("ta_signals"))

    verdicts: list[dict[str, object]] = []
    call_count = _int_or_zero(state.get("llm_call_count"))

    for index, mover in enumerate(movers):
        symbol = str(mover.get("symbol", "?"))
        bull = bull_args[index] if index < len(bull_args) else {}
        bear = bear_args[index] if index < len(bear_args) else {}
        relevant_events = [
            event
            for event in event_data
            if symbol.upper() in {related.upper() for related in _string_list(event.get("related_symbols"))}
        ]
        ta_data = _as_mapping(ta_signals.get(symbol.upper()))

        user_prompt = f"""Debate for: {symbol}

BULL CASE:
Thesis: {bull.get('thesis', 'N/A')}
Evidence: {bull.get('evidence', [])}
Confidence: {bull.get('confidence', 'N/A')}
Catalyst: {bull.get('catalyst', 'N/A')}

BEAR CASE:
Thesis: {bear.get('thesis', 'N/A')}
Evidence: {bear.get('evidence', [])}
Bull weaknesses: {bear.get('bull_weaknesses', [])}
Confidence: {bear.get('confidence', 'N/A')}
Risk trigger: {bear.get('risk_trigger', 'N/A')}

CALIBRATED EVENT EVIDENCE:
{_format_symbol_events(relevant_events)}

TECHNICAL ANALYSIS:
{_format_ta_data(ta_data)}

Deliver a decisive verdict for {symbol}. Choose the side with the stronger evidence unless it is truly balanced."""

        result = await llm.generate_json(
            system_prompt=_system_prompt(state),
            user_prompt=user_prompt,
            temperature=0.0,
        )
        result["symbol"] = symbol
        result["display_name"] = symbol
        verdicts.append(result)
        call_count += 1

    return {"debate_verdicts": verdicts, "llm_call_count": call_count}


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _mapping_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key).upper(): item for key, item in value.items()}
    return {}


def _system_prompt(state: Mapping[str, object]) -> str:
    lessons = str(state.get("reflection_lessons", "")).strip()
    if not lessons:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}{lessons}"


def _int_or_zero(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _float_or_zero(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _format_symbol_events(events: list[dict[str, object]]) -> str:
    if not events:
        return "  None"

    lines: list[str] = []
    for event in events:
        remaining = event.get("remaining_drift_pct")
        remaining_text = "n/a"
        if isinstance(remaining, int | float):
            remaining_text = f"{float(remaining):+.1f}%"
        lines.append(
            "  "
            f"{event.get('event_type', 'unknown')} "
            f"({(_float_or_zero(event.get('confidence'))):.0%} confidence) | "
            f"hist {_float_or_zero(event.get('expected_car_pct')):+.1f}% over "
            f"{event.get('expected_car_window', 'N/A')} | "
            f"current {_float_or_zero(event.get('actual_move_pct')):+.1f}% | "
            f"remaining {remaining_text}"
        )
    return "\n".join(lines)


def _format_ta_data(ta_data: Mapping[str, object]) -> str:
    if not ta_data:
        return "  None"
    return "\n".join(
        [
            f"  RSI(14): {_float_or_zero(ta_data.get('rsi_value')):.1f} ({ta_data.get('rsi_signal', 'unknown')})",
            f"  MACD: {ta_data.get('macd_signal', 'unknown')} (histogram: {_float_or_zero(ta_data.get('macd_histogram')):+.3f})",
            f"  Bollinger: {ta_data.get('bb_position', 'unknown')} {'(SQUEEZE)' if bool(ta_data.get('bb_squeeze')) else ''}".rstrip(),
            f"  Trend: {ta_data.get('trend', 'unknown')}",
            f"  ATR: {_float_or_zero(ta_data.get('atr_pct')):.1f}% ({ta_data.get('volatility', 'unknown')} volatility)",
            f"  Composite signal: {ta_data.get('overall_signal', 'unknown')} ({_float_or_zero(ta_data.get('confidence')):.0%})",
        ]
    )
