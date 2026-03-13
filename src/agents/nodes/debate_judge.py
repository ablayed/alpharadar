"""
Synthesizes bull and bear arguments into a balanced verdict.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """You are an impartial financial debate judge.
You receive the bull and bear cases for an asset and must deliver a balanced
verdict that helps the reader form their own view.

You are not giving advice or simply choosing a winner.

Respond in JSON:
{
  "prevailing_view": "bullish | bearish | neutral",
  "confidence": 0.0,
  "summary": "2-3 sentence balanced verdict",
  "bull_in_brief": "one sentence summary of the bull case",
  "bear_in_brief": "one sentence summary of the bear case",
  "key_risk": "single most important risk to monitor",
  "what_to_watch": "one specific thing to watch over the next 1-5 days"
}"""


async def debate_judge_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Judge the debate and return one verdict per mover."""
    movers = _dict_list(state.get("selected_movers"))
    bull_args = _dict_list(state.get("bull_arguments"))
    bear_args = _dict_list(state.get("bear_arguments"))

    verdicts: list[dict[str, object]] = []
    call_count = _int_or_zero(state.get("llm_call_count"))

    for index, mover in enumerate(movers):
        symbol = str(mover.get("symbol", "?"))
        bull = bull_args[index] if index < len(bull_args) else {}
        bear = bear_args[index] if index < len(bear_args) else {}

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

Deliver a balanced verdict for {symbol}."""

        result = await llm.generate_json(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        result["symbol"] = symbol
        result["display_name"] = symbol
        verdicts.append(result)
        call_count += 1

    return {"debate_verdicts": verdicts, "llm_call_count": call_count}


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _int_or_zero(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
