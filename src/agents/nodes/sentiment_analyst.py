"""
Interprets FinBERT sentiment scores across collected headlines.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """You are a financial sentiment analyst.
You receive news headlines with FinBERT sentiment scores (range -1 to +1).

Produce a concise sentiment analysis in JSON:
{
  "summary": "2-3 sentence sentiment overview",
  "signals": ["signal 1", "signal 2", "signal 3"],
  "dominant_narrative": "The theme driving sentiment today",
  "contrarian_signal": "Any sentiment that contradicts price action, or null"
}

Focus on overall sentiment balance, extreme scores (>0.7 or <-0.7), and whether
sentiment aligns with or diverges from price action. Do not give financial advice."""


async def sentiment_analyst_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Analyze headline-level sentiment and narrative themes."""
    news = _dict_list(state.get("news_items"))
    current_count = _int_or_zero(state.get("llm_call_count"))

    if not news:
        return {
            "sentiment_report": {
                "summary": "No news data available.",
                "signals": [],
                "dominant_narrative": None,
                "contrarian_signal": None,
            },
            "llm_call_count": current_count,
        }

    headlines_text = "\n".join(
        (
            f"  [{item.get('sentiment_label', 'unknown')} "
            f"{_float_or_zero(item.get('sentiment_score')):+.2f}] "
            f"({', '.join(_string_list(item.get('related_symbols')))}) "
            f"{str(item.get('headline', ''))}"
        )
        for item in news
    )

    user_prompt = (
        "Today's news headlines with FinBERT sentiment scores:\n\n"
        f"{headlines_text}\n\n"
        "Analyze the overall sentiment landscape and identify key themes."
    )
    result = await llm.generate_json(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    return {
        "sentiment_report": result,
        "llm_call_count": current_count + 1,
    }


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _float_or_zero(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int_or_zero(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
