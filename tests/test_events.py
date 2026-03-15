from __future__ import annotations

import pytest

from src.briefing.formatter import _format_event_annotation
from src.data.base import NewsItem
from src.events.calibration import compute_remaining_drift, get_car_profile
from src.events.classifier import EventClassifier
from src.events.models import EventType


class FakeEventLLM:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> object:
        del system_prompt, user_prompt, temperature, max_tokens
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return []


def _news_item(
    *,
    headline: str,
    event_type: str,
    event_confidence: float,
    expected_car_pct: float,
    expected_car_window: str,
    remaining_drift_pct: float | None,
) -> NewsItem:
    return NewsItem(
        headline=headline,
        summary="",
        source="Test",
        url="https://example.com",
        published_at="2026-03-05T06:00:00+00:00",
        related_symbols=["NVDA"],
        sentiment_label=None,
        sentiment_score=None,
        sentiment_confidence=None,
        event_type=event_type,
        event_confidence=event_confidence,
        expected_car_pct=expected_car_pct,
        expected_car_window=expected_car_window,
        remaining_drift_pct=remaining_drift_pct,
    )


def test_car_table_has_profiles_for_all_event_types() -> None:
    for event_type in EventType:
        profile = get_car_profile(event_type)
        assert profile.event_window
        assert profile.direction in {"positive", "negative", "neutral"}


def test_compute_remaining_drift_known_case() -> None:
    remaining, interpretation = compute_remaining_drift(EventType.EARNINGS_BEAT, 0.8)

    assert remaining == pytest.approx(1.0)
    assert "remaining" in interpretation.lower()
    assert "+1.0%" in interpretation


@pytest.mark.asyncio
async def test_keyword_classifier_matches_sample_headlines(db) -> None:
    classifier = EventClassifier(db=db)
    events = await classifier.classify_headlines(
        [
            {
                "headline": "Microsoft raises full-year guidance on cloud demand",
                "symbols": ["MSFT"],
                "actual_move_pct": 1.1,
            },
            {
                "headline": "Broker downgrades Tesla and cuts price target",
                "symbols": ["TSLA"],
                "actual_move_pct": -2.1,
            },
            {
                "headline": "Token listed on Binance after exchange announcement",
                "symbols": ["TAO"],
                "actual_move_pct": 5.8,
            },
        ]
    )

    assert [event.event_type for event in events] == [
        EventType.GUIDANCE_RAISE,
        EventType.ANALYST_DOWNGRADE,
        EventType.EXCHANGE_LISTING,
    ]
    assert await db.get_event_cache_count() == 3


def test_format_event_annotation_renders_tag_and_drift_line() -> None:
    news_item = _news_item(
        headline="NVIDIA raises guidance",
        event_type="guidance_raise",
        event_confidence=0.82,
        expected_car_pct=2.0,
        expected_car_window="[-1, +3] days",
        remaining_drift_pct=1.4,
    )

    annotation = _format_event_annotation(news_item)

    assert "[guidance_raise]" in annotation
    assert "Hist. avg: +2.0% over [-1, +3] days" in annotation
    assert "remaining 1.4% upside expected" in annotation


@pytest.mark.asyncio
async def test_llm_classifier_uses_mocked_batch_response_and_cache(db) -> None:
    llm = FakeEventLLM(
        [[{"index": 0, "event_type": "hack_exploit", "confidence": 0.91}]]
    )
    classifier = EventClassifier(db=db, gemini_client=llm)
    headlines = [
        {
            "headline": "Protocol drained after exploit as attackers steal funds",
            "symbols": ["SOL"],
            "actual_move_pct": -4.2,
        }
    ]

    first = await classifier.classify_headlines(headlines)
    second = await classifier.classify_headlines(headlines)

    assert first[0].event_type == EventType.HACK_EXPLOIT
    assert second[0].event_type == EventType.HACK_EXPLOIT
    assert llm.calls == 1
    assert await db.get_event_cache_count() == 1
