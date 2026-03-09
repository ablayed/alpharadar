from __future__ import annotations

import importlib.util

import pytest

from src.sentiment.analyzer import SentimentAnalyzer

MODEL_FILE = SentimentAnalyzer.default_model_path() / SentimentAnalyzer.MODEL_FILENAME
DEPS_AVAILABLE = (
    importlib.util.find_spec("optimum") is not None
    and importlib.util.find_spec("transformers") is not None
)
MODEL_AVAILABLE = DEPS_AVAILABLE and MODEL_FILE.exists()


@pytest.mark.asyncio
async def test_analyze_empty_input_returns_empty_list() -> None:
    analyzer = SentimentAnalyzer()
    assert await analyzer.analyze([]) == []


@pytest.mark.asyncio
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="FinBERT ONNX model is not available locally")
async def test_known_headlines_match_expected_direction() -> None:
    analyzer = SentimentAnalyzer()
    results = await analyzer.analyze(
        [
            "NVIDIA reports record quarterly revenue of $35 billion",
            "Tesla recalls 500,000 vehicles due to safety concerns",
            "Apple scheduled to report earnings next Thursday",
        ]
    )

    assert results[0].label == "positive"
    assert results[0].score > 0.3
    assert results[1].label == "negative"
    assert results[1].score < -0.3
    assert abs(results[2].score) < 0.3


@pytest.mark.asyncio
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="FinBERT ONNX model is not available locally")
async def test_batch_processing_returns_valid_results() -> None:
    analyzer = SentimentAnalyzer()
    headlines = [
        "NVIDIA reports record quarterly revenue of $35 billion",
        "Tesla recalls 500,000 vehicles due to safety concerns",
        "Apple scheduled to report earnings next Thursday",
        "Microsoft expands cloud capacity in Europe",
    ] * 5

    results = await analyzer.analyze(headlines)

    assert len(results) == len(headlines)
    assert all(result.label in {"positive", "negative", "neutral"} for result in results)
    assert all(-1.0 <= result.score <= 1.0 for result in results)
