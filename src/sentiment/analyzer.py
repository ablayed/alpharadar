from __future__ import annotations

import asyncio
import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SentimentResult:
    label: str
    score: float
    confidence: float
    positive: float
    negative: float
    neutral: float


class SentimentAnalyzer:
    """
    Lazy-loading FinBERT sentiment analysis via ONNX Runtime.

    The model is stored on disk and loaded only for inference batches so the
    bot stays within the VM's tight memory budget between briefings.
    """

    MODEL_ID = "ProsusAI/finbert"
    MODEL_FILENAME = "model.onnx"
    MODEL_VERSION = "finbert-onnx-v1"

    def __init__(self, model_path: Path | None = None, batch_size: int = 16) -> None:
        self.model_path = model_path or self.default_model_path()
        self.batch_size = batch_size
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded = False

    @classmethod
    def default_model_path(cls) -> Path:
        override = os.getenv("FINBERT_MODEL_PATH")
        if override:
            return Path(override)
        return Path(__file__).resolve().parents[2] / "models" / "finbert-onnx"

    @property
    def model_file(self) -> Path:
        return self.model_path / self.MODEL_FILENAME

    def is_available(self) -> bool:
        return self.model_file.exists()

    def is_loaded(self) -> bool:
        return self._loaded

    async def initialize(self) -> None:
        if self.is_available():
            logger.info("FinBERT ONNX model found at %s", self.model_path)
            return

        logger.info("FinBERT ONNX model not found. Downloading and exporting...")
        logger.info("This may take a few minutes the first time it runs.")
        self.model_path.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._export_model)
        logger.info("FinBERT ONNX model saved to %s", self.model_path)

    def close(self) -> None:
        self._unload_model()

    def _export_model(self) -> None:
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT export dependencies are unavailable. Bake the ONNX model into the Docker image "
                "or install the export dependencies in the build environment."
            ) from exc

        model = ORTModelForSequenceClassification.from_pretrained(self.MODEL_ID, export=True)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        try:
            model.save_pretrained(self.model_path)
            tokenizer.save_pretrained(self.model_path)
        finally:
            del model
            del tokenizer
            gc.collect()

    def _load_model(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise FileNotFoundError(
                f"FinBERT ONNX model not found at {self.model_file}. "
                "Build the Docker image with the baked model or run initialize() first."
            )

        logger.info("Loading FinBERT ONNX model into memory...")
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._model = ORTModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            provider="CPUExecutionProvider",
        )
        self._loaded = True
        logger.info("FinBERT model loaded successfully")

    def _unload_model(self) -> None:
        if not self._loaded:
            return

        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        logger.info("FinBERT model unloaded from memory")

    async def analyze(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        return await asyncio.to_thread(self._analyze_sync, texts)

    def _analyze_sync(self, texts: list[str]) -> list[SentimentResult]:
        import numpy as np

        self._load_model()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("FinBERT model failed to initialize")

        labels = ("positive", "negative", "neutral")
        results: list[SentimentResult] = []

        try:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="np",
                )
                outputs = self._model(**inputs)
                logits = np.asarray(outputs.logits, dtype=np.float32)
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                for row in probs:
                    positive = float(row[0])
                    negative = float(row[1])
                    neutral = float(row[2])
                    max_idx = int(np.argmax(row))
                    results.append(
                        SentimentResult(
                            label=labels[max_idx],
                            score=round(positive - negative, 4),
                            confidence=round(float(row[max_idx]), 4),
                            positive=round(positive, 4),
                            negative=round(negative, 4),
                            neutral=round(neutral, 4),
                        )
                    )
            return results
        finally:
            self._unload_model()
