"""
FinBERT sentiment analysis using ONNX Runtime directly.
No dependency on optimum or torch at inference time.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    MODEL_PATH = "/app/models/finbert-onnx"
    MODEL_FILENAME = "model.onnx"
    MODEL_VERSION = "finbert-onnx-v1"
    LABELS = ["positive", "negative", "neutral"]

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or self.default_model_path()
        self._session = None
        self._tokenizer = None
        self._loaded = False

    @classmethod
    def default_model_path(cls) -> Path:
        override = os.getenv("FINBERT_MODEL_PATH")
        if override:
            return Path(override)

        app_model_path = Path(cls.MODEL_PATH)
        if app_model_path.exists():
            return app_model_path

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

        raise FileNotFoundError(
            f"FinBERT ONNX model not found at {self.model_file}. "
            "Export it on your laptop and upload to models/finbert-onnx/"
        )

    def close(self) -> None:
        self._unload_model()

    def _load_model(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise FileNotFoundError(
                f"FinBERT ONNX model not found at {self.model_file}. "
                "Export it on your laptop and upload to models/finbert-onnx/"
            )

        logger.info("Loading FinBERT ONNX model into memory...")

        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(self.model_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self._loaded = True
        logger.info("FinBERT model loaded successfully")

    def _unload_model(self) -> None:
        if not self._loaded:
            return

        self._session = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        logger.info("FinBERT model unloaded from memory")

    async def analyze(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        return await asyncio.to_thread(self._analyze_sync, texts)

    def _analyze_sync(self, texts: list[str]) -> list[SentimentResult]:
        self._load_model()
        if self._session is None or self._tokenizer is None:
            raise RuntimeError("FinBERT model failed to initialize")

        try:
            results: list[SentimentResult] = []
            batch_size = 16

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="np",
                )

                ort_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }

                if "token_type_ids" in inputs:
                    ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

                outputs = self._session.run(None, ort_inputs)
                logits = np.asarray(outputs[0], dtype=np.float32)

                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                for j in range(len(batch)):
                    p_pos = float(probs[j][0])
                    p_neg = float(probs[j][1])
                    p_neu = float(probs[j][2])

                    max_idx = int(np.argmax(probs[j]))
                    label = self.LABELS[max_idx]
                    confidence = float(probs[j][max_idx])
                    score = p_pos - p_neg

                    results.append(
                        SentimentResult(
                            label=label,
                            score=round(score, 4),
                            confidence=round(confidence, 4),
                            positive=round(p_pos, 4),
                            negative=round(p_neg, 4),
                            neutral=round(p_neu, 4),
                        )
                    )

            return results

        finally:
            self._unload_model()
