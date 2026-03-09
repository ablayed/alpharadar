from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable

import aiosqlite

from src.sentiment.analyzer import SentimentAnalyzer, SentimentResult
from src.storage.database import Database

logger = logging.getLogger(__name__)


class SentimentCache:
    """SQLite-backed cache for normalized headline sentiment results."""

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def normalize_headline(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def hash_headline(cls, text: str) -> str:
        normalized = cls.normalize_headline(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def get_cached(self, headlines: list[str]) -> dict[str, SentimentResult | None]:
        cached: dict[str, SentimentResult | None] = {headline: None for headline in headlines}
        if not headlines:
            return cached

        hash_to_headlines: dict[str, list[str]] = {}
        for headline in headlines:
            hash_to_headlines.setdefault(self.hash_headline(headline), []).append(headline)

        async with aiosqlite.connect(self.db.db_path) as conn:
            await self.db._configure_connection(conn)
            conn.row_factory = aiosqlite.Row

            for chunk in _chunked(hash_to_headlines.keys(), size=200):
                placeholders = ",".join("?" for _ in chunk)
                cursor = await conn.execute(
                    f"""
                    SELECT headline_hash, label, score, confidence, positive, negative, neutral
                    FROM sentiment_cache
                    WHERE headline_hash IN ({placeholders})
                    """,
                    tuple(chunk),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    result = SentimentResult(
                        label=str(row["label"]),
                        score=float(row["score"]),
                        confidence=float(row["confidence"]),
                        positive=float(row["positive"]),
                        negative=float(row["negative"]),
                        neutral=float(row["neutral"]),
                    )
                    for headline in hash_to_headlines.get(str(row["headline_hash"]), []):
                        cached[headline] = result

        return cached

    async def store_results(self, headlines: list[str], results: list[SentimentResult]) -> None:
        if len(headlines) != len(results):
            raise ValueError("Headlines and results must have the same length")
        if not headlines:
            return

        rows = [
            (
                self.hash_headline(headline),
                headline,
                result.label,
                result.score,
                result.confidence,
                result.positive,
                result.negative,
                result.neutral,
                SentimentAnalyzer.MODEL_VERSION,
            )
            for headline, result in zip(headlines, results, strict=True)
        ]

        async with self.db._write_lock:
            async with aiosqlite.connect(self.db.db_path) as conn:
                await self.db._configure_connection(conn)
                await conn.executemany(
                    """
                    INSERT INTO sentiment_cache (
                        headline_hash,
                        headline_text,
                        label,
                        score,
                        confidence,
                        positive,
                        negative,
                        neutral,
                        analyzed_at,
                        model_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
                    ON CONFLICT(headline_hash)
                    DO UPDATE SET
                        headline_text = excluded.headline_text,
                        label = excluded.label,
                        score = excluded.score,
                        confidence = excluded.confidence,
                        positive = excluded.positive,
                        negative = excluded.negative,
                        neutral = excluded.neutral,
                        analyzed_at = datetime('now'),
                        model_version = excluded.model_version
                    """,
                    rows,
                )
                await conn.commit()
        logger.info("Stored %s sentiment result(s) in cache", len(rows))


def _chunked(values: Iterable[str], size: int) -> list[tuple[str, ...]]:
    chunk: list[str] = []
    chunks: list[tuple[str, ...]] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            chunks.append(tuple(chunk))
            chunk = []
    if chunk:
        chunks.append(tuple(chunk))
    return chunks
