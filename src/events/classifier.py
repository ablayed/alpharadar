"""
Event classification with Gemini-first inference and SQLite-backed caching.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Protocol

import aiosqlite

from src.events.calibration import compute_remaining_drift, get_car_profile
from src.events.models import DetectedEvent, EventType
from src.storage.database import Database

logger = logging.getLogger(__name__)

MODEL_VERSION = "gemini-2.5-flash-event-v1"
FALLBACK_MODEL_VERSION = "keyword-event-v1"
EVENT_TYPES_STR = "\n".join(
    f"  - {event_type.value}"
    for event_type in EventType
    if event_type not in {EventType.UNKNOWN}
)
SYSTEM_PROMPT = f"""You are a financial event classifier.
Classify each headline into exactly one event type from this list:

{EVENT_TYPES_STR}

If none fit clearly, use "general_news".

Respond only with a JSON array. Each item must be:
{{"index": 0, "event_type": "earnings_beat", "confidence": 0.85}}

Rules:
- Preserve headline order using the input index.
- confidence must be between 0.0 and 1.0.
- Use only the listed event types.
- Classify from headline content only. Do not speculate beyond the text.
"""


class _JSONGenerator(Protocol):
    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> object: ...


class EventClassifier:
    """Batch classifier with SQLite caching and keyword fallback."""

    def __init__(self, db: Database, gemini_client: _JSONGenerator | None = None) -> None:
        self._db = db
        self._llm = gemini_client

    @staticmethod
    def normalize_headline(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def hash_headline(cls, text: str) -> str:
        normalized = cls.normalize_headline(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def classify_headlines(
        self,
        headlines: list[dict[str, object]],
    ) -> list[DetectedEvent]:
        """Classify a batch of headlines and attach CAR calibration."""

        if not headlines:
            return []

        cached = await self._get_cached([_headline_text(item) for item in headlines])
        uncached_texts = self._unique_uncached_headlines(headlines, cached)
        fresh: dict[str, tuple[EventType, float]] = {}

        if uncached_texts:
            if self._llm is not None:
                try:
                    fresh = await self._classify_batch_with_llm(uncached_texts)
                    await self._store_cached(fresh, MODEL_VERSION)
                    logger.info("Event classification: %s uncached headlines via Gemini", len(fresh))
                except Exception:
                    logger.exception("Gemini event classification failed; using keyword fallback")
                    fresh = self._classify_batch_with_keywords(uncached_texts)
                    await self._store_cached(fresh, FALLBACK_MODEL_VERSION)
            else:
                fresh = self._classify_batch_with_keywords(uncached_texts)
                await self._store_cached(fresh, FALLBACK_MODEL_VERSION)
                logger.info("Event classification: %s uncached headlines via keyword fallback", len(fresh))

        events: list[DetectedEvent] = []
        for item in headlines:
            headline = _headline_text(item)
            cached_result = cached.get(headline) or fresh.get(headline)
            event_type, confidence = cached_result or (EventType.GENERAL_NEWS, 0.3)
            events.append(self._build_detected_event(item, event_type, confidence))

        return events

    async def _classify_batch_with_llm(
        self,
        headlines: list[str],
    ) -> dict[str, tuple[EventType, float]]:
        """Classify only uncached unique headlines with Gemini."""

        llm = self._llm
        if llm is None:
            raise RuntimeError("Gemini client is not configured")

        numbered = "\n".join(f"[{index}] {headline}" for index, headline in enumerate(headlines))
        user_prompt = f"Classify these headlines:\n\n{numbered}"
        result = await llm.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1024,
        )

        classifications = _extract_classifications(result)
        if not classifications:
            raise ValueError("Gemini returned no usable event classifications")
        mapped: dict[str, tuple[EventType, float]] = {}
        for index, headline in enumerate(headlines):
            match = next(
                (
                    item
                    for item in classifications
                    if isinstance(item, Mapping) and _int_or_none(item.get("index")) == index
                ),
                None,
            )
            if match is None:
                mapped[headline] = (EventType.GENERAL_NEWS, 0.3)
                continue
            mapped[headline] = (
                _parse_event_type(match.get("event_type")),
                _clamp_confidence(match.get("confidence")),
            )
        return mapped

    def _classify_batch_with_keywords(
        self,
        headlines: list[str],
    ) -> dict[str, tuple[EventType, float]]:
        return {
            headline: self._classify_keyword_headline(headline)
            for headline in headlines
        }

    def _classify_keyword_headline(self, headline: str) -> tuple[EventType, float]:
        event_type = self._keyword_match(headline.lower())
        confidence = 0.62 if event_type not in {EventType.GENERAL_NEWS, EventType.UNKNOWN} else 0.3
        return event_type, confidence

    def _build_detected_event(
        self,
        headline_data: Mapping[str, object],
        event_type: EventType,
        confidence: float,
    ) -> DetectedEvent:
        actual_move = _float_or_zero(headline_data.get("actual_move_pct"))
        car_profile = get_car_profile(event_type)
        remaining_drift, _ = compute_remaining_drift(event_type, actual_move)
        return DetectedEvent(
            event_type=event_type,
            confidence=confidence,
            headline=_headline_text(headline_data),
            related_symbols=_string_list(headline_data.get("symbols")),
            expected_car_pct=car_profile.avg_car_pct,
            expected_car_window=car_profile.event_window,
            car_direction=car_profile.direction,
            car_magnitude=car_profile.magnitude,
            actual_move_pct=actual_move,
            remaining_drift_pct=remaining_drift,
        )

    async def _get_cached(
        self,
        headlines: list[str],
    ) -> dict[str, tuple[EventType, float] | None]:
        cached: dict[str, tuple[EventType, float] | None] = {headline: None for headline in headlines}
        if not headlines:
            return cached

        hash_to_headlines: dict[str, list[str]] = {}
        for headline in headlines:
            hash_to_headlines.setdefault(self.hash_headline(headline), []).append(headline)

        async with aiosqlite.connect(self._db.db_path) as conn:
            await self._db._configure_connection(conn)
            conn.row_factory = aiosqlite.Row

            for chunk in _chunked(hash_to_headlines.keys(), size=200):
                placeholders = ",".join("?" for _ in chunk)
                cursor = await conn.execute(
                    f"""
                    SELECT headline_hash, event_type, confidence
                    FROM event_cache
                    WHERE headline_hash IN ({placeholders})
                    """,
                    tuple(chunk),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    event_type = _parse_event_type(row["event_type"])
                    confidence = _clamp_confidence(row["confidence"])
                    for headline in hash_to_headlines.get(str(row["headline_hash"]), []):
                        cached[headline] = (event_type, confidence)

        return cached

    async def _store_cached(
        self,
        results: dict[str, tuple[EventType, float]],
        model_version: str,
    ) -> None:
        if not results:
            return

        rows = [
            (
                self.hash_headline(headline),
                headline,
                event_type.value,
                confidence,
                model_version,
            )
            for headline, (event_type, confidence) in results.items()
        ]

        async with self._db._write_lock:
            async with aiosqlite.connect(self._db.db_path) as conn:
                await self._db._configure_connection(conn)
                await conn.executemany(
                    """
                    INSERT INTO event_cache (
                        headline_hash,
                        headline_text,
                        event_type,
                        confidence,
                        classified_at,
                        model_version
                    )
                    VALUES (?, ?, ?, ?, datetime('now'), ?)
                    ON CONFLICT(headline_hash)
                    DO UPDATE SET
                        headline_text = excluded.headline_text,
                        event_type = excluded.event_type,
                        confidence = excluded.confidence,
                        classified_at = datetime('now'),
                        model_version = excluded.model_version
                    """,
                    rows,
                )
                await conn.commit()

    @staticmethod
    def _unique_uncached_headlines(
        headlines: list[dict[str, object]],
        cached: Mapping[str, tuple[EventType, float] | None],
    ) -> list[str]:
        uncached: list[str] = []
        seen_hashes: set[str] = set()
        for item in headlines:
            headline = _headline_text(item)
            if cached.get(headline) is not None:
                continue
            headline_hash = EventClassifier.hash_headline(headline)
            if headline_hash in seen_hashes:
                continue
            seen_hashes.add(headline_hash)
            uncached.append(headline)
        return uncached

    @staticmethod
    def _keyword_match(text: str) -> EventType:
        if _contains_any(text, "acqui", "merger", "buyout", "takeover"):
            return EventType.ACQUISITION
        if _contains_any(text, "beats", "beat", "topped", "tops", "surpass", "exceeded") and _contains_any(
            text,
            "earnings",
            "revenue",
            "eps",
            "quarter",
        ):
            return EventType.EARNINGS_BEAT
        if _contains_any(text, "miss", "missed", "fell short", "below estimate", "disappoint") and _contains_any(
            text,
            "earnings",
            "revenue",
            "eps",
            "quarter",
        ):
            return EventType.EARNINGS_MISS
        if _contains_any(text, "guidance", "outlook", "forecast") and _contains_any(
            text,
            "raise",
            "raises",
            "raised",
            "boost",
            "increas",
            "lifts",
            "lifted",
        ):
            return EventType.GUIDANCE_RAISE
        if _contains_any(text, "guidance", "outlook", "forecast") and _contains_any(
            text,
            "cut",
            "cuts",
            "lower",
            "reduces",
            "trim",
            "slashes",
        ):
            return EventType.GUIDANCE_CUT
        if _contains_any(text, "contract", "awarded", "agreement", "deal", "order win"):
            return EventType.NEW_CONTRACT
        if _contains_any(text, "launch", "unveil", "release", "introduc", "new product"):
            return EventType.NEW_PRODUCT
        if _contains_any(text, "buyback", "repurchase", "share repurchase"):
            return EventType.STOCK_BUYBACK
        if _contains_any(text, "dividend", "payout"):
            return EventType.DIVIDEND_CHANGE
        if _contains_any(text, "ceo", "cfo", "cto", "steps down", "resigns", "appointed", "appoints"):
            return EventType.EXECUTIVE_CHANGE
        if _contains_any(text, "fda", "phase 2", "phase 3", "trial result", "trial data", "approval"):
            return EventType.CLINICAL_TRIAL
        if _contains_any(text, "lawsuit", "sued", "litigation", "settlement", "legal action"):
            return EventType.LEGAL_ACTION
        if _contains_any(text, "probe", "investigat", "antitrust", "sec", "regulator", "fine") and _contains_any(
            text,
            "company",
            "shares",
            "stock",
            "chip",
            "firm",
            "bank",
            "maker",
        ):
            return EventType.REGULATORY_ACTION
        if _contains_any(text, "partner", "collaborat", "joint venture", "alliance"):
            return EventType.PARTNERSHIP
        if _contains_any(text, "layoff", "restructur", "job cut", "workforce reduction"):
            return EventType.LAYOFFS
        if _contains_any(text, "upgrade", "outperform", "overweight", "price target raise", "buy rating"):
            return EventType.ANALYST_UPGRADE
        if _contains_any(text, "downgrade", "underperform", "underweight", "price target cut", "sell rating"):
            return EventType.ANALYST_DOWNGRADE
        if _contains_any(text, "listed on", "listing", "exchange debut", "coinbase", "binance") and _contains_any(
            text,
            "token",
            "coin",
            "crypto",
            "pair",
        ):
            return EventType.EXCHANGE_LISTING
        if _contains_any(text, "mainnet", "hard fork", "soft fork", "v2", "v3", "protocol upgrade"):
            return EventType.PROTOCOL_UPGRADE
        if _contains_any(text, "hack", "exploit", "breach", "stolen", "drained"):
            return EventType.HACK_EXPLOIT
        if _contains_any(text, "unlock", "vesting", "token release"):
            return EventType.TOKEN_UNLOCK
        if _contains_any(text, "whale", "large transfer", "moved to exchange"):
            return EventType.WHALE_MOVEMENT
        if _contains_any(text, "etf", "crypto rule", "token ban", "exchange crackdown", "sec") and _contains_any(
            text,
            "bitcoin",
            "ethereum",
            "crypto",
            "token",
            "exchange",
        ):
            return EventType.REGULATORY_CRYPTO
        if _contains_any(text, "tvl", "total value locked", "defi inflow", "defi outflow"):
            return EventType.DEFI_TVL_SHIFT
        if _contains_any(text, "fomc", "fed", "powell", "rate decision"):
            return EventType.FED_RATE_DECISION
        if _contains_any(text, "cpi", "inflation", "pce", "consumer price"):
            return EventType.INFLATION_DATA
        if _contains_any(text, "tariff", "trade war", "sanction", "embargo", "export ban"):
            return EventType.TRADE_POLICY
        if _contains_any(text, "war", "conflict", "tension", "military", "invasion"):
            return EventType.GEOPOLITICAL
        return EventType.GENERAL_NEWS


def _extract_classifications(result: object) -> list[Mapping[str, object]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, Mapping)]
    if isinstance(result, Mapping):
        raw_items = result.get("classifications", [])
        if isinstance(raw_items, list):
            return [item for item in raw_items if isinstance(item, Mapping)]
    return []


def _parse_event_type(value: object) -> EventType:
    if isinstance(value, EventType):
        return value
    if isinstance(value, str):
        try:
            return EventType(value)
        except ValueError:
            return EventType.GENERAL_NEWS
    return EventType.GENERAL_NEWS


def _headline_text(value: Mapping[str, object]) -> str:
    return str(value.get("headline", "")).strip()


def _float_or_zero(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _clamp_confidence(value: object) -> float:
    if isinstance(value, int | float):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            numeric = 0.3
    else:
        numeric = 0.3
    return max(0.0, min(1.0, numeric))


def _contains_any(text: str, *keywords: str) -> bool:
    return any(keyword in text for keyword in keywords)


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
