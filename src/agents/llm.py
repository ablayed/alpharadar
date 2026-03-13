"""
Thin async wrapper around the Google Gemini API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, ClassVar

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Async Gemini client with light rate limiting for the daily agent pipeline.
    """

    MODEL = "gemini-2.5-flash-lite"
    _limiter: ClassVar[AsyncLimiter] = AsyncLimiter(max_rate=12, time_period=60)

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Any | None = None
        self._call_count = 0

    async def initialize(self) -> None:
        """Create the underlying google-genai client."""
        from google import genai  # type: ignore[import-untyped]

        self._client = genai.Client(api_key=self._api_key)
        logger.info("Gemini client initialized (model=%s)", self.MODEL)

    @property
    def call_count(self) -> int:
        return self._call_count

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a text response from Gemini.
        """
        if self._client is None:
            raise RuntimeError("GeminiClient is not initialized")

        async with self._limiter:
            response_text = await asyncio.to_thread(
                self._generate_sync,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._call_count += 1
            return response_text

    def _generate_sync(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        from google.genai import types  # type: ignore[import-untyped]

        if self._client is None:
            raise RuntimeError("GeminiClient is not initialized")

        response = self._client.models.generate_content(
            model=self.MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict[str, object]:
        """
        Ask Gemini for JSON and parse the result conservatively.
        """
        text = await self.generate(
            system_prompt=(
                f"{system_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. "
                "Do not use markdown or code fences."
            ),
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini JSON response: %s", text[:200])
            return {"raw": text}

        if isinstance(parsed, dict):
            return parsed

        logger.warning("Gemini returned non-object JSON: %s", type(parsed).__name__)
        return {"raw": text}
