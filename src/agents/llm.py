"""
Thin async wrapper around the Groq chat completions API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, ClassVar

import httpx

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Async Groq client with light rate limiting for the daily agent pipeline.

    The class name stays as GeminiClient to avoid touching the wider import graph.
    """

    MODEL = "llama-3.3-70b-versatile"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    _limiter: ClassVar[AsyncLimiter] = AsyncLimiter(max_rate=25, time_period=60)

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._initialized = False
        self._call_count = 0

    async def initialize(self) -> None:
        """Validate the configured API key and mark the client ready for use."""
        if not self._api_key:
            raise RuntimeError("Groq API key is not configured")
        self._initialized = True
        logger.info("Groq client initialized (model=%s)", self.MODEL)

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
        Generate a text response from Groq.
        """
        if not self._initialized:
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
        if not self._initialized:
            raise RuntimeError("GeminiClient is not initialized")

        response = httpx.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        return message if isinstance(message, str) else str(message)

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Any:
        """
        Ask Groq for JSON and parse the result conservatively.
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
            logger.warning("Failed to parse Groq JSON response: %s", text[:200])
            return {"raw": text}

        if isinstance(parsed, dict | list):
            return parsed

        logger.warning("Groq returned unsupported JSON type: %s", type(parsed).__name__)
        return {"raw": text}
