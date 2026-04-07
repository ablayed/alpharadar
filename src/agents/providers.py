"""
Multi-provider LLM client with automatic fallback.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum

import httpx
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

GROQ_QWEN_MODEL = "qwen/qwen3-32b"
GROQ_LLAMA_MODEL = "llama-3.3-70b-versatile"
GEMINI_PRO_MODEL = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
GROQ_REQUESTS_PER_MINUTE = 25
GROQ_MIN_INTERVAL_SECONDS = 2.2
GEMINI_REQUESTS_PER_MINUTE = 4


class LLMRole(Enum):
    """Which pipeline role is making the call."""

    BULL = "bull"
    BEAR = "bear"
    JUDGE = "judge"
    PLAYBOOK = "playbook"
    REFLECTION = "reflection"
    GENERAL = "general"


@dataclass(slots=True, frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_used: int
    role: LLMRole


def _response_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200] or "no response body"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return response.text[:200] or "no response body"


class MultiProviderLLM:
    """
    Route LLM calls to the best provider for each role and fall back on failure.
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        gemini_api_key: str | None = None,
    ) -> None:
        self._groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self._gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._groq_available = bool(self._groq_key)
        self._gemini_available = bool(self._gemini_key)
        self._groq_limiter = AsyncLimiter(max_rate=GROQ_REQUESTS_PER_MINUTE, time_period=60)
        self._groq_request_lock = asyncio.Lock()
        self._groq_next_request_at = 0.0
        self._gemini_limiter = AsyncLimiter(max_rate=GEMINI_REQUESTS_PER_MINUTE, time_period=60)

        if self._gemini_available:
            logger.info("Gemini configured for judge/reflection routing")
        else:
            logger.warning("No GEMINI_API_KEY configured; Gemini routes will use Groq fallback")

        if self._groq_available:
            logger.info("Groq configured for primary general/bull/bear/playbook routing")
        else:
            logger.warning("No GROQ_API_KEY configured; Groq routes are unavailable")

    def has_available_provider(self) -> bool:
        return self._groq_available or self._gemini_available

    def provider_status(self) -> dict[str, dict[str, object]]:
        return {
            "groq": {
                "configured": self._groq_available,
                "primary_model": GROQ_QWEN_MODEL,
                "fallback_model": GROQ_LLAMA_MODEL,
            },
            "gemini": {
                "configured": self._gemini_available,
                "primary_model": GEMINI_PRO_MODEL,
                "fallback_model": GEMINI_FLASH_MODEL,
            },
        }

    async def complete(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        """
        Route an LLM call to the best provider for the given role.
        """
        if not self.has_available_provider():
            raise RuntimeError("No LLM provider is configured. Set GROQ_API_KEY or GEMINI_API_KEY.")

        if role in {LLMRole.JUDGE, LLMRole.REFLECTION}:
            chain = [
                ("gemini", GEMINI_PRO_MODEL),
                ("groq", GROQ_QWEN_MODEL),
                ("groq", GROQ_LLAMA_MODEL),
            ]
        elif role == LLMRole.PLAYBOOK:
            chain = [
                ("groq", GROQ_LLAMA_MODEL),
                ("gemini", GEMINI_FLASH_MODEL),
                ("groq", GROQ_QWEN_MODEL),
            ]
        else:
            chain = [
                ("groq", GROQ_QWEN_MODEL),
                ("groq", GROQ_LLAMA_MODEL),
                ("gemini", GEMINI_FLASH_MODEL),
            ]

        last_error: Exception | None = None
        for provider, model in chain:
            if provider == "gemini" and not self._gemini_available:
                continue
            if provider == "groq" and not self._groq_available:
                continue

            try:
                if provider == "gemini":
                    return await self._call_gemini(
                        role=role,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                return await self._call_groq(
                    role=role,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM call failed: provider=%s model=%s role=%s error=%s",
                    provider,
                    model,
                    role.value,
                    str(exc)[:160],
                )

        raise RuntimeError(
            f"All LLM providers failed for role={role.value}. Last error: {last_error}"
        )

    async def _call_groq(
        self,
        *,
        role: LLMRole,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model == GROQ_QWEN_MODEL:
            payload["reasoning_effort"] = "default"
            payload["reasoning_format"] = "hidden"

        async with self._groq_limiter:
            await self._wait_for_groq_slot()
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._groq_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    detail = _response_error_detail(exc.response)
                    if status_code == 401:
                        raise RuntimeError("Groq rejected the API key. Check GROQ_API_KEY.") from exc
                    if status_code == 429:
                        raise RuntimeError("Groq rate limit or quota hit. Retry in a minute.") from exc
                    raise RuntimeError(f"Groq request failed ({status_code}): {detail}") from exc

        data = response.json()
        message = data["choices"][0]["message"]["content"]
        return LLMResponse(
            text=message if isinstance(message, str) else str(message),
            model=model,
            provider="groq",
            tokens_used=int(data.get("usage", {}).get("total_tokens", 0) or 0),
            role=role,
        )

    async def _wait_for_groq_slot(self) -> None:
        async with self._groq_request_lock:
            now = time.monotonic()
            wait_seconds = self._groq_next_request_at - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._groq_next_request_at = time.monotonic() + GROQ_MIN_INTERVAL_SECONDS

    async def _call_gemini(
        self,
        *,
        role: LLMRole,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        async with self._gemini_limiter:
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={
                            "x-goog-api-key": str(self._gemini_key),
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    detail = _response_error_detail(exc.response)
                    if status_code == 401:
                        raise RuntimeError("Gemini rejected the API key. Check GEMINI_API_KEY.") from exc
                    if status_code == 429:
                        raise RuntimeError("Gemini rate limit or quota hit. Retry in a minute.") from exc
                    raise RuntimeError(f"Gemini request failed ({status_code}): {detail}") from exc

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")

        return LLMResponse(
            text=text,
            model=model,
            provider="gemini",
            tokens_used=int(data.get("usageMetadata", {}).get("totalTokenCount", 0) or 0),
            role=role,
        )


_provider: MultiProviderLLM | None = None


def get_provider() -> MultiProviderLLM:
    global _provider
    if _provider is None:
        _provider = MultiProviderLLM()
    return _provider


def init_provider(groq_key: str | None = None, gemini_key: str | None = None) -> None:
    global _provider
    _provider = MultiProviderLLM(groq_key, gemini_key)
