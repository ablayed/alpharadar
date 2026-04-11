"""
Multi-provider LLM router for AlphaRadar.

Primary: OpenRouter (Gemini 2.5 Pro for Judge/Reflection, DeepSeek for all other roles)
Fallback: Groq (Qwen3-32B, Llama 3.3 70B)

All OpenRouter calls use the OpenAI-compatible chat/completions endpoint.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_JUDGE_MODEL = "google/gemini-2.5-pro-preview"
OPENROUTER_REASONING_MODEL = "deepseek/deepseek-r1-0528"
OPENROUTER_GENERAL_MODEL = "deepseek/deepseek-v3.2"
GROQ_QWEN_MODEL = "qwen/qwen3-32b"
GROQ_LLAMA_MODEL = "llama-3.3-70b-versatile"

GROQ_MIN_DELAY_SECONDS = 2.5
OPENROUTER_MIN_DELAY_SECONDS = 1.0

_last_call: dict[str, float] = {"groq": 0.0, "openrouter": 0.0}
_provider_locks: dict[str, asyncio.Lock] = {
    "groq": asyncio.Lock(),
    "openrouter": asyncio.Lock(),
}


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


ROLE_CHAINS: dict[LLMRole, list[tuple[str, str]]] = {
    LLMRole.JUDGE: [
        ("openrouter", OPENROUTER_JUDGE_MODEL),
        ("openrouter", OPENROUTER_REASONING_MODEL),
        ("groq", GROQ_LLAMA_MODEL),
    ],
    LLMRole.REFLECTION: [
        ("openrouter", OPENROUTER_JUDGE_MODEL),
        ("openrouter", OPENROUTER_REASONING_MODEL),
        ("groq", GROQ_LLAMA_MODEL),
    ],
    LLMRole.BULL: [
        ("openrouter", OPENROUTER_GENERAL_MODEL),
        ("groq", GROQ_QWEN_MODEL),
        ("groq", GROQ_LLAMA_MODEL),
    ],
    LLMRole.BEAR: [
        ("openrouter", OPENROUTER_GENERAL_MODEL),
        ("groq", GROQ_QWEN_MODEL),
        ("groq", GROQ_LLAMA_MODEL),
    ],
    LLMRole.PLAYBOOK: [
        ("openrouter", OPENROUTER_REASONING_MODEL),
        ("openrouter", OPENROUTER_GENERAL_MODEL),
        ("groq", GROQ_LLAMA_MODEL),
    ],
    LLMRole.GENERAL: [
        ("openrouter", OPENROUTER_GENERAL_MODEL),
        ("groq", GROQ_QWEN_MODEL),
        ("groq", GROQ_LLAMA_MODEL),
    ],
}


def _resolve_api_key(explicit_value: str | None, env_var: str) -> str | None:
    candidate = explicit_value if explicit_value is not None else os.getenv(env_var)
    if candidate is None:
        return None
    stripped = candidate.strip()
    return stripped or None


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


def _extract_message_text(data: dict[str, object]) -> str:
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response contained no choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("LLM response choice payload was malformed")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response message payload was malformed")

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def _strip_reasoning_traces(model: str, text: str) -> str:
    if "r1" not in model.lower() and "qwq" not in model.lower():
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _is_empty_response_text(text: str) -> bool:
    return not text or text.strip().lower() in {"", "none", "null"}


class MultiProviderLLM:
    """Route LLM calls to the best provider for each role and fall back on failure."""

    def __init__(
        self,
        groq_api_key: str | None = None,
        openrouter_api_key: str | None = None,
    ) -> None:
        self._groq_key = _resolve_api_key(groq_api_key, "GROQ_API_KEY")
        self._openrouter_key = _resolve_api_key(openrouter_api_key, "OPENROUTER_API_KEY")
        self._groq_available = bool(self._groq_key)
        self._openrouter_available = bool(self._openrouter_key)

        if self._openrouter_available:
            logger.info(
                "OpenRouter configured: Gemini 2.5 Pro for judge/reflection, "
                "DeepSeek R1/V3.2 for other roles"
            )
        else:
            logger.warning(
                "No OPENROUTER_API_KEY configured; OpenRouter routes will use Groq fallback"
            )

        if self._groq_available:
            logger.info("Groq configured as fallback")
        else:
            logger.warning("No GROQ_API_KEY configured; Groq routes are unavailable")

    def has_available_provider(self) -> bool:
        return self._openrouter_available or self._groq_available

    def provider_status(self) -> dict[str, dict[str, object]]:
        return {
            "openrouter": {
                "configured": self._openrouter_available,
                "judge_model": OPENROUTER_JUDGE_MODEL,
                "reasoning_model": OPENROUTER_REASONING_MODEL,
                "general_model": OPENROUTER_GENERAL_MODEL,
            },
            "groq": {
                "configured": self._groq_available,
                "primary_model": GROQ_QWEN_MODEL,
                "fallback_model": GROQ_LLAMA_MODEL,
            },
        }

    async def complete(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Route an LLM call to the best provider for the given role."""
        if not self.has_available_provider():
            raise RuntimeError("No LLM provider is configured. Set OPENROUTER_API_KEY or GROQ_API_KEY.")

        chain = ROLE_CHAINS.get(role, ROLE_CHAINS[LLMRole.GENERAL])
        last_error: Exception | None = None

        for provider, model in chain:
            if provider == "openrouter" and not self._openrouter_available:
                continue
            if provider == "groq" and not self._groq_available:
                continue

            try:
                if provider == "openrouter":
                    response = await self._call_openrouter(
                        role=role,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    response = await self._call_groq(
                        role=role,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                if _is_empty_response_text(response.text):
                    raise RuntimeError(f"Empty response from {provider}/{model}")
                return response
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM call failed: provider=%s model=%s role=%s error=%s; trying next",
                    provider,
                    model,
                    role.value,
                    str(exc)[:160],
                )

        raise RuntimeError(
            f"All LLM providers failed for role={role.value}. Last error: {last_error}"
        )

    async def _call_openrouter(
        self,
        *,
        role: LLMRole,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        await self._respect_rate_limit("openrouter", OPENROUTER_MIN_DELAY_SECONDS)
        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/ablayed/alpharadar",
                        "X-Title": "AlphaRadar",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                detail = _response_error_detail(exc.response)
                if status_code == 401:
                    raise RuntimeError(
                        "OpenRouter rejected the API key. Check OPENROUTER_API_KEY."
                    ) from exc
                if status_code == 402:
                    raise RuntimeError("OpenRouter credit exhausted. Falling back.") from exc
                if status_code == 429:
                    raise RuntimeError("OpenRouter rate limit hit. Retry in a minute.") from exc
                raise RuntimeError(f"OpenRouter request failed ({status_code}): {detail}") from exc

        data = response.json()
        text = _strip_reasoning_traces(model, _extract_message_text(data))
        return LLMResponse(
            text=text,
            model=model,
            provider="openrouter",
            tokens_used=int(data.get("usage", {}).get("total_tokens", 0) or 0),
            role=role,
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

        await self._respect_rate_limit("groq", GROQ_MIN_DELAY_SECONDS)
        async with httpx.AsyncClient(timeout=30.0) as client:
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
                if status_code == 413:
                    raise RuntimeError("Groq payload too large for this model.") from exc
                if status_code == 429:
                    raise RuntimeError("Groq rate limit or quota hit. Retry in a minute.") from exc
                raise RuntimeError(f"Groq request failed ({status_code}): {detail}") from exc

        data = response.json()
        return LLMResponse(
            text=_extract_message_text(data),
            model=model,
            provider="groq",
            tokens_used=int(data.get("usage", {}).get("total_tokens", 0) or 0),
            role=role,
        )

    async def _respect_rate_limit(self, provider: str, min_delay: float) -> None:
        lock = _provider_locks[provider]
        async with lock:
            elapsed = time.monotonic() - _last_call.get(provider, 0.0)
            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)
            _last_call[provider] = time.monotonic()

    async def get_openrouter_credit(self) -> float | None:
        """Fetch remaining OpenRouter credit balance."""
        if not self._openrouter_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/auth/key",
                    headers={"Authorization": f"Bearer {self._openrouter_key}"},
                )
                response.raise_for_status()
        except Exception:
            logger.debug("Could not fetch OpenRouter credit balance", exc_info=True)
            return None

        data = response.json()
        payload = data.get("data", {})
        if not isinstance(payload, dict):
            return None

        limit = payload.get("limit")
        usage = payload.get("usage")
        if not isinstance(limit, int | float) or not isinstance(usage, int | float):
            return None
        return round(float(limit) - float(usage), 4)


_provider: MultiProviderLLM | None = None


def get_provider() -> MultiProviderLLM:
    global _provider
    if _provider is None:
        _provider = MultiProviderLLM()
    return _provider


def init_provider(
    groq_key: str | None = None,
    openrouter_key: str | None = None,
) -> None:
    global _provider
    _provider = MultiProviderLLM(groq_key, openrouter_key)
