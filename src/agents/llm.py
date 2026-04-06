"""
Role-aware async wrapper around the multi-provider LLM router.
"""
from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from typing import Any

from src.agents.providers import (
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    GROQ_LLAMA_MODEL,
    GROQ_QWEN_MODEL,
    LLMResponse,
    LLMRole,
    get_provider,
)
from src.agents.state import LLMUsageSummary

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    GROQ_QWEN_MODEL: "Qwen3-32B",
    GROQ_LLAMA_MODEL: "Llama 3.3 70B",
    GEMINI_PRO_MODEL: "Gemini 2.5 Pro",
    GEMINI_FLASH_MODEL: "Gemini 2.5 Flash",
}

ROLE_DISPLAY_NAMES = {
    LLMRole.BULL.value: "Bull",
    LLMRole.BEAR.value: "Bear",
    LLMRole.JUDGE.value: "Judge",
    LLMRole.PLAYBOOK.value: "Playbook",
    LLMRole.REFLECTION.value: "Reflection",
    LLMRole.GENERAL.value: "General",
}


def model_display_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def role_display_name(role: str) -> str:
    return ROLE_DISPLAY_NAMES.get(role, role.replace("_", " ").title())


class AgentLLM:
    """Compatibility wrapper over the multi-provider router."""

    def __init__(self) -> None:
        self._provider = get_provider()
        self._initialized = False
        self._call_count = 0
        self._history: list[LLMResponse] = []

    async def initialize(self) -> None:
        if not self._provider.has_available_provider():
            raise RuntimeError("No LLM provider is configured")
        self._initialized = True
        status = self._provider.provider_status()
        logger.info(
            "LLM client initialized (groq=%s, gemini=%s)",
            status["groq"]["configured"],
            status["gemini"]["configured"],
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    def mark_usage(self) -> int:
        return len(self._history)

    def usage_since(self, marker: int) -> list[LLMUsageSummary]:
        grouped: OrderedDict[tuple[str, str, str], LLMUsageSummary] = OrderedDict()
        for record in self._history[marker:]:
            key = (record.role.value, record.provider, record.model)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = LLMUsageSummary(
                    role=record.role.value,
                    provider=record.provider,
                    model=record.model,
                    model_display=model_display_name(record.model),
                    calls=1,
                    tokens_used=record.tokens_used,
                )
                continue
            existing.calls += 1
            existing.tokens_used += record.tokens_used
        return list(grouped.values())

    def provider_status(self) -> dict[str, dict[str, object]]:
        return self._provider.provider_status()

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        role: LLMRole = LLMRole.GENERAL,
    ) -> str:
        if not self._initialized:
            raise RuntimeError("GeminiClient is not initialized")

        response = await self._provider.complete(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._history.append(response)
        self._call_count += 1
        logger.info(
            "LLM call succeeded: role=%s provider=%s model=%s tokens=%d",
            role.value,
            response.provider,
            response.model,
            response.tokens_used,
        )
        return response.text

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        role: LLMRole = LLMRole.GENERAL,
    ) -> Any:
        text = await self.generate(
            system_prompt=(
                f"{system_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. "
                "Do not use markdown or code fences."
            ),
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            role=role,
        )

        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response: %s", text[:200])
            return {"raw": text}

        if isinstance(parsed, dict | list):
            return parsed

        logger.warning("LLM returned unsupported JSON type: %s", type(parsed).__name__)
        return {"raw": text}

    async def call_bull(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=1500,
            role=LLMRole.BULL,
        )

    async def call_bear(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=1500,
            role=LLMRole.BEAR,
        )

    async def call_judge(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2000,
            role=LLMRole.JUDGE,
        )

    async def call_playbook(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=2000,
            role=LLMRole.PLAYBOOK,
        )

    async def call_reflection(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2000,
            role=LLMRole.REFLECTION,
        )


class GeminiClient(AgentLLM):
    """
    Backward-compatible name for the shared agent LLM wrapper.
    """

