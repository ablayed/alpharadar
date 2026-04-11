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
    GROQ_LLAMA_MODEL,
    GROQ_QWEN_MODEL,
    OPENROUTER_GENERAL_MODEL,
    OPENROUTER_JUDGE_MODEL,
    OPENROUTER_REASONING_MODEL,
    LLMResponse,
    LLMRole,
    get_provider,
)
from src.agents.state import LLMUsageSummary

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    OPENROUTER_JUDGE_MODEL: "Gemini 2.5 Pro",
    OPENROUTER_REASONING_MODEL: "DeepSeek R1 0528",
    OPENROUTER_GENERAL_MODEL: "DeepSeek V3.2",
    GROQ_QWEN_MODEL: "Qwen3-32B",
    GROQ_LLAMA_MODEL: "Llama 3.3 70B",
}

PROVIDER_DISPLAY_NAMES = {
    "openrouter": "OpenRouter",
    "groq": "Groq",
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


def provider_display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.replace("_", " ").title())


def role_display_name(role: str) -> str:
    return ROLE_DISPLAY_NAMES.get(role, role.replace("_", " ").title())


def parse_llm_json(text: str) -> Any | None:
    """Parse JSON from LLM output, tolerating fences, traces, and wrapper text."""
    if not text or text.strip().lower() in {"", "none", "null"}:
        return None

    # Strip reasoning traces
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences (multiple patterns)
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract between first { and last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        json_str = cleaned[first_brace : last_brace + 1]

        # Fix common LLM JSON issues
        # Replace literal newlines inside strings with \n
        json_str = re.sub(r'(?<=: \")(.*?)(?=\"[,\n}])',
                          lambda m: m.group(0).replace('\n', ' '),
                          json_str, flags=re.DOTALL)
        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Last resort: raw_decode from first { or [
    decoder = json.JSONDecoder()
    for char in ["{", "["]:
        pos = cleaned.find(char)
        if pos != -1:
            try:
                parsed, _ = decoder.raw_decode(cleaned[pos:])
                return parsed
            except json.JSONDecodeError:
                continue

    return None

class AgentLLM:
    """Compatibility wrapper over the multi-provider router."""

    _usage_log: list[LLMResponse] = []

    def __init__(self) -> None:
        self._provider = get_provider()
        self._initialized = False
        self._call_count = 0
        self._history: list[LLMResponse] = []

    @classmethod
    def reset_usage(cls) -> None:
        AgentLLM._usage_log = []

    @classmethod
    def get_usage_summary(cls) -> str:
        if not AgentLLM._usage_log:
            return ""

        grouped: OrderedDict[tuple[str, str, str], int] = OrderedDict()
        for record in AgentLLM._usage_log:
            key = (
                role_display_name(record.role.value),
                provider_display_name(record.provider),
                model_display_name(record.model),
            )
            grouped[key] = grouped.get(key, 0) + 1

        lines: list[str] = []
        for (role_name, provider_name, model_name), count in grouped.items():
            call_label = "call" if count == 1 else "calls"
            lines.append(f"   {role_name}: {provider_name} / {model_name} ({count} {call_label})")
        return "\n".join(lines)

    async def initialize(self) -> None:
        if not self._provider.has_available_provider():
            raise RuntimeError("No LLM provider is configured")
        self._initialized = True
        status = self._provider.provider_status()
        logger.info(
            "LLM client initialized (openrouter=%s, groq=%s)",
            status["openrouter"]["configured"],
            status["groq"]["configured"],
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
            raise RuntimeError("AgentLLM is not initialized")

        response = await self._provider.complete(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.text or response.text.strip().lower() in {"", "none", "null"}:
            raise RuntimeError(
                f"Empty response from {response.provider}/{response.model} for role={role.value}"
            )
        self._history.append(response)
        AgentLLM._usage_log.append(response)
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
        parsed = parse_llm_json(text)
        if parsed is None:
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
            max_tokens=2000,
            role=LLMRole.BULL,
        )

    async def call_bear(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=2000,
            role=LLMRole.BEAR,
        )

    async def call_judge(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2500,
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
            max_tokens=2500,
            role=LLMRole.REFLECTION,
        )

    async def call_general(self, system_prompt: str, user_prompt: str) -> str:
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1000,
            role=LLMRole.GENERAL,
        )


class GeminiClient(AgentLLM):
    """
    Backward-compatible name for the shared agent LLM wrapper.
    """
