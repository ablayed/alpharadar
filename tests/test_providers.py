from __future__ import annotations

import pytest

from src.agents.providers import (
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    GROQ_LLAMA_MODEL,
    GROQ_QWEN_MODEL,
    LLMResponse,
    LLMRole,
    MultiProviderLLM,
)


@pytest.mark.asyncio
async def test_judge_role_prefers_gemini_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", gemini_api_key="gemini-test")
    calls: list[tuple[str, str]] = []

    async def fake_gemini(**kwargs: object) -> LLMResponse:
        calls.append(("gemini", str(kwargs["model"])))
        return LLMResponse(
            text="judge",
            model=str(kwargs["model"]),
            provider="gemini",
            tokens_used=42,
            role=LLMRole.JUDGE,
        )

    async def fail_groq(**kwargs: object) -> LLMResponse:
        raise AssertionError(f"Groq should not be called: {kwargs}")

    monkeypatch.setattr(provider, "_call_gemini", fake_gemini)
    monkeypatch.setattr(provider, "_call_groq", fail_groq)

    response = await provider.complete(
        role=LLMRole.JUDGE,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "gemini"
    assert response.model == GEMINI_PRO_MODEL
    assert calls == [("gemini", GEMINI_PRO_MODEL)]


@pytest.mark.asyncio
async def test_judge_role_falls_back_from_gemini_to_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", gemini_api_key="gemini-test")
    calls: list[tuple[str, str]] = []

    async def fail_gemini(**kwargs: object) -> LLMResponse:
        calls.append(("gemini", str(kwargs["model"])))
        raise RuntimeError("Gemini unavailable")

    async def fake_groq(**kwargs: object) -> LLMResponse:
        calls.append(("groq", str(kwargs["model"])))
        return LLMResponse(
            text="judge",
            model=str(kwargs["model"]),
            provider="groq",
            tokens_used=55,
            role=LLMRole.JUDGE,
        )

    monkeypatch.setattr(provider, "_call_gemini", fail_gemini)
    monkeypatch.setattr(provider, "_call_groq", fake_groq)

    response = await provider.complete(
        role=LLMRole.JUDGE,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "groq"
    assert response.model == GROQ_QWEN_MODEL
    assert calls == [
        ("gemini", GEMINI_PRO_MODEL),
        ("groq", GROQ_QWEN_MODEL),
    ]


@pytest.mark.asyncio
async def test_general_role_uses_gemini_flash_when_groq_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MultiProviderLLM(groq_api_key=None, gemini_api_key="gemini-test")
    calls: list[str] = []

    async def fake_gemini(**kwargs: object) -> LLMResponse:
        calls.append(str(kwargs["model"]))
        return LLMResponse(
            text="general",
            model=str(kwargs["model"]),
            provider="gemini",
            tokens_used=18,
            role=LLMRole.GENERAL,
        )

    monkeypatch.setattr(provider, "_call_gemini", fake_gemini)

    response = await provider.complete(
        role=LLMRole.BULL,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "gemini"
    assert response.model == GEMINI_FLASH_MODEL
    assert calls == [GEMINI_FLASH_MODEL]


@pytest.mark.asyncio
async def test_playbook_role_prefers_llama_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", gemini_api_key="gemini-test")
    calls: list[tuple[str, str]] = []

    async def fake_groq(**kwargs: object) -> LLMResponse:
        calls.append(("groq", str(kwargs["model"])))
        return LLMResponse(
            text="playbook",
            model=str(kwargs["model"]),
            provider="groq",
            tokens_used=64,
            role=LLMRole.PLAYBOOK,
        )

    async def fail_gemini(**kwargs: object) -> LLMResponse:
        raise AssertionError(f"Gemini should not be called: {kwargs}")

    monkeypatch.setattr(provider, "_call_groq", fake_groq)
    monkeypatch.setattr(provider, "_call_gemini", fail_gemini)

    response = await provider.complete(
        role=LLMRole.PLAYBOOK,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "groq"
    assert response.model == GROQ_LLAMA_MODEL
    assert calls == [("groq", GROQ_LLAMA_MODEL)]


@pytest.mark.asyncio
async def test_playbook_role_falls_back_from_llama_to_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", gemini_api_key="gemini-test")
    calls: list[tuple[str, str]] = []

    async def fail_groq(**kwargs: object) -> LLMResponse:
        calls.append(("groq", str(kwargs["model"])))
        raise RuntimeError("Groq unavailable")

    async def fake_gemini(**kwargs: object) -> LLMResponse:
        calls.append(("gemini", str(kwargs["model"])))
        return LLMResponse(
            text="playbook",
            model=str(kwargs["model"]),
            provider="gemini",
            tokens_used=32,
            role=LLMRole.PLAYBOOK,
        )

    monkeypatch.setattr(provider, "_call_groq", fail_groq)
    monkeypatch.setattr(provider, "_call_gemini", fake_gemini)

    response = await provider.complete(
        role=LLMRole.PLAYBOOK,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "gemini"
    assert response.model == GEMINI_FLASH_MODEL
    assert calls == [
        ("groq", GROQ_LLAMA_MODEL),
        ("gemini", GEMINI_FLASH_MODEL),
    ]


@pytest.mark.asyncio
async def test_complete_raises_when_no_provider_is_configured() -> None:
    provider = MultiProviderLLM(groq_api_key=None, gemini_api_key=None)

    with pytest.raises(RuntimeError, match="No LLM provider is configured"):
        await provider.complete(
            role=LLMRole.GENERAL,
            system_prompt="system",
            user_prompt="user",
        )
