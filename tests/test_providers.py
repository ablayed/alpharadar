from __future__ import annotations

import pytest

from src.agents.providers import (
    GROQ_LLAMA_MODEL,
    GROQ_QWEN_MODEL,
    LLMResponse,
    LLMRole,
    MultiProviderLLM,
    OPENROUTER_GENERAL_MODEL,
    OPENROUTER_JUDGE_MODEL,
    OPENROUTER_REASONING_MODEL,
)


@pytest.mark.asyncio
async def test_judge_role_prefers_openrouter_gemini_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", openrouter_api_key="or-test")
    calls: list[tuple[str, str]] = []

    async def fake_openrouter(**kwargs: object) -> LLMResponse:
        calls.append(("openrouter", str(kwargs["model"])))
        return LLMResponse(
            text="judge",
            model=str(kwargs["model"]),
            provider="openrouter",
            tokens_used=42,
            role=LLMRole.JUDGE,
        )

    async def fail_groq(**kwargs: object) -> LLMResponse:
        raise AssertionError(f"Groq should not be called: {kwargs}")

    monkeypatch.setattr(provider, "_call_openrouter", fake_openrouter)
    monkeypatch.setattr(provider, "_call_groq", fail_groq)

    response = await provider.complete(
        role=LLMRole.JUDGE,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "openrouter"
    assert response.model == OPENROUTER_JUDGE_MODEL
    assert calls == [("openrouter", OPENROUTER_JUDGE_MODEL)]


@pytest.mark.asyncio
async def test_judge_role_falls_back_from_gemini_to_deepseek_r1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", openrouter_api_key="or-test")
    calls: list[tuple[str, str]] = []

    async def fake_openrouter(**kwargs: object) -> LLMResponse:
        model = str(kwargs["model"])
        calls.append(("openrouter", model))
        if model == OPENROUTER_JUDGE_MODEL:
            raise RuntimeError("Gemini unavailable")
        return LLMResponse(
            text="judge",
            model=model,
            provider="openrouter",
            tokens_used=55,
            role=LLMRole.JUDGE,
        )

    async def fail_groq(**kwargs: object) -> LLMResponse:
        raise AssertionError(f"Groq should not be called: {kwargs}")

    monkeypatch.setattr(provider, "_call_openrouter", fake_openrouter)
    monkeypatch.setattr(provider, "_call_groq", fail_groq)

    response = await provider.complete(
        role=LLMRole.JUDGE,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "openrouter"
    assert response.model == OPENROUTER_REASONING_MODEL
    assert calls == [
        ("openrouter", OPENROUTER_JUDGE_MODEL),
        ("openrouter", OPENROUTER_REASONING_MODEL),
    ]


@pytest.mark.asyncio
async def test_judge_role_falls_back_to_groq_after_openrouter_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", openrouter_api_key="or-test")
    calls: list[tuple[str, str]] = []

    async def fail_openrouter(**kwargs: object) -> LLMResponse:
        calls.append(("openrouter", str(kwargs["model"])))
        raise RuntimeError("OpenRouter unavailable")

    async def fake_groq(**kwargs: object) -> LLMResponse:
        calls.append(("groq", str(kwargs["model"])))
        return LLMResponse(
            text="judge",
            model=str(kwargs["model"]),
            provider="groq",
            tokens_used=21,
            role=LLMRole.JUDGE,
        )

    monkeypatch.setattr(provider, "_call_openrouter", fail_openrouter)
    monkeypatch.setattr(provider, "_call_groq", fake_groq)

    response = await provider.complete(
        role=LLMRole.JUDGE,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "groq"
    assert response.model == GROQ_LLAMA_MODEL
    assert calls == [
        ("openrouter", OPENROUTER_JUDGE_MODEL),
        ("openrouter", OPENROUTER_REASONING_MODEL),
        ("groq", GROQ_LLAMA_MODEL),
    ]


@pytest.mark.asyncio
async def test_general_role_uses_groq_when_openrouter_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", openrouter_api_key=None)
    calls: list[str] = []

    async def fake_groq(**kwargs: object) -> LLMResponse:
        calls.append(str(kwargs["model"]))
        return LLMResponse(
            text="general",
            model=str(kwargs["model"]),
            provider="groq",
            tokens_used=18,
            role=LLMRole.GENERAL,
        )

    monkeypatch.setattr(provider, "_call_groq", fake_groq)

    response = await provider.complete(
        role=LLMRole.GENERAL,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "groq"
    assert response.model == GROQ_QWEN_MODEL
    assert calls == [GROQ_QWEN_MODEL]


@pytest.mark.asyncio
async def test_playbook_role_prefers_openrouter_reasoning_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", openrouter_api_key="or-test")
    calls: list[tuple[str, str]] = []

    async def fake_openrouter(**kwargs: object) -> LLMResponse:
        calls.append(("openrouter", str(kwargs["model"])))
        return LLMResponse(
            text="playbook",
            model=str(kwargs["model"]),
            provider="openrouter",
            tokens_used=64,
            role=LLMRole.PLAYBOOK,
        )

    async def fail_groq(**kwargs: object) -> LLMResponse:
        raise AssertionError(f"Groq should not be called: {kwargs}")

    monkeypatch.setattr(provider, "_call_openrouter", fake_openrouter)
    monkeypatch.setattr(provider, "_call_groq", fail_groq)

    response = await provider.complete(
        role=LLMRole.PLAYBOOK,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "openrouter"
    assert response.model == OPENROUTER_REASONING_MODEL
    assert calls == [("openrouter", OPENROUTER_REASONING_MODEL)]


@pytest.mark.asyncio
async def test_bull_role_prefers_v32_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", openrouter_api_key="or-test")
    calls: list[tuple[str, str]] = []

    async def fake_openrouter(**kwargs: object) -> LLMResponse:
        model = str(kwargs["model"])
        calls.append(("openrouter", model))
        if model == OPENROUTER_REASONING_MODEL:
            raise RuntimeError("R1 unavailable")
        return LLMResponse(
            text="bull",
            model=model,
            provider="openrouter",
            tokens_used=27,
            role=LLMRole.BULL,
        )

    async def fail_groq(**kwargs: object) -> LLMResponse:
        raise AssertionError(f"Groq should not be called: {kwargs}")

    monkeypatch.setattr(provider, "_call_openrouter", fake_openrouter)
    monkeypatch.setattr(provider, "_call_groq", fail_groq)

    response = await provider.complete(
        role=LLMRole.BULL,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "openrouter"
    assert response.model == OPENROUTER_GENERAL_MODEL
    assert calls == [("openrouter", OPENROUTER_GENERAL_MODEL)]


@pytest.mark.asyncio
async def test_bull_role_falls_back_to_groq_when_v32_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MultiProviderLLM(groq_api_key="groq-test", openrouter_api_key="or-test")
    calls: list[tuple[str, str]] = []

    async def fake_openrouter(**kwargs: object) -> LLMResponse:
        model = str(kwargs["model"])
        calls.append(("openrouter", model))
        return LLMResponse(
            text="None",
            model=model,
            provider="openrouter",
            tokens_used=0,
            role=LLMRole.BULL,
        )

    async def fake_groq(**kwargs: object) -> LLMResponse:
        model = str(kwargs["model"])
        calls.append(("groq", model))
        return LLMResponse(
            text="fallback",
            model=model,
            provider="groq",
            tokens_used=12,
            role=LLMRole.BULL,
        )

    monkeypatch.setattr(provider, "_call_openrouter", fake_openrouter)
    monkeypatch.setattr(provider, "_call_groq", fake_groq)

    response = await provider.complete(
        role=LLMRole.BULL,
        system_prompt="system",
        user_prompt="user",
    )

    assert response.provider == "groq"
    assert response.model == GROQ_QWEN_MODEL
    assert calls == [
        ("openrouter", OPENROUTER_GENERAL_MODEL),
        ("groq", GROQ_QWEN_MODEL),
    ]


@pytest.mark.asyncio
async def test_complete_raises_when_no_provider_is_configured() -> None:
    provider = MultiProviderLLM(groq_api_key=None, openrouter_api_key=None)

    with pytest.raises(RuntimeError, match="No LLM provider is configured"):
        await provider.complete(
            role=LLMRole.GENERAL,
            system_prompt="system",
            user_prompt="user",
        )
