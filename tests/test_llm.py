from __future__ import annotations

import pytest

from src.agents.llm import AgentLLM, parse_llm_json
from src.agents.providers import LLMRole


def test_parse_llm_json_strips_fences_and_think_tags() -> None:
    payload = """<think>long reasoning</think>
```json
{"summary":"bearish","direction":"bearish"}
```"""

    parsed = parse_llm_json(payload)

    assert parsed == {"summary": "bearish", "direction": "bearish"}


def test_parse_llm_json_extracts_wrapped_array() -> None:
    payload = """
Here is the result:
[
  {"index": 0, "event_type": "earnings_beat", "confidence": 0.85}
]
Thanks.
"""

    parsed = parse_llm_json(payload)

    assert parsed == [
        {"index": 0, "event_type": "earnings_beat", "confidence": 0.85}
    ]


@pytest.mark.asyncio
async def test_generate_json_handles_fenced_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = AgentLLM()

    async def fake_generate(
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        role: LLMRole = LLMRole.GENERAL,
    ) -> str:
        del system_prompt, user_prompt, temperature, max_tokens, role
        return """```json
{"summary":"ok","signals":["a","b"]}
```"""

    monkeypatch.setattr(llm, "generate", fake_generate)

    parsed = await llm.generate_json(
        system_prompt="system",
        user_prompt="user",
        role=LLMRole.BULL,
    )

    assert parsed == {"summary": "ok", "signals": ["a", "b"]}
