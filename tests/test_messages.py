from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest

from src.bot.messages import _plain_text_fallback, _send_long_message


def test_plain_text_fallback_strips_supported_tags_only() -> None:
    text = '<b>HOT LIST</b> <a href="https://dexscreener.com">review</a> <symbol>'

    assert _plain_text_fallback(text) == "HOT LIST review <symbol>"


@pytest.mark.asyncio
async def test_send_long_message_retries_as_plain_text_on_parse_error() -> None:
    bot = SimpleNamespace(
        send_message=AsyncMock(
            side_effect=[
                BadRequest('Can\'t parse entities: unsupported start tag "symbol" at byte offset 10'),
                None,
            ]
        )
    )

    await _send_long_message(bot, 123, "<b>HOT LIST</b> <symbol>")

    first_call = bot.send_message.await_args_list[0].kwargs
    second_call = bot.send_message.await_args_list[1].kwargs
    assert first_call == {"chat_id": 123, "text": "<b>HOT LIST</b> <symbol>"}
    assert second_call == {"chat_id": 123, "text": "HOT LIST <symbol>", "parse_mode": None}


@pytest.mark.asyncio
async def test_send_long_message_retries_edit_as_plain_text_on_parse_error() -> None:
    message = SimpleNamespace(
        edit_text=AsyncMock(
            side_effect=[
                BadRequest('Can\'t parse entities: unsupported start tag "symbol" at byte offset 10'),
                None,
            ]
        )
    )
    bot = SimpleNamespace(send_message=AsyncMock())

    await _send_long_message(bot, 123, "<b>STATUS</b> <symbol>", initial_message=message)

    first_call = message.edit_text.await_args_list[0]
    second_call = message.edit_text.await_args_list[1]
    assert first_call.args == ("<b>STATUS</b> <symbol>",)
    assert second_call.args == ("STATUS <symbol>",)
    assert second_call.kwargs == {"parse_mode": None}
