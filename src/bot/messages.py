from __future__ import annotations

import re
from typing import TYPE_CHECKING

from telegram import Bot
from telegram.error import BadRequest

from src.briefing.formatter import _send_briefing_parts

if TYPE_CHECKING:
    from telegram import Message


_SUPPORTED_HTML_TAG_RE = re.compile(
    r"</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|blockquote|tg-spoiler)(?:\s+[^>]*)?>",
    re.IGNORECASE,
)
_LINK_TAG_RE = re.compile(r"</?a(?:\s+[^>]*)?>", re.IGNORECASE)


def _plain_text_fallback(text: str) -> str:
    """Strip supported Telegram HTML tags while preserving literal angle-bracket text."""
    text = _LINK_TAG_RE.sub("", text)
    return _SUPPORTED_HTML_TAG_RE.sub("", text)


async def _send_part(bot: Bot, chat_id: int | str, text: str) -> None:
    """Send a single part, falling back to plain text if Telegram rejects HTML parsing."""
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except BadRequest as exc:
        if "Can't parse entities" not in str(exc):
            raise
        await bot.send_message(chat_id=chat_id, text=_plain_text_fallback(text), parse_mode=None)


async def _edit_part(message: Message, text: str) -> None:
    """Edit a single part, falling back to plain text if Telegram rejects HTML parsing."""
    try:
        await message.edit_text(text)
    except BadRequest as exc:
        if "Can't parse entities" not in str(exc):
            raise
        await message.edit_text(_plain_text_fallback(text), parse_mode=None)


async def _send_long_message(
    bot: Bot,
    chat_id: int | str,
    text: str,
    *,
    initial_message: Message | None = None,
) -> None:
    """Send or edit a message, splitting at formatter-defined boundaries when needed."""
    parts = _send_briefing_parts(text)
    if not parts:
        return

    if initial_message is not None:
        await _edit_part(initial_message, parts[0])
        for part in parts[1:]:
            await _send_part(bot, chat_id, part)
        return

    for part in parts:
        await _send_part(bot, chat_id, part)
