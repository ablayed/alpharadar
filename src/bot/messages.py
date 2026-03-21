from __future__ import annotations

from typing import TYPE_CHECKING

from telegram import Bot

from src.briefing.formatter import _send_briefing_parts

if TYPE_CHECKING:
    from telegram import Message


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
        await initial_message.edit_text(parts[0])
        for part in parts[1:]:
            await bot.send_message(chat_id=chat_id, text=part)
        return

    for part in parts:
        await bot.send_message(chat_id=chat_id, text=part)
