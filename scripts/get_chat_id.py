"""
Run this script, then send any message to your bot on Telegram.
It will print your chat ID.
"""

from __future__ import annotations

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters


async def echo_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    if update.message is None or update.effective_chat is None:
        return
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Your chat ID is: {chat_id}")
    print(f"\n✅ Your chat ID is: {chat_id}")
    print("Add this to your .env file as TELEGRAM_CHAT_ID")


def main() -> None:
    token = input("Enter bot token: ").strip()
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.ALL, echo_chat_id))
    print("Bot started. Send any message to your bot on Telegram...")
    app.run_polling()


if __name__ == "__main__":
    main()

