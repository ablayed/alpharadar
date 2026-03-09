from __future__ import annotations

import io
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Fix Windows console encoding for emoji support.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, Defaults

from src.bot.handlers import (
    add_command,
    briefing_command,
    handle_asset_type_callback,
    help_command,
    remove_command,
    sentiment_command,
    start_command,
    status_command,
    watchlist_command,
)
from src.bot.scheduler import setup_scheduler
from src.briefing.formatter import BriefingFormatter
from src.config import settings
from src.data.aggregator import DataAggregator
from src.storage.database import Database

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        stream_handler,
        logging.handlers.RotatingFileHandler(
            "logs/alpharadar.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


async def post_init(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    """Initialize shared resources after application starts."""
    db = Database(settings.db_path)
    await db.initialize()

    aggregator = DataAggregator(db=db, settings=settings)
    await aggregator.initialize()

    formatter = BriefingFormatter()

    application.bot_data["db"] = db
    application.bot_data["aggregator"] = aggregator
    application.bot_data["formatter"] = formatter
    application.bot_data["sentiment_analyzer"] = aggregator.sentiment_analyzer
    application.bot_data["started_at"] = datetime.now(ZoneInfo(settings.timezone))

    await setup_scheduler(application)
    logger.info("AlphaRadar initialized successfully")


async def post_shutdown(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    aggregator = application.bot_data.get("aggregator")
    if isinstance(aggregator, DataAggregator):
        await aggregator.close()
    logger.info("AlphaRadar shutdown complete")


async def error_handler(update, context) -> None:
    import traceback

    print(f"ERROR: {context.error}")
    traceback.print_exception(type(context.error), context.error, context.error.__traceback__)


def main() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    application = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .concurrent_updates(True)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("briefing", briefing_command))
    application.add_handler(CommandHandler("watchlist", watchlist_command))
    application.add_handler(CommandHandler("add", add_command))
    application.add_handler(CommandHandler("remove", remove_command))
    application.add_handler(CommandHandler("sentiment", sentiment_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        CallbackQueryHandler(handle_asset_type_callback, pattern=r"^asset_type:")
    )
    application.add_error_handler(error_handler)

    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
