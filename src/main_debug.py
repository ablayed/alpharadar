from __future__ import annotations

import io
import logging
import logging.handlers
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine
from zoneinfo import ZoneInfo

# Fix Windows console encoding for emoji support.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    Defaults,
    MessageHandler,
    ContextTypes,
    filters,
)

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.DEBUG,
    handlers=[
        stream_handler,
        logging.handlers.RotatingFileHandler(
            "logs/alpharadar_debug.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("DEBUG_MAIN")

# Keep networking noise low so debug signals are obvious.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.Updater").setLevel(logging.INFO)


async def log_all_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs for every update before command handlers."""
    del context
    chat = update.effective_chat
    user = update.effective_user
    msg = update.effective_message

    logger.info("INCOMING UPDATE")
    logger.info("Update ID:  %s", update.update_id)
    logger.info("Chat ID:    %s", chat.id if chat else "None")
    logger.info("Chat type:  %s", chat.type if chat else "None")
    logger.info(
        "User:       %s (id=%s)",
        user.first_name if user else "None",
        user.id if user else "None",
    )
    logger.info("Message:    %s", (msg.text[:80] if msg and msg.text else "None"))
    logger.info("")


def debug_wrap(
    handler_name: str,
    handler_func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, Any]],
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, Any]]:
    """Wrap any handler with pre/post logging."""

    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
        logger.info(">>> ENTERING handler: %s", handler_name)
        try:
            result = await handler_func(update, context)
            logger.info("<<< EXITING  handler: %s (success)", handler_name)
            return result
        except Exception as exc:
            logger.error("!!! CRASHED  handler: %s", handler_name)
            logger.error("    Exception: %s: %s", type(exc).__name__, exc)
            logger.error(traceback.format_exc())
            raise

    return wrapper


def debug_wrap_job(
    job_name: str,
    job_func: Callable[[ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]],
) -> Callable[[ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]:
    """Wrap a scheduled job callback with logging."""

    async def wrapper(context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.info("=== SCHEDULER FIRING: %s ===", job_name)
        logger.info("    Job data: %s", context.job.data if context.job else "No job")
        logger.info("    Chat ID:  %s", context.job.chat_id if context.job else "No job")
        try:
            await job_func(context)
            logger.info("=== SCHEDULER DONE: %s (success) ===", job_name)
        except Exception as exc:
            logger.error("=== SCHEDULER CRASHED: %s ===", job_name)
            logger.error("    Exception: %s: %s", type(exc).__name__, exc)
            logger.error(traceback.format_exc())

    return wrapper


async def debug_owner_only(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Owner-only checker with detailed logging."""
    del context
    from src.config import settings

    chat = update.effective_chat
    chat_id_str = str(chat.id) if chat else "None"
    expected = settings.telegram_chat_id

    logger.info("    owner_only check: got=%r expected=%r", chat_id_str, expected)
    logger.info("    types: got=%s expected=%s", type(chat_id_str), type(expected))
    logger.info(
        "    repr match: %r == %r -> %s",
        repr(chat_id_str),
        repr(expected),
        chat_id_str == expected,
    )

    if chat_id_str != expected:
        logger.warning(
            "    MISMATCH! Bytes: got=%r expected=%r",
            chat_id_str.encode(),
            expected.encode(),
        )
        if chat_id_str.strip() == expected.strip():
            logger.warning("    WHITESPACE ISSUE: values match after .strip()")
        return False

    logger.info("    owner_only: PASSED")
    return True


async def post_init(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    """Initialize data/services with full debug logging."""
    logger.info("post_init: Starting initialization...")

    from src.config import settings
    from src.agents.providers import init_provider
    from src.storage.database import Database
    from src.data.aggregator import DataAggregator
    from src.briefing.formatter import BriefingFormatter

    init_provider(
        groq_key=(settings.groq_api_key.get_secret_value() if settings.groq_api_key is not None else None),
        openrouter_key=(
            settings.openrouter_api_key.get_secret_value()
            if settings.openrouter_api_key is not None
            else None
        ),
    )

    logger.info("post_init: Initializing database...")
    db = Database(settings.db_path)
    await db.initialize()
    logger.info("post_init: Database OK")

    logger.info("post_init: Initializing aggregator...")
    aggregator = DataAggregator(db=db, settings=settings)
    await aggregator.initialize()
    logger.info("post_init: Aggregator OK")

    formatter = BriefingFormatter()
    logger.info("post_init: Formatter OK")

    application.bot_data["db"] = db
    application.bot_data["aggregator"] = aggregator
    application.bot_data["formatter"] = formatter
    application.bot_data["sentiment_analyzer"] = aggregator.sentiment_analyzer
    application.bot_data["started_at"] = datetime.now(ZoneInfo(settings.timezone))

    logger.info("post_init: Setting up scheduler...")
    from src.bot.scheduler import send_scheduled_briefing

    tz = ZoneInfo(settings.timezone)
    import datetime as dt

    job_queue = application.job_queue
    if job_queue is None:
        logger.error("post_init: JOB QUEUE IS NONE - job-queue extras not installed!")
        return

    scheduled_time = dt.time(
        hour=settings.briefing_hour,
        minute=settings.briefing_minute,
        tzinfo=tz,
    )
    logger.info(
        "post_init: Scheduling daily briefing at %s (%s)",
        scheduled_time,
        settings.timezone,
    )
    logger.info("post_init: Chat ID for job: %s", settings.telegram_chat_id)

    job_queue.run_daily(
        callback=debug_wrap_job("daily_briefing", send_scheduled_briefing),
        time=scheduled_time,
        days=tuple(range(7)),
        name="daily_briefing",
        chat_id=int(settings.telegram_chat_id),
    )

    jobs = job_queue.jobs()
    logger.info("post_init: %s job(s) registered:", len(jobs))
    for job in jobs:
        next_run = getattr(job, "next_t", None)
        if next_run is None:
            next_run = getattr(job, "next_run_time", None)
        logger.info("  - %s: next_run=%s, chat_id=%s", job.name, next_run, job.chat_id)

    logger.info("post_init: Initialization complete!")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("UNHANDLED ERROR")
    logger.error("Update: %s", update)
    logger.error("Error: %s", context.error)
    if context.error:
        traceback.print_exception(type(context.error), context.error, context.error.__traceback__)


def main() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    from src.config import settings

    logger.info("Building application...")

    application = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .concurrent_updates(True)
        .post_init(post_init)
        .build()
    )

    from src.bot.handlers import (
        start_command,
        briefing_command,
        analyze_command,
        funding_command,
        watchlist_command,
        add_command,
        remove_command,
        sentiment_command,
        status_command,
        help_command,
        handle_asset_type_callback,
    )

    application.add_handler(
        MessageHandler(filters.ALL, log_all_updates),
        group=-1,
    )

    application.add_handler(CommandHandler("start", debug_wrap("start_command", start_command)))
    application.add_handler(CommandHandler("briefing", debug_wrap("briefing_command", briefing_command)))
    application.add_handler(CommandHandler("analyze", debug_wrap("analyze_command", analyze_command)))
    application.add_handler(CommandHandler("funding", debug_wrap("funding_command", funding_command)))
    application.add_handler(CommandHandler("watchlist", debug_wrap("watchlist_command", watchlist_command)))
    application.add_handler(CommandHandler("add", debug_wrap("add_command", add_command)))
    application.add_handler(CommandHandler("remove", debug_wrap("remove_command", remove_command)))
    application.add_handler(CommandHandler("sentiment", debug_wrap("sentiment_command", sentiment_command)))
    application.add_handler(CommandHandler("status", debug_wrap("status_command", status_command)))
    application.add_handler(CommandHandler("help", debug_wrap("help_command", help_command)))
    application.add_handler(
        CallbackQueryHandler(
            debug_wrap("handle_asset_type_callback", handle_asset_type_callback),
            pattern=r"^asset_type:",
        )
    )

    application.add_error_handler(error_handler)

    logger.info("Starting polling...")
    logger.info("Registered %s handler(s) in group 0", len(application.handlers.get(0, [])))
    logger.info(
        "Registered %s handler(s) in group -1 (pre-processor)",
        len(application.handlers.get(-1, [])),
    )

    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
