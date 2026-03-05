from __future__ import annotations

import datetime
import logging
from typing import Any
from zoneinfo import ZoneInfo

from telegram.ext import Application, ContextTypes

from src.config import settings
from src.data.aggregator import DataAggregator
from src.storage.database import Database

logger = logging.getLogger(__name__)


async def setup_scheduler(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    """Called once during startup to register scheduled jobs."""
    tz = ZoneInfo(settings.timezone)
    job_queue = application.job_queue
    if job_queue is None:
        raise RuntimeError("Job queue is not available. Install python-telegram-bot with job-queue extras.")

    job_queue.run_daily(
        callback=send_scheduled_briefing,
        time=datetime.time(
            hour=settings.briefing_hour,
            minute=settings.briefing_minute,
            tzinfo=tz,
        ),
        days=tuple(range(7)),
        name="daily_briefing",
        chat_id=int(settings.telegram_chat_id),
    )


async def send_scheduled_briefing(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and send the scheduled daily briefing."""
    briefing_type = "daily"
    aggregator = context.bot_data.get("aggregator")
    formatter = context.bot_data.get("formatter")
    db = context.bot_data.get("db")
    if not isinstance(aggregator, DataAggregator) or formatter is None or not isinstance(db, Database):
        logger.error("Scheduler dependencies missing in bot_data")
        return

    try:
        data = await aggregator.gather_briefing_data()
        if data.is_weekend:
            briefing_type = "weekend"
            message = formatter.format_weekend(data)
        else:
            message = formatter.format_daily(data)

        if context.job is not None and context.job.chat_id is not None:
            chat_id: int | str = context.job.chat_id
        else:
            chat_id = int(settings.telegram_chat_id)
        await context.bot.send_message(
            chat_id=chat_id,
            text=message,
        )

        status = "partial" if data.failed_sections else "sent"
        await db.log_briefing(briefing_type, len(message), status, sections=data.failed_sections)
    except Exception as exc:
        logger.exception("Scheduled briefing failed")
        retry_count = 0
        if context.job is not None and isinstance(context.job.data, dict):
            retry_count = int(context.job.data.get("retry_count", 0))

        if context.job is not None and context.job.chat_id is not None:
            chat_id = context.job.chat_id
        else:
            chat_id = int(settings.telegram_chat_id)
        job_queue = context.job_queue
        if retry_count < 3:
            if job_queue is None:
                raise RuntimeError("Job queue unavailable for retries")
            job_queue.run_once(
                send_scheduled_briefing,
                when=300,
                chat_id=chat_id,
                name=f"briefing_retry_{retry_count + 1}",
                data={"retry_count": retry_count + 1},
            )
            return

        await context.bot.send_message(
            chat_id=chat_id,
            text=formatter.format_error(str(exc)),
        )
        await db.log_briefing(briefing_type, 0, "failed", error=str(exc))
