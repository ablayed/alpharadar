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
    """Register the daily briefing job with a verified fallback path."""
    job_queue = application.job_queue
    if job_queue is None:
        raise RuntimeError(
            "Job queue is not available. "
            "Install python-telegram-bot with: pip install 'python-telegram-bot[job-queue]'"
        )

    tz = ZoneInfo(settings.timezone)
    chat_id = int(settings.telegram_chat_id)
    briefing_time = datetime.time(
        hour=settings.briefing_hour,
        minute=settings.briefing_minute,
        second=0,
        tzinfo=tz,
    )

    logger.info(
        "Scheduling daily briefing at %s (%s) for chat_id=%s",
        briefing_time,
        settings.timezone,
        chat_id,
    )

    job = job_queue.run_daily(
        callback=send_scheduled_briefing,
        time=briefing_time,
        days=tuple(range(7)),
        name="daily_briefing",
        chat_id=chat_id,
    )

    next_run = getattr(job, "next_t", None) or getattr(job, "next_run_time", None)
    if next_run is not None:
        logger.info("Daily briefing scheduled. Next run: %s", next_run)
        return

    logger.warning(
        "run_daily produced next_run=None. "
        "Falling back to manual run_once + repeating strategy."
    )
    job.schedule_removal()

    now = datetime.datetime.now(tz)
    next_fire = now.replace(
        hour=settings.briefing_hour,
        minute=settings.briefing_minute,
        second=0,
        microsecond=0,
    )
    if next_fire <= now:
        next_fire += datetime.timedelta(days=1)

    seconds_until = (next_fire - now).total_seconds()
    logger.info(
        "Fallback: scheduling first briefing in %.0f seconds (at %s)",
        seconds_until,
        next_fire.strftime("%Y-%m-%d %H:%M %Z"),
    )

    job_queue.run_once(
        callback=_first_briefing_then_repeat,
        when=seconds_until,
        chat_id=chat_id,
        name="daily_briefing_fallback",
        data={"tz": settings.timezone},
    )


async def _first_briefing_then_repeat(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fire once, then schedule the next fire at the configured local time."""
    logger.info("Fallback scheduler: firing briefing")
    await send_scheduled_briefing(context)

    tz = ZoneInfo(settings.timezone)
    now = datetime.datetime.now(tz)
    next_fire = now.replace(
        hour=settings.briefing_hour,
        minute=settings.briefing_minute,
        second=0,
        microsecond=0,
    ) + datetime.timedelta(days=1)

    seconds_until = (next_fire - now).total_seconds()
    if context.job is not None and context.job.chat_id is not None:
        chat_id = context.job.chat_id
    else:
        chat_id = int(settings.telegram_chat_id)

    logger.info("Fallback scheduler: next briefing in %.0fs", seconds_until)
    if context.job_queue is not None:
        context.job_queue.run_once(
            callback=_first_briefing_then_repeat,
            when=seconds_until,
            chat_id=chat_id,
            name="daily_briefing_fallback",
            data={"tz": settings.timezone},
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
        logger.info("Briefing sent successfully (%s, %s chars)", briefing_type, len(message))
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
            logger.info("Retrying briefing (attempt %s/3) in 5 minutes", retry_count + 1)
            job_queue.run_once(
                send_scheduled_briefing,
                when=300,
                chat_id=chat_id,
                name=f"briefing_retry_{retry_count + 1}",
                data={"retry_count": retry_count + 1},
            )
            return

        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=formatter.format_error(str(exc)),
            )
        except Exception:
            logger.exception("Failed to send error message")
        await db.log_briefing(briefing_type, 0, "failed", error=str(exc))
