from __future__ import annotations

import datetime
import logging
from typing import Any
from zoneinfo import ZoneInfo

from telegram.ext import Application, ContextTypes

from src.alerts import AlertPipeline, AlertSeverity
from src.bot.messages import _send_long_message
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
    job_queue.run_daily(
        callback=set_stock_reference_prices,
        time=datetime.time(hour=9, minute=30, second=0, tzinfo=ZoneInfo("America/New_York")),
        days=(0, 1, 2, 3, 4),
        name="stock_reference_prices",
    )
    job_queue.run_daily(
        callback=set_crypto_reference_prices,
        time=datetime.time(hour=0, minute=0, second=0, tzinfo=ZoneInfo("UTC")),
        days=tuple(range(7)),
        name="crypto_reference_prices",
    )
    job_queue.run_daily(
        callback=record_prediction_outcomes,
        time=datetime.time(hour=22, minute=0, second=0, tzinfo=ZoneInfo("UTC")),
        days=tuple(range(7)),
        name="record_prediction_outcomes",
    )
    job_queue.run_daily(
        callback=run_weekly_reflection_job,
        time=datetime.time(hour=8, minute=0, second=0, tzinfo=ZoneInfo("UTC")),
        days=(6,),
        name="weekly_reflection",
    )
    job_queue.run_daily(
        callback=update_correlation_engine_job,
        time=datetime.time(hour=23, minute=0, second=0, tzinfo=ZoneInfo("UTC")),
        days=tuple(range(7)),
        name="update_correlation_engine",
    )
    job_queue.run_daily(
        callback=detect_and_store_regime_job,
        time=datetime.time(hour=23, minute=5, second=0, tzinfo=ZoneInfo("UTC")),
        days=tuple(range(7)),
        name="detect_and_store_regime",
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
        context.bot_data["latest_briefing_data"] = data
        context.bot_data["current_regime"] = data.regime_state
        context.bot_data["composite_scores"] = data.composite_scores
        context.bot_data["lead_lag_pairs"] = data.lead_lag_pairs
        if data.is_weekend:
            briefing_type = "weekend"
            message = formatter.format_weekend(data)
        else:
            message = formatter.format_daily(data)

        if context.job is not None and context.job.chat_id is not None:
            chat_id: int | str = context.job.chat_id
        else:
            chat_id = int(settings.telegram_chat_id)
        await _send_long_message(context.bot, chat_id, message)

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


async def set_stock_reference_prices(context: ContextTypes.DEFAULT_TYPE) -> None:
    aggregator = context.bot_data.get("aggregator")
    pipeline = context.bot_data.get("alert_pipeline")
    if not isinstance(aggregator, DataAggregator) or not isinstance(pipeline, AlertPipeline):
        logger.error("Stock reference job dependencies missing in bot_data")
        return

    try:
        prices = await aggregator.fetch_stock_prices()
        for price in prices:
            pipeline.set_reference_price(price.symbol, price.current_price)
        logger.info("Set stock reference prices for %d symbols", len(prices))
    except Exception:
        logger.exception("Failed to set stock reference prices")


async def set_crypto_reference_prices(context: ContextTypes.DEFAULT_TYPE) -> None:
    aggregator = context.bot_data.get("aggregator")
    pipeline = context.bot_data.get("alert_pipeline")
    if not isinstance(aggregator, DataAggregator) or not isinstance(pipeline, AlertPipeline):
        logger.error("Crypto reference job dependencies missing in bot_data")
        return

    try:
        prices = await aggregator.fetch_crypto_prices()
        for price in prices:
            pipeline.set_reference_price(price.symbol, price.current_price)
        logger.info("Set crypto reference prices for %d symbols", len(prices))
    except Exception:
        logger.exception("Failed to set crypto reference prices")


async def record_prediction_outcomes(context: ContextTypes.DEFAULT_TYPE) -> None:
    aggregator = context.bot_data.get("aggregator")
    if not isinstance(aggregator, DataAggregator):
        logger.error("Outcome tracker dependency missing in bot_data")
        return

    try:
        count = await aggregator.record_prediction_outcomes()
        logger.info("Prediction outcome job completed: %d updates", count)
    except Exception:
        logger.exception("Prediction outcome job failed")


async def run_weekly_reflection_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    aggregator = context.bot_data.get("aggregator")
    if not isinstance(aggregator, DataAggregator):
        logger.error("Weekly reflection dependency missing in bot_data")
        return

    try:
        reflection = await aggregator.run_weekly_reflection()
        if reflection:
            logger.info("Weekly reflection stored successfully")
    except Exception:
        logger.exception("Weekly reflection job failed")


async def update_correlation_engine_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    aggregator = context.bot_data.get("aggregator")
    pipeline = context.bot_data.get("alert_pipeline")
    if not isinstance(aggregator, DataAggregator):
        logger.error("Correlation engine dependency missing in bot_data")
        return

    try:
        snapshots = await aggregator.update_correlation_engine(persist=True)
        context.bot_data["correlation_engine"] = aggregator.correlation_engine
        logger.info("Correlation engine updated for %d window(s)", len(snapshots))
        if isinstance(pipeline, AlertPipeline):
            await _send_correlation_anomaly_alerts(context, aggregator, pipeline)
    except Exception:
        logger.exception("Correlation engine update failed")


async def detect_and_store_regime_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    aggregator = context.bot_data.get("aggregator")
    if not isinstance(aggregator, DataAggregator):
        logger.error("Regime detection dependency missing in bot_data")
        return

    try:
        regime_state = await aggregator.refresh_regime_state(persist=True)
        context.bot_data["current_regime"] = regime_state
        if regime_state is not None:
            logger.info("Stored market regime: %s (%.0f%%)", regime_state.regime.value, regime_state.confidence * 100)
    except Exception:
        logger.exception("Regime detection job failed")


async def _send_correlation_anomaly_alerts(
    context: ContextTypes.DEFAULT_TYPE,
    aggregator: DataAggregator,
    pipeline: AlertPipeline,
) -> None:
    chat_id = int(settings.telegram_chat_id)
    anomalies = [pair for pair in aggregator.correlation_engine.get_top_movers(5) if abs(pair.z_score) >= 2.0]
    for pair in anomalies[:2]:
        severity = AlertSeverity.CRITICAL if abs(pair.z_score) >= 3.0 else AlertSeverity.WARNING
        message = (
            f"🔗 <b>CORRELATION ALERT</b> [{severity.value.upper()}]\n\n"
            f"{pair.symbol_a}-{pair.symbol_b} shifted to {pair.corr_20d:+.2f} (20d) vs {pair.corr_90d:+.2f} (90d)\n"
            f"Δ={pair.divergence:+.2f} | z={pair.z_score:+.1f} | {pair.regime_signal}"
        )
        alert = pipeline.build_manual_alert(
            symbol=f"{pair.symbol_a}-{pair.symbol_b}",
            asset_type="correlation",
            severity=severity,
            message=message,
            signed_change=pair.divergence,
            current_value=pair.corr_20d,
            reference_value=pair.corr_90d,
        )
        if alert is None:
            continue
        pipeline.record_sent(alert)
        try:
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
        except Exception:
            logger.exception("Failed to send correlation anomaly alert for %s-%s", pair.symbol_a, pair.symbol_b)
