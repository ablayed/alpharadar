from __future__ import annotations

import datetime
import html
import logging
from typing import Any
from zoneinfo import ZoneInfo

from telegram.ext import Application, ContextTypes

from src.alerts import AlertPipeline, AlertSeverity
from src.bot.handlers import save_signal_to_log
from src.bot.messages import _send_long_message
from src.config import settings
from src.data.aggregator import DataAggregator
from src.data.discovery_runner import run_discovery
from src.data.hot_list import init_hot_list_table
from src.briefing.models import BriefingData
from src.events.models import DetectedEvent
from src.ml import MLEnsemble, train_all_models
from src.storage.database import Database
from src.ta.signals import TASignal
from src.trade import PortfolioTracker, TradeSignal, generate_crypto_trade_signal

logger = logging.getLogger(__name__)


async def setup_scheduler(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    """Register the daily briefing job with a verified fallback path."""
    job_queue = application.job_queue
    if job_queue is None:
        raise RuntimeError(
            "Job queue is not available. "
            "Install python-telegram-bot with: pip install 'python-telegram-bot[job-queue]'"
        )

    await init_hot_list_table(settings.db_path)

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
    for hour in (10, 22):
        job_queue.run_daily(
            callback=record_prediction_outcomes,
            time=datetime.time(hour=hour, minute=0, second=0, tzinfo=ZoneInfo("UTC")),
            days=tuple(range(7)),
            name=f"record_prediction_outcomes_{hour:02d}",
        )
    job_queue.run_daily(
        callback=weekly_ml_training,
        time=datetime.time(hour=6, minute=0, second=0, tzinfo=ZoneInfo(settings.timezone)),
        # python-telegram-bot maps 0-6 to Sunday-Saturday.
        days=(6,),
        name="weekly_ml_training",
    )
    job_queue.run_daily(
        callback=run_weekly_reflection_job,
        time=datetime.time(hour=8, minute=0, second=0, tzinfo=ZoneInfo("UTC")),
        # python-telegram-bot maps 0-6 to Sunday-Saturday.
        days=(0,),
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
    for hour in (6, 18):
        job_queue.run_daily(
            callback=run_discovery_job,
            time=datetime.time(hour=hour, minute=30, second=0, tzinfo=ZoneInfo(settings.timezone)),
            days=tuple(range(7)),
            name=f"discovery_scan_{hour:02d}",
        )
    job_queue.run_repeating(
        callback=check_open_positions,
        interval=4 * 3600,
        first=4 * 3600,
        name="check_open_positions",
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


async def _schedule_trade_scans(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    """Schedule the twice-daily crypto trade scans at 7 AM and 7 PM London."""
    job_queue = application.job_queue
    if job_queue is None:
        raise RuntimeError(
            "Job queue is not available. "
            "Install python-telegram-bot with: pip install 'python-telegram-bot[job-queue]'"
        )

    for hour in (7, 19):
        _schedule_trade_scan_job(job_queue, hour=hour, minute=0)


async def run_discovery_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the scheduled discovery scan before the trade scans."""
    logger.info("Running scheduled discovery scan")

    try:
        aggregator = context.bot_data.get("aggregator")
        if not isinstance(aggregator, DataAggregator):
            logger.error("Discovery scan dependencies missing in bot_data")
            return

        fixed_watchlist = {
            item.symbol.upper()
            for item in await aggregator.db.get_watchlist("crypto")
        }
        chat_id: int | str | None = None
        try:
            chat_id = int(settings.telegram_chat_id)
        except ValueError:
            chat_id = settings.telegram_chat_id

        await run_discovery(
            db_path=aggregator.db.db_path,
            coingecko_api_key=aggregator.settings.coingecko_api_key.get_secret_value(),
            fixed_watchlist=fixed_watchlist,
            bot=context.bot,
            chat_id=chat_id,
        )
        context.bot_data.pop("latest_briefing_data", None)
    except Exception:
        logger.exception("Scheduled discovery scan failed")


async def auto_trade_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the crypto trade scan and notify only when actionable setups exist."""
    job_data = context.job.data if context.job is not None and isinstance(context.job.data, dict) else {}
    hour = int(job_data.get("hour", 7))
    minute = int(job_data.get("minute", 0))

    logger.info("Running auto trade scan for %02d:%02d Europe/London", hour, minute)

    try:
        aggregator = context.bot_data.get("aggregator")
        if not isinstance(aggregator, DataAggregator):
            logger.error("Auto trade scan dependencies missing in bot_data")
            return

        data = await aggregator.gather_briefing_data(include_agent_analysis=True)
        _cache_briefing_snapshot(context, data)

        regime_key = (
            data.regime_state.regime.value
            if data.regime_state is not None
            else "default"
        )
        btc_change = _btc_change_pct(data)
        signals: list[TradeSignal] = []
        ensemble = context.bot_data.get("ml_ensemble")

        for price_item in data.crypto_prices:
            symbol = price_item.symbol.upper()
            ta_signal = data.ta_signals.get(symbol)
            signal = generate_crypto_trade_signal(
                symbol=symbol,
                current_price=price_item.current_price,
                ta_signal=ta_signal,
                composite=data.composite_scores.get(symbol),
                verdict=_verdict_for_symbol(data, symbol),
                event_data=_event_payload_for_symbol(data, symbol),
                regime=regime_key,
                btc_change_pct=btc_change,
                funding=data.funding.get(symbol),
                oi=data.oi.get(symbol),
                cme_gap=data.cme_gap if symbol == "BTC" else None,
                fear_greed=data.fear_greed,
                dxy=data.dxy,
                ml_prediction=_ml_prediction_for_signal(ensemble, symbol, ta_signal),
            )
            if signal is not None:
                signals.append(signal)

        if not signals:
            logger.info("Auto trade scan: no setups found, staying silent")
            return

        signals.sort(key=lambda item: item.confidence, reverse=True)

        tracker = context.bot_data.get("tracker")
        db = context.bot_data.get("db")
        db_path = db.db_path if isinstance(db, Database) else None
        for signal in signals:
            prediction_id: int | None = None
            if tracker is not None:
                prediction_id = await tracker.log_prediction(
                    symbol=signal.symbol,
                    asset_type="crypto",
                    predicted_direction="bullish" if signal.direction == "LONG" else "bearish",
                    confidence=signal.confidence,
                    price_at_prediction=signal.entry_high,
                    source="auto_trade_scan",
                    reasoning=signal.reasoning,
                    predicted_target=signal.tp2,
                )
            if db_path is not None:
                await save_signal_to_log(signal, db_path, signal_id=prediction_id)

        time_label = "MORNING" if hour < 12 else "EVENING"
        lines = [f"🎯 <b>{time_label} TRADE SCAN</b>  {len(signals)} setup(s) found", ""]
        for signal in signals[:5]:
            direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
            lines.append(
                f"{direction_emoji} <b>{html.escape(signal.symbol)}</b> "
                f"{html.escape(signal.direction if signal.direction == 'LONG' else 'SHORT BIAS')} "
                f"({signal.confidence:.0%}) {_trade_scan_stars(signal.confluence_count)}".rstrip()
            )
            lines.append(
                f"   Entry: {_format_trade_scan_price(signal.entry_low)} - "
                f"{_format_trade_scan_price(signal.entry_high)}"
            )
            lines.append(
                f"   SL: {_format_trade_scan_price(signal.stop_loss)} | "
                f"TP1: {_format_trade_scan_price(signal.tp1)} | "
                f"TP2: {_format_trade_scan_price(signal.tp2)}"
            )
            lines.append(
                f"   R:R 1:{signal.risk_reward} | {html.escape(signal.timeframe)} | "
                f"Risk: {signal.suggested_risk_pct:.0f}%"
            )
            lines.append("")

        lines.append("Use /trade &lt;SYMBOL&gt; for full spot execution details.")
        lines.append("")
        lines.append("Not financial advice. Max 2% risk per trade.")

        await context.bot.send_message(
            chat_id=int(settings.telegram_chat_id),
            text="\n".join(lines),
            parse_mode="HTML",
        )
        logger.info("Auto trade scan: sent %d signal(s)", len(signals))
    except Exception:
        logger.exception("Auto trade scan failed")
    finally:
        if context.job_queue is not None:
            _schedule_trade_scan_job(context.job_queue, hour=hour, minute=minute)


async def check_open_positions(context: ContextTypes.DEFAULT_TYPE) -> None:
    portfolio = context.bot_data.get("portfolio_tracker")
    aggregator = context.bot_data.get("aggregator")
    if not isinstance(portfolio, PortfolioTracker) or not isinstance(aggregator, DataAggregator):
        logger.error("Open position checker dependencies missing in bot_data")
        return
    if not portfolio.positions:
        return

    chat_id = _scheduler_chat_id()

    try:
        stale_symbols = [
            symbol
            for symbol in portfolio.get_dead_money_trades()
            if not portfolio.positions[symbol].dead_money_alerted
        ]
        if stale_symbols:
            lines = [
                "<b>DEAD MONEY ALERT</b>",
                "",
                "These trades have been open 7+ days without hitting TP1:",
            ]
            for symbol in stale_symbols:
                lines.append(f"- {html.escape(symbol)}")
                portfolio.mark_dead_money_alerted(symbol)
            lines.extend(
                [
                    "",
                    "Consider closing them to free up capital for better setups.",
                ]
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text="\n".join(lines),
                parse_mode="HTML",
            )

        for position in list(portfolio.positions.values()):
            if position.tp1_hit or position.tp1 is None:
                continue
            current_price = await aggregator.get_current_price(position.symbol, position.asset_type)
            if current_price is None or not _tp1_reached(position, current_price):
                continue

            portfolio.mark_tp1_hit(position.symbol)
            direction_label = "LONG" if position.direction == "LONG" else "SHORT"
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"<b>TP1 HIT</b>\n\n"
                    f"{html.escape(position.symbol)} {direction_label} reached TP1 at "
                    f"{_format_trade_scan_price(position.tp1)}.\n"
                    f"Move your stop to breakeven at {_format_trade_scan_price(position.entry_price)}."
                ),
                parse_mode="HTML",
            )
    except Exception:
        logger.exception("Open position checker failed")


def _schedule_trade_scan_job(job_queue: object, *, hour: int, minute: int) -> None:
    if not hasattr(job_queue, "run_once"):
        raise RuntimeError("Job queue does not support run_once scheduling")

    tz = ZoneInfo("Europe/London")
    now = datetime.datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)

    seconds_until = (target - now).total_seconds()
    logger.info(
        "Scheduled trade scan at %s (in %.0f seconds)",
        target.strftime("%Y-%m-%d %H:%M %Z"),
        seconds_until,
    )
    job_queue.run_once(
        auto_trade_scan,
        when=seconds_until,
        name=f"trade_scan_{hour:02d}_{minute:02d}",
        data={"hour": hour, "minute": minute},
    )


def _tp1_reached(position: object, current_price: float) -> bool:
    tp1 = getattr(position, "tp1", None)
    direction = getattr(position, "direction", "")
    if not isinstance(tp1, int | float):
        return False
    if direction == "LONG":
        return current_price >= float(tp1)
    return current_price <= float(tp1)


def _cache_briefing_snapshot(context: ContextTypes.DEFAULT_TYPE, data: BriefingData) -> None:
    context.bot_data["latest_briefing_data"] = data
    context.bot_data["current_regime"] = data.regime_state
    context.bot_data["composite_scores"] = data.composite_scores
    context.bot_data["lead_lag_pairs"] = data.lead_lag_pairs
    context.bot_data["ta_signals"] = data.ta_signals
    context.bot_data["agent_analysis"] = data.agent_analysis
    context.bot_data["funding"] = data.funding
    context.bot_data["oi"] = data.oi
    context.bot_data["fear_greed"] = data.fear_greed
    context.bot_data["dxy"] = data.dxy
    context.bot_data["cme_gap"] = data.cme_gap
    context.bot_data["btc_dominance"] = data.btc_dominance


def _btc_change_pct(data: BriefingData) -> float:
    for item in data.crypto_prices:
        if item.symbol.upper() == "BTC":
            return item.change_24h_pct
    return 0.0


def _verdict_for_symbol(data: BriefingData, symbol: str) -> object | None:
    analysis = data.agent_analysis
    if analysis is None:
        return None
    symbol_upper = symbol.upper()
    for verdict in analysis.debate_verdicts:
        if verdict.symbol.upper() == symbol_upper:
            return verdict
    return None


def _event_payload_for_symbol(data: BriefingData, symbol: str) -> dict[str, object] | None:
    event = _event_for_symbol(data, symbol)
    if event is None:
        return None
    return {
        "event_type": event.event_type.value,
        "remaining_drift_pct": event.remaining_drift_pct,
        "expected_car_pct": event.expected_car_pct,
    }


def _event_for_symbol(data: BriefingData, symbol: str) -> DetectedEvent | None:
    symbol_upper = symbol.upper()
    best_event: DetectedEvent | None = None
    best_score = -1.0
    for event in data.event_data:
        if symbol_upper not in {related.upper() for related in event.related_symbols}:
            continue
        score = abs(event.remaining_drift_pct or event.expected_car_pct or 0.0) * event.confidence
        if score > best_score:
            best_event = event
            best_score = score
    return best_event


def _format_trade_scan_price(value: float) -> str:
    if value >= 100:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:.2f}"
    return f"${value:.4f}"


def _trade_scan_stars(count: int) -> str:
    filled = max(0, min(5, count))
    return ("*" * filled) + ("." * (5 - filled))


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


async def weekly_ml_training(context: ContextTypes.DEFAULT_TYPE) -> None:
    ensemble = context.bot_data.get("ml_ensemble")
    db = context.bot_data.get("db")
    if not isinstance(ensemble, MLEnsemble) or not isinstance(db, Database):
        logger.info("Skipping weekly ML training because dependencies are missing")
        return

    try:
        crypto_symbols = [item.symbol.upper() for item in await db.get_watchlist("crypto")]
        if not crypto_symbols:
            logger.info("Skipping weekly ML training because the crypto watchlist is empty")
            return

        logger.info("Starting weekly ML training for %d crypto symbols", len(crypto_symbols))
        results = await train_all_models(ensemble, crypto_symbols, db_path=db.db_path)
        success_count = sum(1 for success in results.values() if success)
        logger.info(
            "Weekly ML training complete: %d/%d symbols succeeded",
            success_count,
            len(results),
        )
    except Exception:
        logger.exception("Weekly ML training failed")


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


def _scheduler_chat_id() -> int | str:
    try:
        return int(settings.telegram_chat_id)
    except ValueError:
        return settings.telegram_chat_id


def _ml_prediction_for_signal(
    ensemble_obj: object,
    symbol: str,
    ta_signal: TASignal | None,
) -> object | None:
    if not isinstance(ensemble_obj, MLEnsemble) or ta_signal is None:
        return None
    if (
        ta_signal._close is None
        or ta_signal._high is None
        or ta_signal._low is None
        or ta_signal._volume is None
    ):
        return None
    return ensemble_obj.predict_from_ohlcv(
        symbol,
        ta_signal._close,
        ta_signal._high,
        ta_signal._low,
        ta_signal._volume,
        ta_signal._open,
    )
