from __future__ import annotations

import asyncio
import io
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, Defaults

from src.alerts import AlertPipeline, binance_ws_loop
from src.bot.handlers import (
    accuracy_command,
    add_command,
    alerts_command,
    analyze_command,
    briefing_command,
    check_trade_levels,
    close_command,
    corr_command,
    events_command,
    funding_command,
    handle_asset_type_callback,
    help_command,
    history_command,
    journal_command,
    log_command,
    playbook_command,
    portfolio_command,
    reflect_command,
    regime_command,
    remove_command,
    sentiment_command,
    signals_command,
    start_command,
    status_command,
    ta_command,
    trade_command,
    trades_command,
    watchlist_command,
)
from src.bot.scheduler import _schedule_trade_scans, setup_scheduler
from src.briefing.formatter import BriefingFormatter
from src.config import settings
from src.data.aggregator import DataAggregator
from src.ml import MLEnsemble, train_all_models
from src.storage.database import Database
from src.trade import PortfolioTracker

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
    application.bot_data["db_path"] = db.db_path
    application.bot_data["aggregator"] = aggregator
    application.bot_data["formatter"] = formatter
    application.bot_data["sentiment_analyzer"] = aggregator.sentiment_analyzer
    application.bot_data["tracker"] = aggregator.tracker
    application.bot_data["reflection_evaluator"] = aggregator.reflection_evaluator
    application.bot_data["correlation_engine"] = aggregator.correlation_engine
    application.bot_data["current_regime"] = aggregator.current_regime
    application.bot_data["settings"] = settings
    application.bot_data["started_at"] = datetime.now(ZoneInfo(settings.timezone))
    application.bot_data["alert_pipeline"] = AlertPipeline()
    application.bot_data["portfolio_tracker"] = PortfolioTracker()
    application.bot_data["ml_ensemble"] = MLEnsemble()

    await _seed_alert_reference_prices(application)
    await _start_alert_streams(application)
    await setup_scheduler(application)
    await _schedule_trade_scans(application)
    application.create_task(_run_initial_ml_training(application), name="initial_ml_training")
    logger.info("AlphaRadar initialized successfully")


async def post_shutdown(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    aggregator = application.bot_data.get("aggregator")
    if isinstance(aggregator, DataAggregator):
        await aggregator.close()
    logger.info("AlphaRadar shutdown complete")


async def _seed_alert_reference_prices(
    application: Application[Any, Any, Any, Any, Any, Any],
) -> None:
    aggregator = application.bot_data.get("aggregator")
    pipeline = application.bot_data.get("alert_pipeline")
    if not isinstance(aggregator, DataAggregator) or not isinstance(pipeline, AlertPipeline):
        return

    try:
        stock_prices, crypto_prices = await asyncio.gather(
            aggregator.fetch_stock_prices(),
            aggregator.fetch_crypto_prices(),
        )
        for price in stock_prices:
            pipeline.set_reference_price(price.symbol, price.current_price)
        for price in crypto_prices:
            pipeline.set_reference_price(price.symbol, price.current_price)
        logger.info(
            "Seeded alert reference prices for %d stocks and %d crypto assets",
            len(stock_prices),
            len(crypto_prices),
        )
    except Exception:
        logger.exception("Failed to seed alert reference prices")


async def _start_alert_streams(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    aggregator = application.bot_data.get("aggregator")
    pipeline = application.bot_data.get("alert_pipeline")
    db = application.bot_data.get("db")
    if (
        not isinstance(aggregator, DataAggregator)
        or not isinstance(pipeline, AlertPipeline)
        or not isinstance(db, Database)
    ):
        return

    watchlist = await db.get_watchlist()
    crypto_symbols = [item.symbol for item in watchlist if item.asset_type == "crypto"]
    chat_id = _telegram_chat_id()
    ws_tasks: dict[str, asyncio.Task[object] | None] = {"binance": None}
    application.bot_data["ws_tasks"] = ws_tasks

    async def on_crypto_price(symbol: str, price: float) -> None:
        alert = pipeline.classify_severity(symbol, price, "crypto")
        if alert is not None:
            pipeline.record_sent(alert)
            try:
                await application.bot.send_message(
                    chat_id=chat_id,
                    text=f"🚨 <b>CRYPTO ALERT</b> [{alert.severity.value.upper()}]\n\n{alert.message}",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                logger.exception("Failed to send crypto alert for %s", symbol.upper())
        try:
            await check_trade_levels(
                symbol,
                price,
                db.db_path,
                application.bot,
                chat_id,
                portfolio_tracker=application.bot_data.get("portfolio_tracker"),
            )
        except Exception:
            logger.exception("Failed to check trade levels for %s", symbol.upper())

    if crypto_symbols:
        ws_tasks["binance"] = application.create_task(
            binance_ws_loop(crypto_symbols, on_crypto_price),
            name="binance_ws",
        )


async def _run_initial_ml_training(
    application: Application[Any, Any, Any, Any, Any, Any],
) -> None:
    ensemble = application.bot_data.get("ml_ensemble")
    db = application.bot_data.get("db")
    if not isinstance(ensemble, MLEnsemble) or not isinstance(db, Database):
        return

    try:
        crypto_symbols = [item.symbol.upper() for item in await db.get_watchlist("crypto")]
        if not crypto_symbols:
            logger.info("Skipping initial ML training because the crypto watchlist is empty")
            return

        logger.info("Starting initial ML training for %d crypto symbols", len(crypto_symbols))
        results = await train_all_models(ensemble, crypto_symbols, db_path=db.db_path)
        success_count = sum(1 for success in results.values() if success)
        logger.info(
            "Initial ML training complete: %d/%d symbols succeeded",
            success_count,
            len(results),
        )
    except Exception:
        logger.exception("Initial ML training failed")


def _telegram_chat_id() -> int | str:
    try:
        return int(settings.telegram_chat_id)
    except ValueError:
        return settings.telegram_chat_id


async def error_handler(update: object, context: Any) -> None:
    import traceback

    del update
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
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("playbook", playbook_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("events", events_command))
    application.add_handler(CommandHandler("corr", corr_command))
    application.add_handler(CommandHandler("regime", regime_command))
    application.add_handler(CommandHandler("funding", funding_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("trade", trade_command))
    application.add_handler(CommandHandler("trades", trades_command))
    application.add_handler(CommandHandler("log", log_command))
    application.add_handler(CommandHandler("close", close_command))
    application.add_handler(CommandHandler("journal", journal_command))
    application.add_handler(CommandHandler("ta", ta_command))
    application.add_handler(CommandHandler("alerts", alerts_command))
    application.add_handler(CommandHandler("accuracy", accuracy_command))
    application.add_handler(CommandHandler("reflect", reflect_command))
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
