from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from src.alerts import AlertPipeline, AlertSeverity
from src.bot.messages import _send_long_message
from src.briefing.formatter import BriefingFormatter
from src.briefing.models import BriefingData
from src.correlation import compute_lead_lag
from src.correlation.models import LeadLagResult, PairCorrelation, RegimeState
from src.config import settings
from src.data.aggregator import DataAggregator
from src.events.calibration import format_remaining_drift_status
from src.events.models import DetectedEvent
from src.reflection.evaluator import AccuracyStats, ReflectionEvaluator
from src.sentiment.analyzer import SentimentAnalyzer
from src.signals.models import CompositeScore
from src.storage.database import Database, WatchlistItem
from src.ta.signals import TASignal

logger = logging.getLogger(__name__)

COMMON_CRYPTO_MAP: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "NEAR": "near",
    "FET": "fetch-ai",
    "TAO": "bittensor",
    "RNDR": "render-token",
    "ONDO": "ondo-finance",
    "AKT": "akash-network",
    "HNT": "helium",
    "AR": "arweave",
    "INJ": "injective-protocol",
    "TIA": "celestia",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
}

REGIME_DISPLAY: dict[str, tuple[str, str, str, str, str]] = {
    "risk_on": (
        "\U0001F7E2",
        "Optimistic",
        "The market is in risk-on mode. Investors are confident and buying.",
        "Growth stocks usually have the wind at their back if this mood holds.",
        "Crypto often benefits too, with stronger coins usually moving fastest.",
    ),
    "risk_off": (
        "\U0001F534",
        "Fearful",
        "The market is in risk-off mode. Investors are selling and moving to safety.",
        "Expect continued selling pressure. Defensive sectors tend to hold up better.",
        "Crypto is at risk of following stocks lower, and the swings can be much sharper.",
    ),
    "divergent": (
        "\U0001F504",
        "Split",
        "Crypto and stocks are telling different stories right now.",
        "Broad stock moves matter less when markets are this split.",
        "Crypto may keep reacting to its own catalysts instead of equities.",
    ),
    "volatile": (
        "\U0001F7E1",
        "Chaotic",
        "Volatility is high and direction is unclear.",
        "Expect sharp reversals and weaker follow-through in stocks.",
        "Crypto can overshoot in both directions when conditions are unstable.",
    ),
    "unknown": (
        "\u26AA",
        "Unclear",
        "Signals are mixed and there is no clear market mood yet.",
        "Stocks do not have a strong broad-market tailwind right now.",
        "Crypto is also lacking a clean cross-asset signal.",
    ),
}


async def owner_only(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    _ = context
    chat = update.effective_chat
    if chat is None or str(chat.id) != settings.telegram_chat_id:
        if update.message is not None:
            await update.message.reply_text("🚫 Unauthorized")
        elif update.callback_query is not None:
            await update.callback_query.answer("Unauthorized", show_alert=True)
        return False
    return True


def _help_text() -> str:
    return (
        "Welcome to AlphaRadar! 📡\n\n"
        f"I'll send you a daily financial briefing every morning at "
        f"{settings.briefing_hour:02d}:{settings.briefing_minute:02d} "
        f"{settings.timezone}.\n\n"
        "Commands:\n"
        "/briefing  Get a briefing right now\n"
        "/watchlist  View your watchlist\n"
        "/add AAPL  Add a stock to watchlist\n"
        "/add btc  Add a crypto to watchlist\n"
        "/remove TSLA  Remove from watchlist\n"
        "/sentiment &lt;text&gt;  Analyze ad hoc sentiment\n"
        "/analyze NVDA  Run Bull/Bear AI debate for one symbol\n"
        "/playbook  Daily actionable breakdown for all notable assets\n"
        "/events  Show latest detected events\n"
        "/corr  Show cross-asset correlation shifts\n"
        "/regime  Show the current market regime\n"
        "/signals NVDA  Show the full signal stack for one symbol\n"
        "/ta NVDA  Compute local technical analysis for one symbol\n"
        "/alerts  Show real-time alert pipeline status\n"
        "/accuracy  Show prediction accuracy and calibration buckets\n"
        "/reflect  Show the latest weekly self-reflection\n"
        "/status  API usage and system health\n"
        "/help  Show this message"
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    chat_id = chat.id if chat is not None else "unknown"
    print(f"DEBUG: Received /start from chat_id={chat_id}")
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    await update.message.reply_text(_help_text())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    await update.message.reply_text(_help_text())


async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    chat = update.effective_chat
    if chat is None:
        return

    msg = await update.message.reply_text("⏳ Fetching data...")
    aggregator = _aggregator_from_context(context)
    formatter = context.bot_data["formatter"]
    db = _db_from_context(context)

    try:
        data = await aggregator.gather_briefing_data()
        _store_latest_briefing(context, data)
        briefing_html = formatter.format_on_demand(data)
        await _send_long_message(
            context.bot,
            chat.id,
            briefing_html,
            initial_message=msg,
        )
        status = "partial" if data.failed_sections else "sent"
        await db.log_briefing("on_demand", len(briefing_html), status, sections=data.failed_sections)
    except Exception as exc:
        logger.exception("On-demand briefing failed")
        error_text = formatter.format_error(str(exc))
        await msg.edit_text(error_text)
        await db.log_briefing("on_demand", 0, "failed", error=str(exc))


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    db = _db_from_context(context)
    stocks = await db.get_watchlist(asset_type="stock")
    crypto = await db.get_watchlist(asset_type="crypto")

    text = (
        "📋 <b>YOUR WATCHLIST</b>\n\n"
        f"<b>Stocks ({len(stocks)}):</b>\n"
        f"{_symbol_grid(stocks)}\n\n"
        f"<b>Crypto ({len(crypto)}):</b>\n"
        f"{_symbol_grid(crypto)}"
    )
    await update.message.reply_text(text)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    if not context.args:
        await update.message.reply_text("Usage: /analyze &lt;symbol&gt;")
        return

    symbol = context.args[0].strip().upper()
    if not symbol:
        await update.message.reply_text("Usage: /analyze &lt;symbol&gt;")
        return

    msg = await update.message.reply_text(
        f"\U0001F916 Running agent analysis for {html.escape(symbol)}..."
    )
    aggregator = _aggregator_from_context(context)
    formatter = _formatter_from_context(context)

    try:
        data = await aggregator.gather_briefing_data(include_agent_analysis=False)
        _store_latest_briefing(context, data)
        available_symbols = {
            item.symbol.upper()
            for item in [*data.stock_prices, *data.crypto_prices]
        }
        if symbol not in available_symbols:
            await msg.edit_text(
                f"\u26A0\uFE0F {html.escape(symbol)} is not in the current watchlist data."
            )
            return

        analysis = await aggregator.run_agent_analysis(
            data,
            forced_symbols=[symbol],
            source="analyze",
        )
        if analysis is None:
            await msg.edit_text(
                "\u26A0\uFE0F Agent pipeline unavailable. Check GEMINI_API_KEY or Groq status."
            )
            return

        response = formatter.format_agent_verdict(symbol, analysis)
        if response is None:
            await msg.edit_text(
                f"\u26A0\uFE0F Agent pipeline ran but produced no verdict for {html.escape(symbol)}."
            )
            return
        await _send_long_message(
            context.bot,
            update.effective_chat.id if update.effective_chat is not None else settings.telegram_chat_id,
            response,
            initial_message=msg,
        )
    except Exception as exc:
        logger.exception("Analyze command failed for %s", symbol)
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Analysis failed.\n<i>{escaped_error[:250]}</i>")


async def playbook_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a full daily playbook with actionable context for every notable asset."""
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    chat = update.effective_chat
    if chat is None:
        return

    msg = await update.message.reply_text("\U0001F4CB Building daily playbook...")
    aggregator = _aggregator_from_context(context)

    try:
        data = await aggregator.gather_briefing_data(include_agent_analysis=False)
        _store_latest_briefing(context, data)
        playbook_text, llm_calls, duration = await aggregator.generate_playbook(data)
        message = _format_playbook_message(
            generated_at=data.generated_at,
            playbook_text=playbook_text,
            llm_calls=llm_calls,
            duration=duration,
        )
        await _send_long_message(
            context.bot,
            chat.id,
            message,
            initial_message=msg,
        )
    except Exception as exc:
        logger.exception("Playbook command failed")
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Playbook failed.\n<i>{escaped_error[:250]}</i>")


async def events_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    msg = await update.message.reply_text("\U0001F3F7 Loading latest events...")

    try:
        data = await _latest_briefing_data(context)
        events = _relevant_events(data)
        lines = ["\U0001F3F7 <b>DETECTED EVENTS</b>", ""]

        if events:
            for event in events:
                symbols = ", ".join(event.related_symbols) or "?"
                lines.append(
                    f"\U0001F4CA <b>{html.escape(symbols, quote=False)}</b> "
                    f"{html.escape(event.event_type.value, quote=False)} "
                    f"({event.confidence:.0%} confidence)"
                )
                if event.expected_car_pct is not None and event.expected_car_window is not None:
                    lines.append(
                        f"   Hist: {event.expected_car_pct:+.1f}% over {event.expected_car_window}"
                    )
                if event.actual_move_pct is not None:
                    status = format_remaining_drift_status(event.event_type, event.actual_move_pct)
                    status_text = status[:1].upper() + status[1:] if status else "No status"
                    lines.append(f"   Current: {event.actual_move_pct:+.1f}%  {status_text}")
                lines.append("")
        else:
            lines.append("No calibrated events detected in the latest briefing data.")
            lines.append("")

        no_event_symbols = _symbols_without_events(data, events)
        if no_event_symbols:
            lines.append(f"No event detected: {', '.join(no_event_symbols)}")

        await _send_long_message(
            context.bot,
            update.effective_chat.id if update.effective_chat is not None else settings.telegram_chat_id,
            "\n".join(lines).strip(),
            initial_message=msg,
        )
    except Exception as exc:
        logger.exception("Events command failed")
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Events lookup failed.\n<i>{escaped_error[:250]}</i>")


async def corr_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    msg = await update.message.reply_text("\U0001F4CA Computing correlations...")
    aggregator = _aggregator_from_context(context)

    try:
        if not aggregator.correlation_engine.compute_all_windows():
            await aggregator.update_correlation_engine(persist=False)

        key_pairs = [
            ("BTC", "QQQ", "BTC & tech stocks"),
            ("BTC", "ETH", "BTC & ETH"),
            ("BTC", "SPY", "BTC & S&P 500"),
            ("ETH", "SOL", "ETH & SOL"),
        ]
        lines = ["\U0001F517 <b>MARKET CONNECTIONS</b>", ""]
        for sym_a, sym_b, label in key_pairs:
            pair = aggregator.correlation_engine.get_pair_analysis(sym_a, sym_b)
            if pair is None:
                continue
            lines.append(_format_correlation_pair_line(label, pair))

        movers = aggregator.correlation_engine.get_top_movers(3)
        notable = [mover for mover in movers if abs(mover.divergence) > 0.3]
        if notable:
            lines.extend(["", "<b>Notable shifts:</b>"])
            for mover in notable:
                lines.append(_format_correlation_shift_line(mover))

        lead_lag_pairs = context.bot_data.get("lead_lag_pairs")
        lead_lag_result: LeadLagResult | None = None
        if isinstance(lead_lag_pairs, dict):
            qqq_btc = lead_lag_pairs.get("QQQ:BTC")
            if isinstance(qqq_btc, LeadLagResult):
                lead_lag_result = qqq_btc
        if lead_lag_result is None:
            qqq_returns = aggregator.correlation_engine.get_returns("QQQ")
            btc_returns = aggregator.correlation_engine.get_returns("BTC")
            if qqq_returns is not None and btc_returns is not None:
                lead_lag_result = compute_lead_lag(qqq_returns, btc_returns, "QQQ", "BTC")
        if lead_lag_result is not None:
            lines.extend(["", f"\U0001F4CA {_format_lead_lag_line(lead_lag_result)}"])

        await msg.edit_text("\n".join(lines), parse_mode="HTML")
    except Exception as exc:
        logger.exception("Correlation command failed")
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Correlation lookup failed.\n<i>{escaped_error[:250]}</i>")


async def regime_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    regime = context.bot_data.get("current_regime")
    if regime is None:
        data = await _latest_briefing_data(context)
        regime = data.regime_state
    if regime is None:
        await update.message.reply_text("Regime detection not yet run.")
        return

    text = _format_regime_text(regime)
    await update.message.reply_text(text, parse_mode="HTML")


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    if not context.args:
        await update.message.reply_text("Usage: /signals &lt;symbol&gt;")
        return

    symbol = context.args[0].strip().upper()
    data = await _latest_briefing_data(context)
    composite = data.composite_scores.get(symbol)
    if composite is None:
        await update.message.reply_text(f"No composite data for {html.escape(symbol)}.")
        return

    await update.message.reply_text(
        _format_signals_text(symbol, composite),
        parse_mode="HTML",
    )


async def ta_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    if not context.args:
        await update.message.reply_text("Usage: /ta &lt;symbol&gt;")
        return

    symbol = context.args[0].strip().upper()
    if not symbol:
        await update.message.reply_text("Usage: /ta &lt;symbol&gt;")
        return

    msg = await update.message.reply_text(f"\U0001F4CA Computing TA for {html.escape(symbol)}...")
    aggregator = _aggregator_from_context(context)

    try:
        signal = await aggregator.get_ta_signal(symbol)
        if signal is None:
            await msg.edit_text(
                f"\u26A0\uFE0F Could not fetch enough candle data for {html.escape(symbol)}."
            )
            return

        await msg.edit_text(_format_ta_text(signal), parse_mode="HTML")
    except Exception as exc:
        logger.exception("TA command failed for %s", symbol)
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F TA failed.\n<i>{escaped_error[:250]}</i>")


async def alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    pipeline = context.application.bot_data.get("alert_pipeline")
    if not isinstance(pipeline, AlertPipeline):
        await update.message.reply_text("Alert pipeline not initialized.")
        return

    status = pipeline.status()
    recent_alerts = status.get("recent_alerts", [])
    recent_lines = [
        f"- {html.escape(alert.symbol)} {alert.change_pct:+.1f}% ({alert.severity.value})"
        for alert in recent_alerts
        if hasattr(alert, "symbol") and hasattr(alert, "change_pct") and hasattr(alert, "severity")
    ]

    text = (
        "\U0001F514 <b>ALERT STATUS</b>\n\n"
        f"Alerts this hour: {status['alerts_this_hour']}/{status['global_limit']}\n"
        f"Quiet hours: {'Active \U0001F319' if status['quiet_hours'] else 'Off'}\n"
        f"Assets on cooldown: {status['assets_on_cooldown']}\n\n"
        f"Stock thresholds: {pipeline.THRESHOLDS['stock'][AlertSeverity.INFO]:.0f}% / "
        f"{pipeline.THRESHOLDS['stock'][AlertSeverity.WARNING]:.0f}% / "
        f"{pipeline.THRESHOLDS['stock'][AlertSeverity.CRITICAL]:.0f}%\n"
        f"Crypto thresholds: {pipeline.THRESHOLDS['crypto'][AlertSeverity.INFO]:.0f}% / "
        f"{pipeline.THRESHOLDS['crypto'][AlertSeverity.WARNING]:.0f}% / "
        f"{pipeline.THRESHOLDS['crypto'][AlertSeverity.CRITICAL]:.0f}%"
    )
    if recent_lines:
        text += "\n\n<b>Recent alerts:</b>\n" + "\n".join(recent_lines[:5])
    await update.message.reply_text(text)


async def accuracy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    evaluator = _reflection_evaluator_from_context(context)
    snapshot = await evaluator.accuracy_snapshot()

    def fmt(stats: AccuracyStats) -> str:
        if stats.total == 0:
            return "No data"
        return f"{stats.correct}/{stats.total} ({stats.accuracy:.1f}%)"

    text = (
        "\U0001F4CA <b>PREDICTION ACCURACY</b>\n\n"
        f"<b>All-time:</b> {fmt(snapshot['all_time'])}\n"
        f"<b>Last 7 days:</b> {fmt(snapshot['last_7d'])}\n\n"
        f"<b>High confidence (70%+):</b> {fmt(snapshot['high_confidence'])}\n"
        f"<b>Low confidence (&lt;70%):</b> {fmt(snapshot['low_confidence'])}\n\n"
        "<i>Predictions logged from briefings, /analyze, and /playbook</i>"
    )
    await update.message.reply_text(text)


async def reflect_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    evaluator = _reflection_evaluator_from_context(context)
    reflection = await evaluator.latest_reflection()
    if reflection is None:
        await update.message.reply_text("No reflections yet. First one runs Sunday at 8:00 AM UTC.")
        return

    text = (
        "\U0001F9E0 <b>WEEKLY REFLECTION</b>\n"
        f"Period: {html.escape(reflection.period_start[:10])} to {html.escape(reflection.period_end[:10])}\n"
        f"Predictions: {reflection.total_predictions} | "
        f"Accuracy: {(reflection.accuracy_1d or 0.0):.1f}%\n\n"
        f"{html.escape(reflection.reflection_text, quote=False)}"
    )
    await _send_long_message(context.bot, update.effective_chat.id, text)


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    if not context.args:
        await update.message.reply_text("Usage: /add &lt;symbol&gt;")
        return

    raw_symbol = context.args[0].strip()
    if not raw_symbol:
        await update.message.reply_text("Usage: /add &lt;symbol&gt;")
        return

    symbol = raw_symbol.upper()
    is_stock_like = symbol.isalpha() and 1 <= len(symbol) <= 5 and raw_symbol == symbol
    is_known_crypto = raw_symbol.islower() or symbol in COMMON_CRYPTO_MAP

    if is_stock_like and symbol in COMMON_CRYPTO_MAP:
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("📈 Stock", callback_data=f"asset_type:stock:{symbol}"),
                    InlineKeyboardButton("🪙 Crypto", callback_data=f"asset_type:crypto:{symbol}"),
                ]
            ]
        )
        await update.message.reply_text(
            f'Is "{symbol}" a stock or crypto?',
            reply_markup=keyboard,
        )
        return

    if is_known_crypto:
        result = await _add_crypto(symbol=symbol, raw_symbol=raw_symbol, context=context)
        await update.message.reply_text(result)
        return

    if is_stock_like:
        result = await _add_stock(symbol=symbol, context=context)
        await update.message.reply_text(result)
        return

    await update.message.reply_text("Could not determine asset type. Use a ticker symbol like AAPL or btc.")


async def handle_asset_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    query = update.callback_query
    if query is None or query.data is None:
        return
    await query.answer()

    try:
        _, asset_type, symbol = query.data.split(":", 2)
    except ValueError:
        await query.edit_message_text("Invalid selection.")
        return

    if asset_type == "stock":
        result = await _add_stock(symbol=symbol.upper(), context=context)
    elif asset_type == "crypto":
        result = await _add_crypto(symbol=symbol.upper(), raw_symbol=symbol.lower(), context=context)
    else:
        result = "Invalid asset type."
    await query.edit_message_text(result)


async def remove_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    if not context.args:
        await update.message.reply_text("Usage: /remove &lt;symbol&gt;")
        return

    symbol = context.args[0].strip().upper()
    db = _db_from_context(context)
    removed_stock = await db.remove_from_watchlist(symbol, "stock")
    removed_crypto = await db.remove_from_watchlist(symbol, "crypto")

    if removed_stock or removed_crypto:
        await update.message.reply_text(f"✅ Removed {symbol} from watchlist")
    else:
        await update.message.reply_text(f"❌ {symbol} not found in your watchlist")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    aggregator = _aggregator_from_context(context)
    db = _db_from_context(context)
    health = await aggregator.provider_health()
    cg_used, cg_limit = await aggregator.coingecko.get_remaining_budget()
    stocks = await db.get_watchlist("stock")
    crypto = await db.get_watchlist("crypto")
    sentiment_cache_count = await db.get_sentiment_cache_count()
    model_status = (
        "Loaded (ONNX, CPU)"
        if aggregator.sentiment_analyzer.is_available()
        else "Not available"
    )
    agent_status = aggregator.agent_pipeline_status_text()
    ws_state = _websocket_status_text(context)
    next_briefing = _next_briefing_text()
    uptime = _uptime_text(context)

    text = (
        "🔧 <b>ALPHARADAR STATUS</b>\n\n"
        f"📡 Finnhub: {_status_label(health['finnhub'])}\n"
        f"📡 CoinGecko: {_status_label(health['coingecko'])} "
        f"({cg_used:,}/{cg_limit:,} calls this month)\n"
        f"📡 DeFi Llama: {_status_label(health['defillama'])}\n"
        f"📡 SEC EDGAR: {_status_label(health['edgar'])}\n"
        f"🧠 FinBERT: {model_status}\n"
        f"🤖 Agent Pipeline: {agent_status}\n"
        f"🔌 WebSockets: {ws_state}\n"
        f"📊 Sentiment cache: {sentiment_cache_count:,} headlines cached\n\n"
        f"📋 Watchlist: {len(stocks)} stocks, {len(crypto)} crypto\n"
        f"📅 Next briefing: {next_briefing}\n"
        f"🕐 Uptime: {uptime}"
    )
    await update.message.reply_text(text)


async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    if not context.args:
        await update.message.reply_text("Usage: /sentiment &lt;text to analyze&gt;")
        return

    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Usage: /sentiment &lt;text to analyze&gt;")
        return

    msg = await update.message.reply_text("🧠 Analyzing...")
    analyzer = _sentiment_analyzer_from_context(context)

    try:
        if not analyzer.is_available():
            await analyzer.initialize()

        results = await analyzer.analyze([text])
        result = results[0]
        indicator = (
            "📈" if result.label == "positive" else "📉" if result.label == "negative" else "📊"
        )
        escaped_text = html.escape(text, quote=False)
        response = (
            "🧠 <b>Sentiment Analysis</b>\n\n"
            f'"{escaped_text}"\n\n'
            f"Label: {result.label} ({indicator})\n"
            f"Score: {result.score:+.2f}\n"
            f"Confidence: {result.confidence:.1%}\n\n"
            "Breakdown:\n"
            f"  Positive: {result.positive:.1%}\n"
            f"  Negative: {result.negative:.1%}\n"
            f"  Neutral: {result.neutral:.1%}"
        )
        await msg.edit_text(response)
    except Exception as exc:
        logger.exception("Sentiment command failed")
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"⚠️ Sentiment analysis failed.\n<i>{escaped_error[:250]}</i>")


async def _add_stock(symbol: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    aggregator = _aggregator_from_context(context)
    db = _db_from_context(context)

    try:
        quote = await aggregator.finnhub.get_quote(symbol)
    except Exception:
        logger.exception("Stock validation failed for %s", symbol)
        return f"❌ Could not validate stock symbol {symbol}."

    if quote.current_price <= 0 and quote.previous_close is None:
        return f"❌ Invalid stock symbol: {symbol}"

    added = await db.add_to_watchlist(symbol=symbol, asset_type="stock", display_name=symbol)
    if not added:
        return f"⚠️ {symbol} is already in your watchlist"
    return f"✅ Added {symbol} ({symbol}) to stocks watchlist"


async def _add_crypto(symbol: str, raw_symbol: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    aggregator = _aggregator_from_context(context)
    db = _db_from_context(context)

    coin_id = _resolve_coin_id(symbol=symbol, raw_symbol=raw_symbol)
    if coin_id is None:
        return f"❌ Could not map {symbol} to a CoinGecko ID."

    exists = await aggregator.coingecko.coin_id_exists(coin_id)
    if not exists:
        return f"❌ Invalid crypto symbol or unsupported CoinGecko ID: {symbol}"

    display_name = coin_id.replace("-", " ").title()
    added = await db.add_to_watchlist(
        symbol=symbol,
        asset_type="crypto",
        display_name=display_name,
        coingecko_id=coin_id,
    )
    if not added:
        return f"⚠️ {symbol} is already in your watchlist"
    return f"✅ Added {symbol} ({display_name}) to crypto watchlist"


def _resolve_coin_id(symbol: str, raw_symbol: str) -> str | None:
    if symbol in COMMON_CRYPTO_MAP:
        return COMMON_CRYPTO_MAP[symbol]
    if raw_symbol.islower():
        return raw_symbol
    return raw_symbol.lower() if raw_symbol else None


def _symbol_grid(items: list[WatchlistItem]) -> str:
    if not items:
        return "-"
    symbols = [item.symbol for item in items]
    lines = ["  ".join(symbols[idx : idx + 5]) for idx in range(0, len(symbols), 5)]
    return "\n".join(lines)


def _status_label(ok: bool) -> str:
    return "✅ Operational" if ok else "❌ Degraded"


def _websocket_status_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    ws_tasks = context.application.bot_data.get("ws_tasks")
    if not isinstance(ws_tasks, dict):
        return "Not initialized"

    parts: list[str] = []
    for name, label in (("finnhub", "Finnhub"), ("binance", "Binance")):
        task = ws_tasks.get(name)
        if task is None:
            parts.append(f"{label} idle")
            continue
        if hasattr(task, "cancelled") and task.cancelled():
            parts.append(f"{label} cancelled")
            continue
        if hasattr(task, "done") and task.done():
            try:
                exc = task.exception() if hasattr(task, "exception") else None
            except Exception:
                exc = None
            parts.append(f"{label} stopped" if exc else f"{label} done")
            continue
        parts.append(f"{label} running")
    return " | ".join(parts)


def _next_briefing_text() -> str:
    tz = ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    next_run = now.replace(
        hour=settings.briefing_hour,
        minute=settings.briefing_minute,
        second=0,
        microsecond=0,
    )
    if next_run <= now:
        next_run = next_run + timedelta(days=1)

    delta_days = (next_run.date() - now.date()).days
    day_label = "Today" if delta_days == 0 else "Tomorrow" if delta_days == 1 else next_run.strftime("%A")
    hour_12 = next_run.hour % 12 or 12
    tz_name = next_run.strftime("%Z")
    time_text = f"{hour_12}:{next_run.strftime('%M %p')}"
    if tz_name:
        time_text = f"{time_text} {tz_name}"
    return f"{day_label} {time_text}"


def _uptime_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    started_at = context.application.bot_data.get("started_at")
    if not isinstance(started_at, datetime):
        return "unknown"
    delta = datetime.now(tz=started_at.tzinfo) - started_at
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    return f"{hours}h {minutes}m"


def _db_from_context(context: ContextTypes.DEFAULT_TYPE) -> Database:
    db = context.bot_data.get("db")
    if not isinstance(db, Database):
        raise RuntimeError("Database instance missing from bot_data")
    return db


def _aggregator_from_context(context: ContextTypes.DEFAULT_TYPE) -> DataAggregator:
    aggregator = context.bot_data.get("aggregator")
    if not isinstance(aggregator, DataAggregator):
        raise RuntimeError("DataAggregator missing from bot_data")
    return aggregator


def _formatter_from_context(context: ContextTypes.DEFAULT_TYPE) -> BriefingFormatter:
    formatter = context.bot_data.get("formatter")
    if not isinstance(formatter, BriefingFormatter):
        raise RuntimeError("BriefingFormatter missing from bot_data")
    return formatter


def _sentiment_analyzer_from_context(context: ContextTypes.DEFAULT_TYPE) -> SentimentAnalyzer:
    analyzer = context.bot_data.get("sentiment_analyzer")
    if isinstance(analyzer, SentimentAnalyzer):
        return analyzer
    aggregator = _aggregator_from_context(context)
    context.bot_data["sentiment_analyzer"] = aggregator.sentiment_analyzer
    return aggregator.sentiment_analyzer


def _reflection_evaluator_from_context(context: ContextTypes.DEFAULT_TYPE) -> ReflectionEvaluator:
    evaluator = context.bot_data.get("reflection_evaluator")
    if isinstance(evaluator, ReflectionEvaluator):
        return evaluator
    aggregator = _aggregator_from_context(context)
    context.bot_data["reflection_evaluator"] = aggregator.reflection_evaluator
    return aggregator.reflection_evaluator


async def _latest_briefing_data(context: ContextTypes.DEFAULT_TYPE) -> BriefingData:
    cached = context.bot_data.get("latest_briefing_data")
    if isinstance(cached, BriefingData):
        return cached

    aggregator = _aggregator_from_context(context)
    data = await aggregator.gather_briefing_data(include_agent_analysis=False)
    _store_latest_briefing(context, data)
    return data


def _store_latest_briefing(context: ContextTypes.DEFAULT_TYPE, data: BriefingData) -> None:
    context.bot_data["latest_briefing_data"] = data
    context.bot_data["current_regime"] = data.regime_state
    context.bot_data["composite_scores"] = data.composite_scores
    context.bot_data["lead_lag_pairs"] = data.lead_lag_pairs


def _format_playbook_message(
    *,
    generated_at: datetime,
    playbook_text: str,
    llm_calls: int,
    duration: float,
) -> str:
    header = (
        "\U0001F4CB <b>ALPHARADAR DAILY PLAYBOOK</b>\n"
        f"{generated_at.strftime('%A')}, {generated_at.strftime('%B')} "
        f"{generated_at.day} {generated_at.year}"
    )
    body = html.escape(playbook_text.strip(), quote=False) if playbook_text.strip() else "No playbook generated."
    footer = ""
    if llm_calls > 0:
        label = "call" if llm_calls == 1 else "calls"
        footer = f"<i>\U0001F9E0 {llm_calls} LLM {label} in {duration:.1f}s</i>"
    return "\n\n".join(part for part in [header, body, footer] if part)


def _format_signals_text(symbol: str, composite: CompositeScore) -> str:
    direction_label = _signal_direction_label(composite.direction)
    emoji = _signal_direction_emoji(composite.direction)
    overall = f"{emoji} {direction_label}".strip()
    lines = [f"\U0001F4CA <b>{html.escape(symbol)}</b>  Overall: {html.escape(overall)}", ""]

    component_lines = [
        _signal_component_line(name, value)
        for name, value in sorted(
            composite.signal_breakdown.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    ]
    lines.extend(line for line in component_lines if line)

    regime_line = _signal_regime_line(composite.regime_context)
    if regime_line:
        lines.extend(["", regime_line])

    lines.extend(["", _signal_agreement_line(composite.agreement), _signal_action_line(composite)])
    return "\n".join(lines)


def _format_correlation_pair_line(label: str, pair: PairCorrelation) -> str:
    return f"{html.escape(label)}: {_correlation_relationship_text(pair)}"


def _format_correlation_shift_line(pair: PairCorrelation) -> str:
    if pair.divergence > 0:
        shift = "suddenly moving together"
    else:
        shift = "suddenly drifting apart"
    if pair.regime_signal == "anomalous":
        shift += " (unusual move)"
    return f"{html.escape(pair.symbol_a)}-{html.escape(pair.symbol_b)}: {shift}"


def _format_lead_lag_line(result: LeadLagResult) -> str:
    if result.peak_lag == 0:
        return (
            f"{html.escape(result.symbol_a)} and {html.escape(result.symbol_b)} "
            "have been moving at roughly the same time lately."
        )

    if result.peak_lag > 0:
        leader = result.symbol_a
        follower = result.symbol_b
        lag_days = result.peak_lag
    else:
        leader = result.symbol_b
        follower = result.symbol_a
        lag_days = abs(result.peak_lag)
    return (
        f"{html.escape(leader)} has tended to move before "
        f"{html.escape(follower)} by about {lag_days} day(s) recently."
    )


def _format_regime_text(regime: RegimeState) -> str:
    regime_name = str(getattr(regime.regime, "value", regime.regime))
    emoji, mood, description, stocks_line, crypto_line = REGIME_DISPLAY.get(
        regime_name,
        REGIME_DISPLAY["unknown"],
    )
    duration_line = _regime_duration_line(regime_name)
    lines = [
        f"{emoji} <b>MARKET MOOD: {html.escape(mood)}</b>",
        "",
        html.escape(description, quote=False),
        "",
        f"\U0001F4C8 <b>Stocks:</b> {html.escape(stocks_line, quote=False)}",
        f"\U0001FA99 <b>Crypto:</b> {html.escape(crypto_line, quote=False)}",
    ]
    if duration_line:
        lines.extend(["", html.escape(duration_line, quote=False)])
    return "\n".join(lines)


def _format_ta_text(signal: TASignal) -> str:
    direction_label = _signal_direction_label(signal.overall_signal)
    emoji = _signal_direction_emoji(signal.overall_signal)
    overall = f"{emoji} {direction_label}".strip()
    lines = [
        f"\U0001F4CA <b>{html.escape(signal.symbol)}</b>  {html.escape(overall)}",
        "",
        f"Price: {_format_price(signal.current_price)}",
        f"Trend: {_ta_trend_text(signal)}",
        f"Momentum: {_ta_momentum_text(signal)}",
        f"Volatility: {_ta_volatility_text(signal)}",
        (
            f"Key levels: Support ~{_format_price_level(signal.bb_lower)} | "
            f"Resistance ~{_format_price_level(signal.ema_20)}"
        ),
    ]
    if signal.bb_squeeze:
        lines.append("Setup: Volatility is compressed, so a bigger move may be close.")
    lines.extend(["", _ta_takeaway_text(signal)])
    return "\n".join(lines)


def _relevant_events(data: BriefingData) -> list[DetectedEvent]:
    return [
        event
        for event in data.event_data
        if event.event_type.value not in {"general_news", "unknown"} and event.confidence >= 0.6
    ]


def _symbols_without_events(data: BriefingData, events: list[DetectedEvent]) -> list[str]:
    all_symbols = sorted({item.symbol.upper() for item in [*data.stock_prices, *data.crypto_prices]})
    event_symbols = {
        symbol.upper()
        for event in events
        for symbol in event.related_symbols
    }
    return [symbol for symbol in all_symbols if symbol not in event_symbols]


def _signal_direction_emoji(direction: str) -> str:
    return {
        "strong_buy": "\U0001F7E2\U0001F7E2",
        "buy": "\U0001F7E2",
        "neutral": "",
        "sell": "\U0001F534",
        "strong_sell": "\U0001F534\U0001F534",
    }.get(direction, "")


def _signal_direction_label(direction: str) -> str:
    return {
        "strong_buy": "Strong Bullish",
        "buy": "Bullish",
        "neutral": "Neutral",
        "sell": "Bearish",
        "strong_sell": "Strong Bearish",
    }.get(direction, direction.replace("_", " ").title())


def _signal_component_line(name: str, value: float) -> str:
    direction_key = "positive" if value > 0.02 else "negative" if value < -0.02 else "neutral"
    descriptions = {
        "sentiment": {
            "positive": "\U0001F4C8 News and sentiment lean positive.",
            "negative": "\U0001F4C9 News and sentiment lean negative.",
            "neutral": "News and sentiment are not giving a strong push either way.",
        },
        "ta": {
            "positive": "Technical trend looks constructive with improving momentum.",
            "negative": "Technical trend is still bearish with weak momentum.",
            "neutral": "Technical trend is mixed with no clear direction yet.",
        },
        "event": {
            "positive": "The latest event has historically leaned supportive for price.",
            "negative": "The latest event has historically leaned negative for price.",
            "neutral": "There is no strong event signal right now.",
        },
        "correlation": {
            "positive": "The broader market backdrop is helping rather than hurting.",
            "negative": "The broader market backdrop is still a headwind.",
            "neutral": "The broader market relationship looks fairly normal right now.",
        },
        "agent": {
            "positive": "The AI debate leaned bullish overall.",
            "negative": "The AI debate leaned bearish overall.",
            "neutral": "The AI debate was inconclusive.",
        },
    }
    return descriptions.get(name, {}).get(direction_key, "")


def _signal_regime_line(regime_context: str) -> str:
    regime_key = regime_context or "unknown"
    emoji, mood, _, _, _ = REGIME_DISPLAY.get(regime_key, REGIME_DISPLAY["unknown"])
    explanations = {
        "risk_on": "supportive environment for risk assets",
        "risk_off": "defensive environment",
        "divergent": "stocks and crypto are not aligned",
        "volatile": "fast swings and weaker follow-through",
        "unknown": "mixed backdrop",
        "default": "mixed backdrop",
    }
    return (
        f"Market mood: {emoji} {html.escape(mood)}"
        f" - {html.escape(explanations.get(regime_key, 'mixed backdrop'), quote=False)}"
    )


def _signal_agreement_line(agreement: float) -> str:
    if agreement >= 0.8:
        return "Most signals are aligned, so conviction is relatively strong."
    if agreement >= 0.6:
        return "Most signals point the same way, but conviction is only moderate."
    return "Signals are split 50/50, so there is no strong conviction either way."


def _signal_action_line(composite: CompositeScore) -> str:
    if composite.direction == "neutral" or composite.agreement < 0.6:
        return "Wait for clearer direction before acting."
    if composite.direction in {"buy", "strong_buy"}:
        return "Bias is positive, but it is still better to wait for follow-through than chase it."
    return "Bias is negative, so treat bounces cautiously until the picture improves."


def _correlation_relationship_text(pair: PairCorrelation) -> str:
    corr = pair.corr_20d
    if corr < -0.3:
        description = "moving in opposite directions"
    elif abs(corr) > 0.8:
        description = "tightly linked - they usually move together"
    elif abs(corr) > 0.5:
        description = "moderately connected"
    elif abs(corr) > 0.2:
        description = "loosely connected"
    else:
        description = "moving independently right now"

    if pair.regime_signal == "coupling":
        description += " (connection strengthening)"
    elif pair.regime_signal == "decoupling":
        description += " (connection weakening)"
    elif pair.regime_signal == "anomalous":
        description += " (unusual shift)"
    return description


def _regime_duration_line(regime_name: str) -> str:
    if regime_name == "risk_off":
        return "This mood can last from several days to a few weeks. Watch for volatility to cool before getting aggressive."
    if regime_name == "risk_on":
        return "As long as volatility stays calm, dips are more likely to get bought than sold."
    if regime_name == "volatile":
        return "Conditions like this can reverse quickly, so smaller risk and more patience usually make sense."
    if regime_name == "divergent":
        return "Wait for stocks and crypto to confirm each other before trusting broad market signals too much."
    return "Wait for a cleaner trend before leaning too heavily in one direction."


def _ta_trend_text(signal: TASignal) -> str:
    return {
        "strong_up": "Strong uptrend - price is above both key moving averages.",
        "up": "Uptrend - price is above the short-term trend line.",
        "neutral": "Sideways - there is no clean trend right now.",
        "down": "Downtrend - price is below the short-term trend line.",
        "strong_down": "Downtrend - price is below both key moving averages.",
    }.get(signal.trend, "Trend is unclear right now.")


def _ta_momentum_text(signal: TASignal) -> str:
    if signal.rsi_value < 30:
        return "Oversold - it may be due for a bounce."
    if signal.rsi_value > 70:
        return "Overbought - it may be due for a pullback."
    if signal.macd_signal == "bullish_cross":
        return "Starting to recover, but the turn is not fully confirmed yet."
    if signal.macd_signal == "bearish_cross":
        return "Momentum is rolling over and the downside pressure is building."
    if signal.macd_signal == "bullish":
        return "Momentum is positive, but it is not accelerating strongly."
    return "Momentum is still weak or flat."


def _ta_volatility_text(signal: TASignal) -> str:
    return {
        "low": "Low - moves have been relatively calm, which can precede a breakout.",
        "normal": "Normal - price swings are in a typical range.",
        "high": "High - expect larger swings in either direction.",
    }.get(signal.volatility, "Volatility is unclear.")


def _ta_takeaway_text(signal: TASignal) -> str:
    resistance = _format_price_level(signal.ema_20)
    if signal.overall_signal == "neutral" and signal.trend in {"down", "strong_down"} and signal.macd_signal in {"bullish", "bullish_cross"}:
        return (
            "Overall the technical picture is mixed. The downtrend is still in place, "
            f"but momentum is starting to improve. Wait for confirmation above {resistance}."
        )
    if signal.overall_signal in {"buy", "strong_buy"}:
        return (
            "Overall the technical picture is constructive. Trend and momentum are leaning positive, "
            f"but holding above {resistance} would strengthen the case."
        )
    if signal.overall_signal in {"sell", "strong_sell"}:
        return (
            "Overall the technical picture is weak. Trend and momentum both lean lower, "
            f"so a move back above {resistance} is the first sign conditions are improving."
        )
    return (
        "Overall the technical picture is mixed, with no strong edge either way yet. "
        f"Wait for a cleaner move above or below {resistance} before assuming a trend."
    )


def _format_price(value: float) -> str:
    if value >= 1_000:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.2f}"
    return f"${value:.4g}"


def _format_price_level(value: float) -> str:
    if value >= 1:
        return f"${value:,.0f}"
    return f"${value:.4g}"
