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
from src.config import settings
from src.data.aggregator import DataAggregator
from src.events.calibration import format_remaining_drift_status
from src.events.models import DetectedEvent
from src.reflection.evaluator import AccuracyStats, ReflectionEvaluator
from src.sentiment.analyzer import SentimentAnalyzer
from src.storage.database import Database, WatchlistItem

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
            ("BTC", "QQQ"),
            ("BTC", "ETH"),
            ("ETH", "SOL"),
            ("BTC", "SPY"),
            ("NVDA", "AMD"),
            ("NVDA", "BTC"),
        ]
        lines = ["\U0001F517 <b>CROSS-ASSET CORRELATIONS</b>", ""]
        for sym_a, sym_b in key_pairs:
            pair = aggregator.correlation_engine.get_pair_analysis(sym_a, sym_b)
            if pair is None:
                continue
            emoji = "\U0001F7E2" if pair.corr_20d > 0.5 else ("\U0001F534" if pair.corr_20d < -0.2 else "\u26AA")
            lines.append(
                f"{emoji} <b>{sym_a}-{sym_b}</b>: "
                f"20d={pair.corr_20d:+.2f} | 60d={pair.corr_60d:+.2f} | 90d={pair.corr_90d:+.2f}"
            )
            lines.append(
                f"   \u0394={pair.divergence:+.2f} | z={pair.z_score:+.1f} | {html.escape(pair.regime_signal)}"
            )

        movers = aggregator.correlation_engine.get_top_movers(3)
        if movers:
            lines.extend(["", "\U0001F4C8 <b>BIGGEST CORRELATION SHIFTS</b>"])
            for mover in movers:
                lines.append(
                    f"{mover.symbol_a}-{mover.symbol_b}: \u0394={mover.divergence:+.2f} ({mover.regime_signal})"
                )

        lead_lag_pairs = context.bot_data.get("lead_lag_pairs")
        lead_lag_text: str | None = None
        if isinstance(lead_lag_pairs, dict):
            qqq_btc = lead_lag_pairs.get("QQQ:BTC")
            if qqq_btc is not None and hasattr(qqq_btc, "interpretation"):
                lead_lag_text = str(getattr(qqq_btc, "interpretation"))
        if lead_lag_text is None:
            qqq_returns = aggregator.correlation_engine.get_returns("QQQ")
            btc_returns = aggregator.correlation_engine.get_returns("BTC")
            if qqq_returns is not None and btc_returns is not None:
                lead_lag_text = compute_lead_lag(qqq_returns, btc_returns, "QQQ", "BTC").interpretation
        if lead_lag_text:
            lines.extend(["", f"<i>{html.escape(lead_lag_text, quote=False)}</i>"])

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

    regime_name = str(getattr(regime.regime, "value", regime.regime))
    emoji_map = {
        "risk_on": "\U0001F7E2",
        "risk_off": "\U0001F534",
        "divergent": "\U0001F504",
        "volatile": "\U0001F7E1",
        "unknown": "\u26AA",
    }
    corr_text = f"{regime.btc_qqq_corr:+.2f}" if regime.btc_qqq_corr is not None else "N/A"
    divergence_text = f"{regime.divergence_speed:+.2f}" if regime.divergence_speed is not None else "N/A"
    text = (
        f"{emoji_map.get(regime_name, '\u26AA')} <b>MARKET REGIME: {html.escape(regime_name.upper())}</b>\n"
        f"Confidence: {regime.confidence:.0%}\n\n"
        f"{html.escape(regime.description, quote=False)}\n\n"
        f"\U0001F4C8 <b>Stocks:</b> {html.escape(regime.implication_stocks, quote=False)}\n"
        f"\U0001FA99 <b>Crypto:</b> {html.escape(regime.implication_crypto, quote=False)}\n\n"
        f"BTC-QQQ corr: {corr_text}\n"
        f"VIX: {html.escape(regime.vix_level)}\n"
        f"Divergence speed: {divergence_text}"
    )
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

    lines = [f"\U0001F4CA <b>SIGNAL BREAKDOWN: {html.escape(symbol)}</b>", ""]
    for name, value in sorted(composite.signal_breakdown.items(), key=lambda item: abs(item[1]), reverse=True):
        direction = "\U0001F4C8" if value > 0 else ("\U0001F4C9" if value < 0 else "\u26AA")
        bar = "#" * min(20, max(0, int(abs(value) * 20)))
        bar = bar.ljust(20, "-")
        lines.append(f"{direction} <code>{name:<12} {value:+.3f} [{bar}]</code>")

    lines.extend(
        [
            "",
            f"<b>Composite:</b> {html.escape(composite.direction.replace('_', ' ').upper())} ({composite.score:+.2f})",
            f"Agreement: {composite.agreement:.0%}",
            "",
            f"<i>{html.escape(composite.summary, quote=False)}</i>",
        ]
    )
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


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

        emoji_map = {
            "strong_buy": "\U0001F7E2\U0001F7E2",
            "buy": "\U0001F7E2",
            "neutral": "\u26AA",
            "sell": "\U0001F534",
            "strong_sell": "\U0001F534\U0001F534",
        }
        text = (
            f"\U0001F4CA <b>TECHNICAL ANALYSIS: {html.escape(signal.symbol)}</b>\n"
            f"Price: ${signal.current_price:,.2f}\n\n"
            f"{emoji_map.get(signal.overall_signal, '\u26AA')} <b>Signal:</b> "
            f"{html.escape(signal.overall_signal.replace('_', ' ').upper())} "
            f"({signal.confidence:.0%})\n\n"
            f"<b>RSI(14):</b> {signal.rsi_value:.1f}  {html.escape(signal.rsi_signal)}\n"
            f"<b>MACD:</b> {html.escape(signal.macd_signal.replace('_', ' '))} "
            f"(hist: {signal.macd_histogram:+.3f})\n"
            f"<b>Bollinger:</b> {html.escape(signal.bb_position.replace('_', ' '))} "
            f"({'SQUEEZE' if signal.bb_squeeze else 'normal'})\n"
            f"Upper: ${signal.bb_upper:,.2f} | Lower: ${signal.bb_lower:,.2f}\n"
            f"<b>ATR(14):</b> ${signal.atr_value:,.2f} ({signal.atr_pct:.1f}%)  "
            f"{html.escape(signal.volatility)} vol\n"
            f"<b>Trend:</b> {html.escape(signal.trend.replace('_', ' '))} "
            f"(EMA20: ${signal.ema_20:,.2f} | EMA50: ${signal.ema_50:,.2f})\n\n"
            f"<i>{html.escape(signal.summary, quote=False)}</i>"
        )
        await msg.edit_text(text)
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
