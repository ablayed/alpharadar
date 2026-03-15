from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from src.briefing.formatter import BriefingFormatter, _send_briefing_parts
from src.briefing.models import BriefingData
from src.config import settings
from src.data.aggregator import DataAggregator
from src.events.calibration import format_remaining_drift_status
from src.events.models import DetectedEvent
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
        "/events  Show latest detected events\n"
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
        parts = _send_briefing_parts(briefing_html)
        await msg.edit_text(parts[0])
        for part in parts[1:]:
            await context.bot.send_message(chat_id=chat.id, text=part)
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

        analysis = await aggregator.run_agent_analysis(data, forced_symbols=[symbol])
        if analysis is None:
            await msg.edit_text(
                "\u26A0\uFE0F Agent pipeline unavailable. Check GEMINI_API_KEY or Gemini status."
            )
            return

        response = formatter.format_agent_verdict(symbol, analysis)
        if response is None:
            await msg.edit_text(
                f"\u26A0\uFE0F Agent pipeline ran but produced no verdict for {html.escape(symbol)}."
            )
            return
        await msg.edit_text(response)
    except Exception as exc:
        logger.exception("Analyze command failed for %s", symbol)
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Analysis failed.\n<i>{escaped_error[:250]}</i>")


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

        await msg.edit_text("\n".join(lines).strip())
    except Exception as exc:
        logger.exception("Events command failed")
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Events lookup failed.\n<i>{escaped_error[:250]}</i>")


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
