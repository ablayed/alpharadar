from __future__ import annotations

import html
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import aiosqlite
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from src.alerts import AlertPipeline, AlertSeverity
from src.agents.providers import get_provider
from src.agents.state import AgentAnalysis, DebateVerdict, LLMUsageSummary
from src.bot.messages import _send_long_message
from src.briefing.formatter import BriefingFormatter
from src.briefing.models import BriefingData
from src.correlation import compute_lead_lag
from src.correlation.models import LeadLagResult, PairCorrelation, RegimeState
from src.config import settings
from src.data.base import AssetPrice
from src.data.aggregator import DataAggregator
from src.data.hot_list import (
    MAX_SIZE as MAX_HOT_LIST_SIZE,
    get_active_hot_list,
    get_latest_discovery_run,
)
from src.events.calibration import format_remaining_drift_status
from src.events.models import DetectedEvent
from src.ml import MLEnsemble
from src.reflection.evaluator import AccuracyStats, ReflectionEvaluator
from src.sentiment.analyzer import SentimentAnalyzer
from src.signals.models import CompositeScore
from src.storage.database import Database, WatchlistItem
from src.ta.signals import TASignal
from src.trade import OpenPosition, PortfolioTracker, TradeSignal, generate_crypto_trade_signal, generate_trade_signal

logger = logging.getLogger(__name__)

_alert_cooldowns: dict[str, float] = {}
_alert_zone_active: dict[str, bool] = {}
_alert_level_sides: dict[str, int] = {}
_alert_state_seen: dict[str, float] = {}
ALERT_COOLDOWN_SECONDS = 3600

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
        "/history  Today's biggest crypto moves\n"
        "/hotlist  View the auto-discovered crypto hot list\n"
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
        "/funding  Show derivatives and crypto macro context\n"
        "/signals NVDA  Show the full signal stack for one symbol\n"
        "/trade NVDA  Generate a structured trade setup for one symbol\n"
        "/trades  Scan the watchlist for active trade setups\n"
        "/log BTC 68500 0.015  Log that you entered a trade\n"
        "/close BTC 69500 tp1  Record how a trade was closed\n"
        "/journal  Show realized trading performance\n"
        "/portfolio  Show tracked trade risk and stale positions\n"
        "/ta NVDA  Compute local technical analysis for one symbol\n"
        "/alerts  Show real-time alert pipeline status\n"
        "/accuracy  Show multi-timeframe prediction accuracy\n"
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


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show today's biggest crypto moves."""
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    aggregator = _aggregator_from_context(context)

    try:
        data = await aggregator.gather_briefing_data(include_agent_analysis=False)
        _store_latest_briefing(context, data)
    except Exception:
        logger.exception("History command failed while fetching data")
        await update.message.reply_text("⚠️ Could not fetch current data.")
        return

    cryptos = sorted(
        data.crypto_prices,
        key=lambda price: abs(price.change_24h_pct),
        reverse=True,
    )
    if not cryptos:
        await update.message.reply_text("No crypto data available.")
        return

    now = datetime.now(ZoneInfo("Europe/London"))
    lines = [
        "📜 <b>TODAY'S CRYPTO MOVES</b>",
        now.strftime("%A, %B %d"),
        "",
    ]

    for crypto in cryptos:
        change = crypto.change_24h_pct
        if change > 5:
            emoji = "🚀 "
        elif change > 2:
            emoji = "🟢 "
        elif change < -5:
            emoji = "💥 "
        elif change < -2:
            emoji = "🔴 "
        else:
            emoji = ""

        week_context = ""
        if crypto.change_7d_pct is not None:
            week_context = f" (7d: {crypto.change_7d_pct:+.1f}%)"

        lines.append(
            f"{emoji}<b>{html.escape(crypto.symbol)}</b> "
            f"{_format_trade_price(crypto.current_price)} "
            f"{change:+.1f}%{week_context}"
        )

    btc = next((crypto for crypto in cryptos if crypto.symbol.upper() == "BTC"), None)
    if btc is not None or data.total_tvl is not None:
        lines.append("")

    if btc is not None:
        if btc.change_24h_pct > 2:
            lines.append("📊 BTC leading the market higher today")
        elif btc.change_24h_pct < -2:
            lines.append("📊 BTC dragging the market lower today")
        else:
            lines.append("📊 BTC relatively flat, altcoin moves are more independent today")

    if data.total_tvl is not None:
        tvl_change = data.tvl_change_24h_pct or 0.0
        tvl_emoji = "🟢" if tvl_change >= 0 else "🔴"
        lines.append(f"{tvl_emoji} DeFi TVL: ${data.total_tvl / 1e9:.1f}B ({tvl_change:+.1f}%)")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def hotlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the current hot list of auto-discovered crypto assets."""
    if not await owner_only(update, context):
        return
    if update.message is None or update.effective_chat is None:
        return

    hot_coins = await get_active_hot_list(_db_from_context(context).db_path)
    if not hot_coins:
        await update.message.reply_text("Hot list is empty. Next discovery scan runs at 6:30 AM and 6:30 PM London.")
        return

    lines = ["🔥 <b>HOT LIST</b> (auto-discovered crypto momentum)", ""]
    for coin in hot_coins:
        added = str(coin.get("added_at", ""))[:10]
        expires = str(coin.get("expires_at", ""))[:10]
        pct_change = float(coin.get("price_change_24h", 0) or 0)
        direction = "🟢" if pct_change >= 0 else "🔴"
        lines.append(
            f"{direction} <b>{html.escape(str(coin['symbol']))}</b> "
            f"| Score: {int(coin.get('signal_score', 0) or 0)}/100 "
            f"| {pct_change:+.1f}% "
            f"| Vol: ${float(coin.get('initial_volume', 0) or 0):,.0f}"
        )
        lines.append(
            f"   Added: {html.escape(added)} | Expires: {html.escape(expires)} | "
            f"{html.escape(str(coin.get('trigger_reason', 'unknown')), quote=False)}"
        )

    lines.extend(
        [
            "",
            f"Total: {len(hot_coins)} / {MAX_HOT_LIST_SIZE}",
            "Use /trade <SYMBOL> to analyze any hot-list coin.",
        ]
    )
    await _send_long_message(context.bot, update.effective_chat.id, "\n".join(lines))


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    tracker = context.application.bot_data.get("portfolio_tracker")
    if not isinstance(tracker, PortfolioTracker):
        await update.message.reply_text("Portfolio tracker not initialized.")
        return

    await update.message.reply_text(tracker.format_status(), parse_mode="HTML")


async def log_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    if len(context.args) < 3:
        await update.message.reply_text(
            "Usage: /log <SYMBOL> <ENTRY_PRICE> <SIZE>\n"
            "Example: /log BTC 68500 0.015\n"
            "Example: /log SOL 87.50 10"
        )
        return

    symbol = context.args[0].strip().upper()
    try:
        entry_price = float(context.args[1])
        size = float(context.args[2])
    except ValueError:
        await update.message.reply_text("Price and size must be numbers.")
        return

    if entry_price <= 0 or size <= 0:
        await update.message.reply_text("Entry price and size must be greater than zero.")
        return

    trade = await activate_logged_trade(_db_from_context(context).db_path, symbol, entry_price, size)
    if trade is None:
        await update.message.reply_text(
            f"No pending trade signal for {html.escape(symbol)}.\n"
            f"Run /trade {html.escape(symbol)} first to generate a setup."
        )
        return

    portfolio_tracker = context.application.bot_data.get("portfolio_tracker")
    if isinstance(portfolio_tracker, PortfolioTracker):
        asset_type = await _asset_type_for_symbol(_db_from_context(context), symbol)
        portfolio_tracker.add_position(
            OpenPosition(
                symbol=symbol,
                asset_type=asset_type,
                direction=str(trade["direction"]),
                entry_price=float(trade["entry_price"]),
                stop_loss=float(trade["stop_loss"]),
                tp1=float(trade["tp1"]),
                risk_pct=float(trade["risk_pct"] or 1.0),
                opened_at=datetime.now(ZoneInfo("UTC")).timestamp(),
            )
        )

    direction = str(trade["direction"])
    direction_label = "LONG" if direction == "LONG" else "SHORT"
    text = (
        f"<b>TRADE LOGGED: {html.escape(symbol)}</b>\n\n"
        f"Direction: {'🟢' if direction == 'LONG' else '🔴'} {direction_label}\n"
        f"Entry: {_format_trade_price(float(trade['entry_price']))} x {float(trade['entry_size']):g}\n"
        f"Cost: {_format_trade_price(float(trade['entry_cost']))}\n"
        f"Risk: £{float(trade['risk_amount']):.2f}\n\n"
        f"SL: {_format_trade_price(float(trade['stop_loss']))} | "
        f"TP1: {_format_trade_price(float(trade['tp1']))} | "
        f"TP2: {_format_trade_price(float(trade['tp2']))} | "
        f"TP3: {_format_trade_price(float(trade['tp3']))}\n\n"
        "This trade is now active. I will flag TP and SL touches as live prices update.\n"
        f"Use /close {html.escape(symbol)} <EXIT_PRICE> [reason] when you exit."
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def close_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /close <SYMBOL> <EXIT_PRICE> [reason]\n"
            "Reasons: tp1, tp2, tp3, sl, manual, breakeven, expired"
        )
        return

    symbol = context.args[0].strip().upper()
    try:
        exit_price = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Exit price must be a number.")
        return

    if exit_price <= 0:
        await update.message.reply_text("Exit price must be greater than zero.")
        return

    reason = context.args[2].strip().lower() if len(context.args) > 2 else "manual"
    trade = await close_logged_trade(_db_from_context(context).db_path, symbol, exit_price, reason)
    if trade is None:
        await update.message.reply_text(f"No active trade for {html.escape(symbol)}.")
        return

    portfolio_tracker = context.application.bot_data.get("portfolio_tracker")
    if isinstance(portfolio_tracker, PortfolioTracker):
        portfolio_tracker.remove_position(symbol)

    pnl = float(trade["realized_pnl"])
    emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
    text = (
        f"{emoji} <b>TRADE CLOSED: {html.escape(symbol)}</b>\n\n"
        f"Entry: {_format_trade_price(float(trade['entry_price']))} | "
        f"Exit: {_format_trade_price(float(trade['exit_price']))}\n"
        f"P&L: £{pnl:.2f} ({float(trade['realized_pnl_pct']):+.1f}%)\n"
        f"R-multiple: {float(trade['r_multiple']):+.1f}R\n"
        f"Reason: {html.escape(str(trade['exit_reason']))}\n\n"
        "Use /journal to review realized performance."
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def journal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    snapshot = await load_trade_journal_snapshot(_db_from_context(context).db_path)
    total_trades = int(snapshot["total_trades"])
    active_trades = int(snapshot["active_trades"])
    if total_trades == 0:
        await update.message.reply_text(
            "📓 <b>TRADING JOURNAL</b>\n\n"
            f"No closed trades yet ({active_trades} active).\n"
            "Use /log to record an entry, then /close when you exit.",
            parse_mode="HTML",
        )
        return

    profit_factor = snapshot["profit_factor"]
    profit_factor_text = "inf" if profit_factor == float("inf") else f"{float(profit_factor):.2f}"
    lines = [
        "📓 <b>TRADING JOURNAL</b>",
        "",
        "<b>Overall Performance:</b>",
        f"  Total trades: {total_trades} ({active_trades} active)",
        f"  Win rate: {float(snapshot['win_rate']):.1f}% "
        f"({int(snapshot['wins'])}W / {int(snapshot['losses'])}L)",
        f"  Profit factor: {profit_factor_text}",
        f"  Average R: {float(snapshot['avg_r']):+.2f}R",
        "",
        "<b>P&L Summary:</b>",
        f"  Total profit: £{float(snapshot['total_profit']):.2f}",
        f"  Total loss: £{float(snapshot['total_loss']):.2f}",
        f"  Net P&L: £{float(snapshot['net_pnl']):.2f}",
        f"  Best trade: £{float(snapshot['best_trade']):.2f}",
        f"  Worst trade: £{float(snapshot['worst_trade']):.2f}",
        "",
        "<b>Target Hit Rates:</b>",
        f"  TP1 hit rate: {float(snapshot['tp1_hit_rate']):.0f}%",
        f"  SL hit rate: {float(snapshot['sl_hit_rate']):.0f}%",
    ]

    recent_trades = snapshot["recent_trades"]
    if isinstance(recent_trades, list) and recent_trades:
        lines.extend(["", "<b>Recent Trades:</b>"])
        for trade in recent_trades:
            pnl = float(trade["realized_pnl"])
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            lines.append(
                f"  {emoji} {html.escape(str(trade['symbol']))} {html.escape(str(trade['direction']))} | "
                f"{html.escape(str(trade['exit_reason']))} | "
                f"£{pnl:.2f} ({float(trade['r_multiple']):+.1f}R)"
            )

    suggestions = snapshot["suggestions"]
    if isinstance(suggestions, list) and suggestions:
        lines.extend(["", "<b>Areas to Improve:</b>"])
        lines.extend(f"  {html.escape(str(item), quote=False)}" for item in suggestions)

    if update.effective_chat is None:
        return
    await _send_long_message(context.bot, update.effective_chat.id, "\n".join(lines))


async def funding_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    msg = await update.message.reply_text("\U0001F4CA Loading derivatives context...")
    try:
        data = await _latest_briefing_data(context, include_agent_analysis=False)
        _store_latest_briefing(context, data)
    except Exception as exc:
        logger.exception("Funding command failed while loading briefing data")
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Could not load derivatives data.\n<i>{escaped_error[:250]}</i>")
        return

    symbols = sorted(set(data.funding) | set(data.oi))
    if not symbols and data.fear_greed is None and data.dxy is None and data.cme_gap is None:
        await msg.edit_text("No derivatives or macro data available yet.")
        return

    lines = ["\U0001F4CA <b>DERIVATIVES INTELLIGENCE</b>", ""]
    for symbol in symbols:
        funding = data.funding.get(symbol)
        open_interest = data.oi.get(symbol)
        if funding is not None:
            emoji = (
                "\U0001F7E2" if funding.signal == "bullish" else "\U0001F534" if funding.signal == "bearish" else "\u26AA"
            )
            lines.append(
                f"{emoji} <b>{html.escape(symbol)}</b> Funding: {funding.current_rate * 100:+.4f}%"
            )
        if open_interest is not None:
            lines.append(
                f"   OI 24h: {open_interest.oi_change_24h_pct:+.1f}% | "
                f"4h: {open_interest.oi_change_4h_pct:+.1f}%"
            )

    macro_lines: list[str] = []
    if data.fear_greed is not None:
        macro_lines.append(
            f"\U0001F630 Fear &amp; Greed: {data.fear_greed.value} "
            f"({html.escape(data.fear_greed.classification, quote=False)})"
        )
    if data.dxy is not None:
        macro_lines.append(f"\U0001F4B5 DXY: {data.dxy.current:.2f} ({data.dxy.change_7d_pct:+.1f}% this week)")
    if data.btc_dominance is not None:
        macro_lines.append(
            f"\U0001F451 BTC Dominance: {data.btc_dominance.dominance_pct:.1f}% "
            f"({html.escape(data.btc_dominance.season.replace('_', ' '), quote=False)})"
        )
    if data.cme_gap is not None and not data.cme_gap.filled:
        macro_lines.append(
            f"\U0001F4CA CME Gap: ${data.cme_gap.gap_low:,.0f}-${data.cme_gap.gap_high:,.0f} "
            f"({data.cme_gap.distance_pct:.1f}% away)"
        )
    if macro_lines:
        lines.extend(["", "<b>MACRO</b>", *macro_lines])

    await msg.edit_text("\n".join(lines), parse_mode="HTML")


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
                "\u26A0\uFE0F Agent pipeline unavailable. Check GROQ_API_KEY or Groq status."
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
        # The playbook can still be generated without agent verdicts, which keeps
        # the command reliable when the debate pipeline is unavailable or rate-limited.
        data = await _latest_briefing_data(context, include_agent_analysis=False)
        _store_latest_briefing(context, data)
        playbook_text, llm_calls, duration, llm_usage = await aggregator.generate_playbook(data)
        message = _format_playbook_message(
            generated_at=data.generated_at,
            playbook_text=playbook_text,
            llm_calls=llm_calls,
            duration=duration,
            llm_usage=llm_usage,
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
    data = await _latest_briefing_data(context, include_agent_analysis=True)
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


async def trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return
    if not context.args:
        await update.message.reply_text("Usage: /trade &lt;symbol&gt;")
        return

    symbol = context.args[0].strip().upper()
    if not symbol:
        await update.message.reply_text("Usage: /trade &lt;symbol&gt;")
        return

    msg = await update.message.reply_text(f"\U0001F3AF Generating trade signal for {html.escape(symbol)}...")
    aggregator = _aggregator_from_context(context)

    try:
        data = await _latest_briefing_data(context, include_agent_analysis=True)
        price_item = _price_item_for_symbol(data, symbol)
        if price_item is None:
            data = await aggregator.gather_briefing_data(include_agent_analysis=True)
            _store_latest_briefing(context, data)
            price_item = _price_item_for_symbol(data, symbol)
            if price_item is None:
                await msg.edit_text(f"\u26A0\uFE0F Could not find price data for {html.escape(symbol)}.")
                return

        ta_signal = data.ta_signals.get(symbol)
        if ta_signal is None:
            ta_signal = await aggregator.get_ta_signal(symbol)
            if ta_signal is not None:
                data.ta_signals[symbol] = ta_signal

        verdict = _verdict_for_symbol(data.agent_analysis, symbol)
        if verdict is None:
            forced_analysis = await aggregator.run_agent_analysis(data, forced_symbols=[symbol], source="trade")
            if forced_analysis is not None:
                merged_analysis = _merge_agent_analysis(data.agent_analysis, forced_analysis)
                data = aggregator.apply_agent_analysis(data, merged_analysis)
                _store_latest_briefing(context, data)
                verdict = _verdict_for_symbol(data.agent_analysis, symbol)

        composite = data.composite_scores.get(symbol)
        regime_key = (
            data.regime_state.regime.value
            if data.regime_state is not None
            else "default"
        )
        event_payload = _event_payload(_event_for_symbol(data, symbol))
        if price_item.asset_type.value == "crypto":
            ml_prediction = _ml_prediction_for_signal(
                context.application.bot_data.get("ml_ensemble"),
                symbol,
                ta_signal,
            )
            signal = generate_crypto_trade_signal(
                symbol=symbol,
                current_price=price_item.current_price,
                ta_signal=ta_signal,
                composite=composite,
                verdict=verdict,
                event_data=event_payload,
                regime=regime_key,
                btc_change_pct=_btc_change_pct(data),
                funding=data.funding.get(symbol),
                oi=data.oi.get(symbol),
                cme_gap=data.cme_gap if symbol == "BTC" else None,
                fear_greed=data.fear_greed,
                dxy=data.dxy,
                ml_prediction=ml_prediction,
            )
        else:
            signal = generate_trade_signal(
                symbol=symbol,
                current_price=price_item.current_price,
                ta_signal=ta_signal,
                composite=composite,
                verdict=verdict,
                event_data=event_payload,
                regime=regime_key,
            )

        if signal is None:
            await msg.edit_text(
                _format_no_trade_text(symbol, crypto_style=price_item.asset_type.value == "crypto"),
                parse_mode="HTML",
            )
            return

        if update.effective_chat is None:
            return
        trade_text = (
            _format_crypto_trade_text(signal)
            if price_item.asset_type.value == "crypto"
            else _format_trade_text(signal)
        )
        portfolio_tracker = context.application.bot_data.get("portfolio_tracker")
        if isinstance(portfolio_tracker, PortfolioTracker):
            portfolio_notes = _portfolio_trade_notes(
                symbol=symbol,
                asset_type=price_item.asset_type.value,
                signal=signal,
                portfolio_tracker=portfolio_tracker,
                correlation_engine=context.application.bot_data.get("correlation_engine"),
            )
            if portfolio_notes:
                trade_text += f"\n\n{portfolio_notes}"
        trade_text += (
            f"\n\n<i>When you actually enter, log it with "
            f"/log {html.escape(symbol)} &lt;ENTRY_PRICE&gt; &lt;SIZE&gt;.</i>"
        )
        await _send_long_message(
            context.bot,
            update.effective_chat.id,
            trade_text,
            initial_message=msg,
        )
        prediction_id = await aggregator.tracker.log_prediction(
            symbol=symbol,
            asset_type=price_item.asset_type.value,
            predicted_direction=_trade_direction_to_prediction(signal.direction),
            confidence=signal.confidence,
            price_at_prediction=price_item.current_price,
            source="trade",
            reasoning=signal.reasoning,
            event_type=_event_type_for_symbol(data, symbol),
            predicted_target=signal.tp2,
        )
        await save_signal_to_log(signal, _db_from_context(context).db_path, signal_id=prediction_id)
    except Exception as exc:
        logger.exception("Trade command failed for %s", symbol)
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Trade signal failed.\n<i>{escaped_error[:250]}</i>")


async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    msg = await update.message.reply_text(
        "\U0001F3AF Scanning crypto watchlist for high-probability setups..."
    )
    aggregator = _aggregator_from_context(context)

    try:
        data = await aggregator.gather_briefing_data(include_agent_analysis=True)
        _store_latest_briefing(context, data)
        signals: list[tuple[TradeSignal, AssetPrice]] = []
        regime_key = (
            data.regime_state.regime.value
            if data.regime_state is not None
            else "default"
        )
        btc_change = _btc_change_pct(data)
        for price_item in data.crypto_prices:
            symbol = price_item.symbol.upper()
            ta_signal = data.ta_signals.get(symbol)
            ml_prediction = _ml_prediction_for_signal(
                context.application.bot_data.get("ml_ensemble"),
                symbol,
                ta_signal,
            )
            signal = generate_crypto_trade_signal(
                symbol=symbol,
                current_price=price_item.current_price,
                ta_signal=ta_signal,
                composite=data.composite_scores.get(symbol),
                verdict=_verdict_for_symbol(data.agent_analysis, symbol),
                event_data=_event_payload(_event_for_symbol(data, symbol)),
                regime=regime_key,
                btc_change_pct=btc_change,
                funding=data.funding.get(symbol),
                oi=data.oi.get(symbol),
                cme_gap=data.cme_gap if symbol == "BTC" else None,
                fear_greed=data.fear_greed,
                dxy=data.dxy,
                ml_prediction=ml_prediction,
            )
            if signal is None:
                continue
            signals.append((signal, price_item))

        if not signals:
            await msg.edit_text(
                "<b>NO TRADE SETUPS TODAY</b>\n\n"
                "None of the crypto assets meet the 3+ confluence requirement.\n"
                "This means the risk of entering is higher than the potential reward.\n\n"
                "<i>Patience is a position. The best trade is often no trade.</i>",
                parse_mode="HTML",
            )
            return

        signals.sort(key=lambda item: item[0].confidence, reverse=True)
        lines = [f"\U0001F3AF <b>CRYPTO TRADE SIGNALS</b> ({len(signals)} setups found)", ""]
        for signal, price_item in signals[:5]:
            direction_emoji = "\U0001F7E2" if signal.direction == "LONG" else "\U0001F534"
            stars = _trade_confluence_stars(signal.confluence_count)
            lines.append(
                f"{direction_emoji} <b>{html.escape(signal.symbol)}</b> "
                f"{html.escape(signal.direction)} ({signal.confidence:.0%}) {stars}".rstrip()
            )
            lines.append(
                f"   Entry: {_format_trade_price(signal.entry_low)} - "
                f"{_format_trade_price(signal.entry_high)}"
            )
            lines.append(
                f"   SL: {_format_trade_price(signal.stop_loss)} (-{signal.stop_loss_pct:.1f}%) | "
                f"TP1: {_format_trade_price(signal.tp1)} | "
                f"TP2: {_format_trade_price(signal.tp2)} | "
                f"TP3: {_format_trade_price(signal.tp3)}"
            )
            lines.append(
                f"   R:R 1:{signal.risk_reward} | {html.escape(signal.timeframe)} | "
                f"Risk: {signal.suggested_risk_pct:.0f}%"
            )
            lines.append(f"   <i>{html.escape(signal.reasoning[:150], quote=False)}</i>")
            lines.append("")
            prediction_id = await aggregator.tracker.log_prediction(
                symbol=signal.symbol,
                asset_type=price_item.asset_type.value,
                predicted_direction=_trade_direction_to_prediction(signal.direction),
                confidence=signal.confidence,
                price_at_prediction=price_item.current_price,
                source="trade",
                reasoning=signal.reasoning,
                event_type=_event_type_for_symbol(data, signal.symbol),
                predicted_target=signal.tp2,
            )
            await save_signal_to_log(signal, _db_from_context(context).db_path, signal_id=prediction_id)

        lines.append(
            "<b>NOT financial advice.</b> Never risk more than 2% per trade.\n"
            "Use /trade &lt;symbol&gt; for full details on any setup.\n"
            "After you actually enter, record it with /log &lt;symbol&gt; &lt;entry_price&gt; &lt;size&gt;."
        )
        if update.effective_chat is None:
            return
        await _send_long_message(
            context.bot,
            update.effective_chat.id,
            "\n".join(lines),
            initial_message=msg,
        )
    except Exception as exc:
        logger.exception("Trades command failed")
        escaped_error = html.escape(str(exc), quote=False)
        await msg.edit_text(f"\u26A0\uFE0F Trade scan failed.\n<i>{escaped_error[:250]}</i>")


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
        if (
            hasattr(alert, "symbol")
            and hasattr(alert, "change_pct")
            and hasattr(alert, "severity")
            and getattr(alert, "asset_type", None) != "stock"
        )
    ]

    text = (
        "\U0001F514 <b>ALERT STATUS</b>\n\n"
        "Real-time stream: Binance crypto alerts only\n"
        f"Alerts this hour: {status['alerts_this_hour']}/{status['global_limit']}\n"
        f"Quiet hours: {'Active \U0001F319' if status['quiet_hours'] else 'Off'}\n"
        f"Assets on cooldown: {status['assets_on_cooldown']}\n\n"
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

    lines = [
        "\U0001F4CA <b>PREDICTION ACCURACY</b>",
        "",
        f"<b>All-time:</b> {snapshot.all_time_predictions} predictions",
        "",
        f"<b>1-day accuracy:</b> {fmt(snapshot.one_day)}",
        f"<b>3-day accuracy:</b> {fmt(snapshot.three_day)}",
        f"<b>7-day accuracy:</b> {fmt(snapshot.seven_day)}",
        "",
        f"<b>High confidence (70%+):</b> {fmt(snapshot.high_confidence_1d)} @ 1d",
        f"<b>Low confidence (&lt;70%):</b> {fmt(snapshot.low_confidence_1d)} @ 1d",
    ]

    if snapshot.by_source_1d:
        lines.extend(["", "<b>By source (@ 1d):</b>"])
        for source, stats in snapshot.by_source_1d.items():
            lines.append(f"{html.escape(source)}: {fmt(stats)}")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def reflect_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None:
        return

    evaluator = _reflection_evaluator_from_context(context)
    logger.info(
        "Reflect command: evaluator_id=%s db_path=%s",
        id(evaluator),
        getattr(evaluator, "_db_path", None),
    )
    reflection = await evaluator.latest_reflection()
    if reflection is None:
        row_count: int | None = None
        latest_created: str | None = None
        db_path = str(getattr(evaluator, "_db_path", ""))
        try:
            async with aiosqlite.connect(db_path) as connection:
                cursor = await connection.execute(
                    """
                    SELECT COUNT(*), MAX(created_at)
                    FROM reflections
                    WHERE reflection_type = 'weekly'
                    """
                )
                row = await cursor.fetchone()
            if row is not None:
                row_count = int(row[0] or 0)
                latest_created = str(row[1]) if row[1] is not None else None
        except Exception:
            logger.exception("Reflect command debug query failed for db_path=%s", db_path)
        logger.warning(
            "Reflect command: latest_reflection returned None db_path=%s row_count=%s latest_created=%s",
            db_path,
            row_count,
            latest_created,
        )
        await update.message.reply_text("No reflections yet. First one runs Sunday at 8:00 AM UTC.")
        return
    logger.info(
        "Reflect command: found reflection id=%s created_at=%s period_start=%s period_end=%s",
        reflection.id,
        reflection.created_at,
        reflection.period_start,
        reflection.period_end,
    )

    text = (
        "\U0001F9E0 <b>WEEKLY REFLECTION</b>\n"
        f"Period: {html.escape(reflection.period_start[:10])} to {html.escape(reflection.period_end[:10])}\n"
        f"Predictions: {reflection.total_predictions} | "
        f"Accuracy: {(reflection.accuracy_1d or 0.0):.1f}%\n\n"
        f"{html.escape(reflection.reflection_text, quote=False)}"
    )
    logger.info(
        "Reflect command: sending reflection id=%s text_len=%s",
        reflection.id,
        len(text),
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


async def _legacy_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None or update.effective_chat is None:
        return

    bot_data = context.application.bot_data
    aggregator = _aggregator_from_context(context)
    db = _db_from_context(context)

    try:
        health: dict[str, bool | None] = await aggregator.provider_health()
    except Exception:
        logger.exception("Status command failed while checking provider health")
        health = {
            "finnhub": None,
            "coingecko": None,
            "defillama": None,
            "edgar": None,
        }

    try:
        cg_used, cg_limit = await aggregator.coingecko.get_remaining_budget()
    except Exception:
        logger.exception("Status command failed while loading CoinGecko budget")
        cg_used = 0
        cg_limit = settings.coingecko_monthly_limit

    crypto_items = await db.get_watchlist("crypto")
    crypto_symbols = sorted({item.symbol.upper() for item in crypto_items})
    hot_coins = await get_active_hot_list(db.db_path)
    latest_discovery = await get_latest_discovery_run(db.db_path)
    ml_section, last_trained_text = _format_ml_status_lines(
        bot_data.get("ml_ensemble"),
        crypto_symbols,
    )

    funding = bot_data.get("funding")
    if not isinstance(funding, dict):
        funding = {}
    oi = bot_data.get("oi")
    if not isinstance(oi, dict):
        oi = {}

    n_funding = sum(1 for item in funding.values() if item is not None)
    n_oi = sum(1 for item in oi.values() if item is not None)

    fear_greed = bot_data.get("fear_greed")
    dxy = bot_data.get("dxy")
    btc_dominance = bot_data.get("btc_dominance")
    regime_text = _regime_status_text(bot_data.get("current_regime"))
    cme_status = _format_cme_gap_status(bot_data.get("cme_gap"))

    active_trades, closed_trades, predictions_logged = await _load_status_trade_counts(
        db.db_path,
    )
    portfolio = bot_data.get("portfolio_tracker")
    total_risk = portfolio.get_total_risk() if isinstance(portfolio, PortfolioTracker) else 0.0

    agent_status = aggregator.agent_pipeline_status_text()
    correlation_pair_count = _correlation_pair_count(bot_data)
    ws_state = _websocket_status_text(context)
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


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await owner_only(update, context):
        return
    if update.message is None or update.effective_chat is None:
        return

    bot_data = context.application.bot_data
    aggregator = _aggregator_from_context(context)
    db = _db_from_context(context)

    try:
        health: dict[str, bool | None] = await aggregator.provider_health()
    except Exception:
        logger.exception("Status command failed while checking provider health")
        health = {
            "finnhub": None,
            "coingecko": None,
            "defillama": None,
            "edgar": None,
        }

    try:
        cg_used, cg_limit = await aggregator.coingecko.get_remaining_budget()
    except Exception:
        logger.exception("Status command failed while loading CoinGecko budget")
        cg_used = 0
        cg_limit = settings.coingecko_monthly_limit

    crypto_items = await db.get_watchlist("crypto")
    crypto_symbols = sorted({item.symbol.upper() for item in crypto_items})
    ml_section, last_trained_text = _format_ml_status_lines(
        bot_data.get("ml_ensemble"),
        crypto_symbols,
    )

    funding = bot_data.get("funding")
    if not isinstance(funding, dict):
        funding = {}
    oi = bot_data.get("oi")
    if not isinstance(oi, dict):
        oi = {}

    n_funding = sum(1 for item in funding.values() if item is not None)
    n_oi = sum(1 for item in oi.values() if item is not None)

    fear_greed = bot_data.get("fear_greed")
    dxy = bot_data.get("dxy")
    btc_dominance = bot_data.get("btc_dominance")
    regime_text = _regime_status_text(bot_data.get("current_regime"))
    cme_status = _format_cme_gap_status(bot_data.get("cme_gap"))

    active_trades, closed_trades, predictions_logged = await _load_status_trade_counts(
        db.db_path,
    )
    portfolio = bot_data.get("portfolio_tracker")
    total_risk = portfolio.get_total_risk() if isinstance(portfolio, PortfolioTracker) else 0.0

    agent_status = aggregator.agent_pipeline_status_text()
    llm_status = aggregator.llm_provider_status()
    provider = get_provider()
    credit = await provider.get_openrouter_credit()
    credit_text = f"${credit:.2f} remaining" if credit is not None else "unknown"
    correlation_pair_count = _correlation_pair_count(bot_data)
    ws_state = _websocket_status_text(context)
    uptime = _uptime_text(context)

    lines = [
        "<b>ALPHARADAR STATUS</b>",
        "",
        "\U0001F5A5 Server: DigitalOcean London",
        f"\U0001F4CA RAM: {_memory_usage_text()} | Uptime: {uptime}",
        "",
        "<b>DATA PIPELINE</b>",
        _format_subsystem_line("Finnhub (news + quotes)", health.get("finnhub")),
        _format_subsystem_line("CoinGecko (crypto prices)", health.get("coingecko")),
        _format_subsystem_line("DeFi Llama (TVL)", health.get("defillama")),
        _format_subsystem_line("SEC EDGAR (filings)", health.get("edgar")),
        _format_subsystem_line("Binance Futures (funding + OI)", True if (n_funding or n_oi) else None),
        _format_subsystem_line("Alternative.me (Fear & Greed)", fear_greed is not None),
        _format_subsystem_line(
            "yfinance (DXY + CME + candles)",
            (dxy is not None) or bot_data.get("cme_gap") is not None,
        ),
        "",
        "<b>ANALYSIS ENGINE</b>",
        _format_subsystem_line("FinBERT sentiment (ONNX)", aggregator.sentiment_analyzer.is_available()),
        _format_agent_pipeline_line(agent_status),
        _format_subsystem_line("Technical analysis (14 indicators)", True),
        _format_subsystem_line("SMC detection (FVG + BOS + Order Blocks)", True),
        _format_subsystem_line(
            f"Correlation engine ({correlation_pair_count} pairs tracked)",
            True if correlation_pair_count > 0 else None,
        ),
        f"\U0001F4C8 Regime detection: {html.escape(regime_text, quote=False)}",
        "",
        "<b>LLM PROVIDERS</b>",
        f"OpenRouter: {html.escape(credit_text, quote=False)}",
        "  Judge/Reflection: Gemini 2.5 Pro",
        "  Bull/Bear/Playbook: DeepSeek R1 0528",
        "  General: DeepSeek V3.2",
        (
            "Groq: fallback (Qwen3-32B + Llama 3.3 70B)"
            if llm_status.get("groq")
            else "Groq: not configured"
        ),
        "",
        "<b>ML ENSEMBLE</b>",
        ml_section,
        f"Last trained: {last_trained_text}",
        f"Next retrain: {_next_retrain_text(context)}",
        "",
        "<b>DERIVATIVES</b>",
        f"Funding rates: {n_funding} symbols",
        f"Open Interest: {n_oi} symbols",
        cme_status,
        "",
        "<b>MACRO</b>",
        _format_fear_greed_text(fear_greed),
        _format_dxy_text(dxy),
        _format_btc_dominance_text(btc_dominance),
        "",
        "<b>DISCOVERY</b>",
        "Discovery scan: 6:30 AM + 6:30 PM London",
        f"📊 Hot list: {len(hot_coins)}/{MAX_HOT_LIST_SIZE} active coins",
        f"🆕 Last scan: {_discovery_last_run_text(latest_discovery)}",
        f"📅 Next scan: {_next_discovery_scan_text()}",
        "",
        "<b>TRADING</b>",
        f"\U0001F4CA Active trades: {active_trades}",
        f"\U0001F4BC Portfolio risk: {total_risk:.1f}% (of 6% cap)",
        f"\U0001F4C8 Predictions logged: {predictions_logged}",
        f"\U0001F4D3 Closed trades: {closed_trades}",
        "",
        "<b>SCHEDULING</b>",
        "Daily briefing: 7:00 AM London",
        "Trade scan: 7:00 AM + 7:00 PM London",
        "Weekly reflection: Sunday 8:00 AM UTC",
        "ML retrain: Saturday 6:00 AM London",
        "Dead money check: every 4 hours",
        f"WebSocket: {html.escape(ws_state, quote=False)}",
        "",
        f"CG budget: {cg_used:,}/{cg_limit:,}",
    ]

    await _send_long_message(context.bot, update.effective_chat.id, "\n".join(lines))


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


def _legacy_status_label(ok: bool) -> str:
    return "✅ Operational" if ok else "❌ Degraded"


def _websocket_status_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    ws_tasks = context.application.bot_data.get("ws_tasks")
    if not isinstance(ws_tasks, dict):
        return "Not initialized"

    task = ws_tasks.get("binance")
    suffix = " (crypto only)"
    if task is None:
        return f"Binance idle{suffix}"
    if hasattr(task, "cancelled") and task.cancelled():
        return f"Binance cancelled{suffix}"
    if hasattr(task, "done") and task.done():
        try:
            exc = task.exception() if hasattr(task, "exception") else None
        except Exception:
            exc = None
        return f"Binance stopped{suffix}" if exc else f"Binance done{suffix}"
    return f"Binance running{suffix}"


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


def _status_label(ok: bool | None) -> str:
    if ok is True:
        return "\u2705 Operational"
    if ok is False:
        return "\u274C Degraded"
    return "\u26AA Unknown"


def _format_subsystem_line(name: str, ok: bool | None) -> str:
    return f"{_status_label(ok)} {html.escape(name, quote=False)}"


def _format_agent_pipeline_line(status_text: str) -> str:
    escaped_status = html.escape(status_text, quote=False)
    if status_text.startswith("Ready"):
        return f"\u2705 Multi-agent debate ({escaped_status})"
    if status_text == "Initializing":
        return "\u26AA Multi-agent debate (initializing)"
    return f"\u274C Multi-agent debate ({escaped_status})"


def _memory_usage_text() -> str:
    memory_mb = _memory_usage_mb()
    return f"{memory_mb}MB" if memory_mb is not None else "unavailable"


def _memory_usage_mb() -> int | None:
    cgroup_paths = (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    for path in cgroup_paths:
        try:
            with open(path, encoding="utf-8") as handle:
                value = int(handle.read().strip())
        except Exception:
            continue
        if value > 0:
            return int(round(value / (1024 * 1024)))

    try:
        import psutil  # type: ignore[import-not-found]

        value = psutil.Process().memory_info().rss
        return int(round(value / (1024 * 1024)))
    except Exception:
        pass

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if value <= 0:
            return None
        if value > 10_000_000:
            return int(round(value / (1024 * 1024)))
        return int(round(value / 1024))
    except Exception:
        return None


def _format_ml_status_lines(
    ensemble_obj: object,
    crypto_symbols: list[str],
) -> tuple[str, str]:
    if not isinstance(ensemble_obj, MLEnsemble):
        return "Not initialized", "Never"

    symbols = sorted(set(crypto_symbols))
    loaded_symbols = sorted(str(symbol).upper() for symbol in ensemble_obj._models)
    for symbol in loaded_symbols:
        if symbol not in symbols:
            symbols.append(symbol)

    lines: list[str] = []
    trained_at_values: list[float] = []
    for symbol in symbols:
        models = ensemble_obj._get_models(symbol)
        if models is None:
            lines.append(f"   {html.escape(symbol)}: no model (insufficient data)")
            continue

        trained_at = float(models.get("trained_at", 0.0) or 0.0)
        n_samples = int(models.get("n_samples", 0) or 0)
        age_days = max((time.time() - trained_at) / 86400, 0.0) if trained_at > 0 else 0.0
        lines.append(f"   {html.escape(symbol)}: {n_samples} samples ({age_days:.0f}d ago)")
        if trained_at > 0:
            trained_at_values.append(trained_at)

    if not lines:
        lines.append("   No crypto symbols tracked")

    last_trained_text = _format_timestamp(max(trained_at_values)) if trained_at_values else "Never"
    return "\n".join(lines), last_trained_text


def _format_timestamp(timestamp: float) -> str:
    dt = datetime.fromtimestamp(timestamp, ZoneInfo("UTC"))
    hour_12 = dt.hour % 12 or 12
    return f"{dt:%Y-%m-%d} {hour_12}:{dt:%M %p} {dt:%Z}"


def _next_retrain_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    job_queue = context.application.job_queue
    if job_queue is not None and hasattr(job_queue, "get_jobs_by_name"):
        jobs = job_queue.get_jobs_by_name("weekly_ml_training")
        if jobs:
            next_run = getattr(jobs[0], "next_t", None) or getattr(jobs[0], "next_run_time", None)
            if isinstance(next_run, datetime):
                hour_12 = next_run.hour % 12 or 12
                return f"{next_run:%Y-%m-%d} {hour_12}:{next_run:%M %p} {next_run:%Z}"

    tz = ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    days_until_saturday = (5 - now.weekday()) % 7
    next_run = now.replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=days_until_saturday)
    if next_run <= now:
        next_run = next_run + timedelta(days=7)
    hour_12 = next_run.hour % 12 or 12
    return f"{next_run:%Y-%m-%d} {hour_12}:{next_run:%M %p} {next_run:%Z}"


def _next_discovery_scan_text() -> str:
    tz = ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    candidates = [
        now.replace(hour=6, minute=30, second=0, microsecond=0),
        now.replace(hour=18, minute=30, second=0, microsecond=0),
    ]
    future_runs = [candidate for candidate in candidates if candidate > now]
    next_run = future_runs[0] if future_runs else candidates[0] + timedelta(days=1)
    delta = next_run - now
    total_minutes = max(int(delta.total_seconds() // 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    if hours <= 0:
        return f"in {minutes}m"
    return f"in {hours}h {minutes}m"


def _discovery_last_run_text(run: dict[str, object] | None) -> str:
    if not run:
        return "no scan recorded yet"

    added_count = int(run.get("added_count", 0) or 0)
    run_at = str(run.get("run_at", "") or "")
    timestamp = run_at[:16].replace("T", " ")
    if not timestamp:
        return f"{added_count} new coin(s)"
    return f"{added_count} new coin(s) at {html.escape(timestamp)} UTC"


def _format_cme_gap_status(cme_gap: object) -> str:
    gap_low = getattr(cme_gap, "gap_low", None)
    gap_high = getattr(cme_gap, "gap_high", None)
    distance_pct = getattr(cme_gap, "distance_pct", None)
    filled = bool(getattr(cme_gap, "filled", True))
    if gap_low is None or gap_high is None or filled:
        return "CME gap: none detected"
    return f"CME gap: ${float(gap_low):,.0f}-${float(gap_high):,.0f} ({float(distance_pct or 0.0):.1f}% away)"


def _format_fear_greed_text(fear_greed: object) -> str:
    value = getattr(fear_greed, "value", None)
    classification = getattr(fear_greed, "classification", None)
    if value is None or classification is None:
        return "Fear &amp; Greed: unavailable"
    return f"\U0001F630 Fear &amp; Greed: {int(value)} ({html.escape(str(classification), quote=False)})"


def _format_dxy_text(dxy: object) -> str:
    current = getattr(dxy, "current", None)
    if current is None:
        return "DXY: unavailable"
    return f"\U0001F4B5 DXY: {float(current):.2f}"


def _format_btc_dominance_text(btc_dominance: object) -> str:
    dominance_pct = getattr(btc_dominance, "dominance_pct", None)
    if dominance_pct is None:
        return "BTC Dominance: unavailable"
    return f"\U0001F451 BTC Dominance: {float(dominance_pct):.1f}%"


def _regime_status_text(regime: object) -> str:
    regime_value = getattr(regime, "regime", None)
    if regime_value is None:
        return "Unknown"
    if hasattr(regime_value, "value"):
        regime_value = regime_value.value
    return str(regime_value).replace("_", " ").title()


def _correlation_pair_count(bot_data: dict[object, object]) -> int:
    lead_lag_pairs = bot_data.get("lead_lag_pairs")
    if isinstance(lead_lag_pairs, dict) and lead_lag_pairs:
        return len(lead_lag_pairs)

    correlation_engine = bot_data.get("correlation_engine")
    symbols = getattr(correlation_engine, "_symbols", None)
    if isinstance(symbols, list) and len(symbols) >= 2:
        return 5
    return 0


async def _load_status_trade_counts(db_path: str) -> tuple[int, int, int]:
    try:
        async with aiosqlite.connect(db_path) as connection:
            active_cursor = await connection.execute(
                "SELECT COUNT(*) FROM trade_log WHERE status = 'active'"
            )
            closed_cursor = await connection.execute(
                "SELECT COUNT(*) FROM trade_log WHERE status LIKE 'closed_%'"
            )
            predictions_cursor = await connection.execute("SELECT COUNT(*) FROM predictions")

            active_row = await active_cursor.fetchone()
            closed_row = await closed_cursor.fetchone()
            predictions_row = await predictions_cursor.fetchone()
    except Exception:
        logger.exception("Status command failed while loading trading stats")
        return 0, 0, 0

    active = int(active_row[0] or 0) if active_row else 0
    closed = int(closed_row[0] or 0) if closed_row else 0
    predictions = int(predictions_row[0] or 0) if predictions_row else 0
    return active, closed, predictions


def _db_from_context(context: ContextTypes.DEFAULT_TYPE) -> Database:
    db = context.bot_data.get("db")
    if not isinstance(db, Database):
        raise RuntimeError("Database instance missing from bot_data")
    return db


async def save_signal_to_log(
    signal: TradeSignal,
    db_path: str,
    *,
    signal_id: int | None = None,
) -> int:
    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            INSERT INTO trade_log (
                signal_id, symbol, direction, timeframe, confluence_count, status,
                stop_loss, tp1, tp2, tp3, risk_pct
            )
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                signal.symbol.upper(),
                signal.direction.upper(),
                signal.timeframe,
                signal.confluence_count,
                signal.stop_loss,
                signal.tp1,
                signal.tp2,
                signal.tp3,
                signal.suggested_risk_pct,
            ),
        )
        await connection.commit()
        return int(cursor.lastrowid)


async def activate_logged_trade(
    db_path: str,
    symbol: str,
    entry_price: float,
    size: float,
) -> dict[str, object] | None:
    timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT id, direction, timeframe, confluence_count, stop_loss, tp1, tp2, tp3, risk_pct
            FROM trade_log
            WHERE symbol = ? AND status = 'pending'
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (symbol.upper(),),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        trade_id, direction, timeframe, confluence_count, stop_loss, tp1, tp2, tp3, risk_pct = row
        risk_amount = abs(entry_price - float(stop_loss)) * size
        entry_cost = entry_price * size

        await connection.execute(
            """
            UPDATE trade_log
            SET status = 'active',
                entry_price = ?,
                entry_size = ?,
                entry_cost = ?,
                risk_amount = ?,
                entered_at = ?
            WHERE id = ?
            """,
            (entry_price, size, entry_cost, risk_amount, timestamp, trade_id),
        )
        await connection.commit()

    return {
        "id": int(trade_id),
        "symbol": symbol.upper(),
        "direction": str(direction),
        "timeframe": timeframe,
        "confluence_count": int(confluence_count or 0),
        "entry_price": float(entry_price),
        "entry_size": float(size),
        "entry_cost": float(entry_cost),
        "risk_amount": float(risk_amount),
        "risk_pct": float(risk_pct or 0.0),
        "stop_loss": float(stop_loss),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
    }


async def close_logged_trade(
    db_path: str,
    symbol: str,
    exit_price: float,
    reason: str = "manual",
) -> dict[str, object] | None:
    normalized_reason = reason.strip().lower() or "manual"
    timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")

    async with aiosqlite.connect(db_path) as connection:
        connection.row_factory = aiosqlite.Row
        cursor = await connection.execute(
            """
            SELECT id, direction, entry_price, entry_size, risk_amount,
                   tp1_hit, tp1_hit_at, tp2_hit, tp2_hit_at, tp3_hit, tp3_hit_at, sl_hit, sl_hit_at
            FROM trade_log
            WHERE symbol = ? AND status = 'active'
            ORDER BY entered_at DESC, created_at DESC, id DESC
            LIMIT 1
            """,
            (symbol.upper(),),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        entry_price = row["entry_price"]
        entry_size = row["entry_size"]
        if entry_price is None or entry_size is None:
            return None

        entry = float(entry_price)
        size = float(entry_size)
        risk_amount = float(row["risk_amount"] or 0.0)
        direction = str(row["direction"])

        if direction == "LONG":
            pnl = (exit_price - entry) * size
        else:
            pnl = (entry - exit_price) * size
        pnl_pct = (pnl / (entry * size)) * 100 if entry > 0 and size > 0 else 0.0
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0

        if normalized_reason == "sl":
            status = "closed_sl"
        elif normalized_reason in {"tp1", "tp2", "tp3"}:
            status = f"closed_{normalized_reason}"
        elif normalized_reason == "expired":
            status = "expired"
        else:
            status = "closed_manual"

        tp1_hit = int(row["tp1_hit"] or (normalized_reason in {"tp1", "tp2", "tp3"}))
        tp2_hit = int(row["tp2_hit"] or (normalized_reason in {"tp2", "tp3"}))
        tp3_hit = int(row["tp3_hit"] or (normalized_reason == "tp3"))
        sl_hit = int(row["sl_hit"] or (normalized_reason == "sl"))
        tp1_hit_at = row["tp1_hit_at"] or (timestamp if normalized_reason in {"tp1", "tp2", "tp3"} else None)
        tp2_hit_at = row["tp2_hit_at"] or (timestamp if normalized_reason in {"tp2", "tp3"} else None)
        tp3_hit_at = row["tp3_hit_at"] or (timestamp if normalized_reason == "tp3" else None)
        sl_hit_at = row["sl_hit_at"] or (timestamp if normalized_reason == "sl" else None)

        await connection.execute(
            """
            UPDATE trade_log
            SET status = ?,
                exit_price = ?,
                exit_reason = ?,
                closed_at = ?,
                realized_pnl = ?,
                realized_pnl_pct = ?,
                r_multiple = ?,
                tp1_hit = ?,
                tp1_hit_at = ?,
                tp2_hit = ?,
                tp2_hit_at = ?,
                tp3_hit = ?,
                tp3_hit_at = ?,
                sl_hit = ?,
                sl_hit_at = ?
            WHERE id = ?
            """,
            (
                status,
                exit_price,
                normalized_reason,
                timestamp,
                pnl,
                pnl_pct,
                r_multiple,
                tp1_hit,
                tp1_hit_at,
                tp2_hit,
                tp2_hit_at,
                tp3_hit,
                tp3_hit_at,
                sl_hit,
                sl_hit_at,
                int(row["id"]),
            ),
        )
        await connection.commit()

    return {
        "id": int(row["id"]),
        "symbol": symbol.upper(),
        "direction": direction,
        "entry_price": entry,
        "exit_price": float(exit_price),
        "exit_reason": normalized_reason,
        "status": status,
        "realized_pnl": float(pnl),
        "realized_pnl_pct": float(pnl_pct),
        "r_multiple": float(r_multiple),
    }


async def load_trade_journal_snapshot(db_path: str) -> dict[str, object]:
    base_filter = "(status LIKE 'closed_%' OR status = 'expired')"
    async with aiosqlite.connect(db_path) as connection:
        total_cursor = await connection.execute(
            f"SELECT COUNT(*) FROM trade_log WHERE {base_filter}"
        )
        wins_cursor = await connection.execute(
            f"SELECT COUNT(*) FROM trade_log WHERE realized_pnl > 0 AND {base_filter}"
        )
        losses_cursor = await connection.execute(
            f"SELECT COUNT(*) FROM trade_log WHERE realized_pnl <= 0 AND {base_filter}"
        )
        active_cursor = await connection.execute(
            "SELECT COUNT(*) FROM trade_log WHERE status = 'active'"
        )
        pnl_cursor = await connection.execute(
            f"""
            SELECT
                SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END),
                SUM(CASE WHEN realized_pnl <= 0 THEN realized_pnl ELSE 0 END),
                SUM(realized_pnl),
                AVG(r_multiple),
                MAX(realized_pnl),
                MIN(realized_pnl)
            FROM trade_log
            WHERE {base_filter}
            """
        )
        tp1_cursor = await connection.execute(
            f"SELECT SUM(tp1_hit), COUNT(*) FROM trade_log WHERE {base_filter}"
        )
        sl_cursor = await connection.execute(
            f"SELECT COUNT(*) FROM trade_log WHERE sl_hit = 1 AND {base_filter}"
        )
        recent_cursor = await connection.execute(
            f"""
            SELECT symbol, direction, entry_price, exit_price, realized_pnl, r_multiple, exit_reason, closed_at
            FROM trade_log
            WHERE {base_filter}
            ORDER BY closed_at DESC, id DESC
            LIMIT 5
            """
        )

        total_row = await total_cursor.fetchone()
        wins_row = await wins_cursor.fetchone()
        losses_row = await losses_cursor.fetchone()
        active_row = await active_cursor.fetchone()
        pnl_row = await pnl_cursor.fetchone()
        tp1_row = await tp1_cursor.fetchone()
        sl_row = await sl_cursor.fetchone()
        recent_rows = await recent_cursor.fetchall()

    total_trades = int(total_row[0] or 0)
    wins = int(wins_row[0] or 0)
    losses = int(losses_row[0] or 0)
    active_trades = int(active_row[0] or 0)
    total_profit = float((pnl_row[0] if pnl_row else 0.0) or 0.0)
    total_loss = float((pnl_row[1] if pnl_row else 0.0) or 0.0)
    net_pnl = float((pnl_row[2] if pnl_row else 0.0) or 0.0)
    avg_r = float((pnl_row[3] if pnl_row else 0.0) or 0.0)
    best_trade = float((pnl_row[4] if pnl_row else 0.0) or 0.0)
    worst_trade = float((pnl_row[5] if pnl_row else 0.0) or 0.0)
    tp1_hits = int((tp1_row[0] if tp1_row else 0) or 0)
    tp1_total = int((tp1_row[1] if tp1_row else 0) or 0)
    sl_hits = int((sl_row[0] if sl_row else 0) or 0)

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    tp1_hit_rate = (tp1_hits / tp1_total * 100) if tp1_total > 0 else 0.0
    sl_hit_rate = (sl_hits / total_trades * 100) if total_trades > 0 else 0.0
    if total_loss < 0:
        profit_factor = abs(total_profit / total_loss)
    elif total_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    suggestions: list[str] = []
    if win_rate < 50 and total_trades >= 5:
        suggestions.append("📉 Win rate below 50% - tighten entry criteria or wait for cleaner setups.")
    if avg_r < 1.0 and total_trades >= 5:
        suggestions.append("📉 Average R below 1.0 - let winners run or cut losers faster.")
    if profit_factor != float("inf") and profit_factor < 1.5 and total_trades >= 5:
        suggestions.append("📉 Profit factor below 1.5 - risk/reward needs work.")
    if sl_hits > total_trades * 0.5 and total_trades >= 5:
        suggestions.append("📉 Stopped out on 50%+ of trades - stops may be too tight.")
    if total_trades < 10:
        suggestions.append("📊 Too few trades for strong analysis - keep logging.")
    if win_rate >= 55 and avg_r >= 1.5 and total_trades >= 5:
        suggestions.append("✅ Win rate and R-multiple are both solid. Stay consistent.")

    recent_trades = [
        {
            "symbol": row[0],
            "direction": row[1],
            "entry_price": float(row[2] or 0.0),
            "exit_price": float(row[3] or 0.0),
            "realized_pnl": float(row[4] or 0.0),
            "r_multiple": float(row[5] or 0.0),
            "exit_reason": row[6] or "manual",
            "closed_at": row[7],
        }
        for row in recent_rows
    ]

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "active_trades": active_trades,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "total_loss": total_loss,
        "net_pnl": net_pnl,
        "avg_r": avg_r,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "profit_factor": profit_factor,
        "tp1_hit_rate": tp1_hit_rate,
        "sl_hit_rate": sl_hit_rate,
        "recent_trades": recent_trades,
        "suggestions": suggestions,
    }


async def check_trade_levels(
    symbol: str,
    current_price: float,
    db_path: str,
    bot: object,
    chat_id: int | str,
    *,
    portfolio_tracker: PortfolioTracker | None = None,
) -> None:
    now = time.time()
    timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    symbol_label = html.escape(symbol.upper())

    async with aiosqlite.connect(db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT id, direction, entry_price, tp1, tp2, tp3, stop_loss,
                   tp1_hit, tp2_hit, tp3_hit, sl_hit
            FROM trade_log
            WHERE symbol = ? AND status = 'active'
            """,
            (symbol.upper(),),
        )
        trades = await cursor.fetchall()

        for trade in trades:
            trade_id, direction, entry_price, tp1, tp2, tp3, stop_loss, tp1_hit, tp2_hit, tp3_hit, sl_hit = trade
            if tp3_hit or sl_hit or entry_price is None:
                continue

            entry = float(entry_price)
            if entry <= 0:
                continue

            pct_from_entry = (
                ((current_price - entry) / entry) * 100
                if direction == "LONG"
                else ((entry - current_price) / entry) * 100
            )

            stop_triggered = (
                direction == "LONG" and current_price <= float(stop_loss)
            ) or (
                direction == "SHORT" and current_price >= float(stop_loss)
            )
            if stop_triggered:
                await connection.execute(
                    "UPDATE trade_log SET sl_hit = 1, sl_hit_at = ? WHERE id = ?",
                    (timestamp, trade_id),
                )
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"\U0001F534 <b>STOP LOSS HIT: {symbol_label}</b>\n\n"
                            f"Price reached {_format_trade_price(current_price)} "
                            f"(SL was {_format_trade_price(float(stop_loss))}).\n"
                            f"Use /close {symbol_label} {current_price:.8g} sl to log the exit."
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    logger.exception("Failed to send SL alert for %s", symbol.upper())
                continue

            tp1_triggered = (
                not tp1_hit
                and (
                    (direction == "LONG" and current_price >= float(tp1))
                    or (direction == "SHORT" and current_price <= float(tp1))
                )
            )
            if tp1_triggered:
                await connection.execute(
                    "UPDATE trade_log SET tp1_hit = 1, tp1_hit_at = ? WHERE id = ?",
                    (timestamp, trade_id),
                )
                if portfolio_tracker is not None:
                    portfolio_tracker.mark_tp1_hit(symbol)
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"\U0001F3AF <b>TP1 HIT: {symbol_label}</b>\n\n"
                            f"Price reached {_format_trade_price(current_price)}.\n"
                            f"Sell 1/3 now and move your stop to breakeven "
                            f"({_format_trade_price(entry)}).\n\n"
                            f"Use /close {symbol_label} {current_price:.8g} tp1 "
                            "if you are taking full profit here."
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    logger.exception("Failed to send TP1 alert for %s", symbol.upper())

            tp2_triggered = (
                bool(tp1_hit)
                and not tp2_hit
                and (
                    (direction == "LONG" and current_price >= float(tp2))
                    or (direction == "SHORT" and current_price <= float(tp2))
                )
            )
            if tp2_triggered:
                await connection.execute(
                    "UPDATE trade_log SET tp2_hit = 1, tp2_hit_at = ? WHERE id = ?",
                    (timestamp, trade_id),
                )
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"\U0001F3AF\U0001F3AF <b>TP2 HIT: {symbol_label}</b>\n\n"
                            f"Price reached {_format_trade_price(current_price)}.\n"
                            f"Sell another 1/3 and move your stop to TP1 "
                            f"({_format_trade_price(float(tp1))})."
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    logger.exception("Failed to send TP2 alert for %s", symbol.upper())

            tp3_triggered = (
                bool(tp2_hit)
                and not tp3_hit
                and (
                    (direction == "LONG" and current_price >= float(tp3))
                    or (direction == "SHORT" and current_price <= float(tp3))
                )
            )
            if tp3_triggered:
                await connection.execute(
                    "UPDATE trade_log SET tp3_hit = 1, tp3_hit_at = ? WHERE id = ?",
                    (timestamp, trade_id),
                )
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"\U0001F3AF\U0001F3AF\U0001F3AF <b>TP3 HIT: {symbol_label}</b>\n\n"
                            f"Full target reached at {_format_trade_price(current_price)}.\n"
                            f"Use /close {symbol_label} {current_price:.8g} tp3"
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    logger.exception("Failed to send TP3 alert for %s", symbol.upper())

            effective_sl_hit = bool(sl_hit or stop_triggered)
            effective_tp1_hit = bool(tp1_hit or tp1_triggered)
            effective_tp2_hit = bool(tp2_hit or tp2_triggered)
            effective_tp3_hit = bool(tp3_hit or tp3_triggered)

            proximity_checks = [
                ("SL", float(stop_loss), not effective_sl_hit),
                ("TP1", float(tp1), not effective_tp1_hit),
                ("TP2", float(tp2), effective_tp1_hit and not effective_tp2_hit),
                ("TP3", float(tp3), effective_tp2_hit and not effective_tp3_hit),
            ]
            for level_name, level_price, enabled in proximity_checks:
                alert_key = f"{trade_id}:proximity:{level_name}"
                _alert_state_seen[alert_key] = now
                if not enabled or level_price <= 0:
                    _alert_zone_active[alert_key] = False
                    continue

                distance_pct = abs(current_price - level_price) / max(abs(current_price), 1e-10) * 100
                in_zone = distance_pct <= 1.0
                if not in_zone:
                    _alert_zone_active[alert_key] = False
                    continue

                if (
                    not _alert_zone_active.get(alert_key, False)
                    and now - _alert_cooldowns.get(alert_key, 0.0) > ALERT_COOLDOWN_SECONDS
                ):
                    _alert_cooldowns[alert_key] = now
                    emoji = "\U0001F534" if level_name == "SL" else "\U0001F3AF"
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=(
                                f"{emoji} <b>{symbol_label}</b> approaching {level_name}\n"
                                f"Price: {_format_trade_price(current_price)} ({pct_from_entry:+.1f}% from entry)\n"
                                f"{level_name} at {_format_trade_price(level_price)} - {distance_pct:.1f}% away"
                            ),
                            parse_mode="HTML",
                        )
                    except Exception:
                        logger.exception(
                            "Failed to send %s proximity alert for %s",
                            level_name,
                            symbol.upper(),
                        )
                _alert_zone_active[alert_key] = True

            entry_key = f"{trade_id}:cross:entry"
            entry_distance_pct = abs(current_price - entry) / entry * 100
            entry_in_zone = entry_distance_pct <= 0.3
            previous_entry_side = _alert_level_sides.get(entry_key)
            _alert_state_seen[entry_key] = now
            if (
                previous_entry_side is not None
                and previous_entry_side != 0
                and entry_in_zone
                and not _alert_zone_active.get(entry_key, False)
                and now - _alert_cooldowns.get(entry_key, 0.0) > ALERT_COOLDOWN_SECONDS
            ):
                _alert_cooldowns[entry_key] = now
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"\U0001F4CD <b>{symbol_label}</b> back at entry level\n"
                            f"Price: {_format_trade_price(current_price)} ({pct_from_entry:+.1f}% from entry)"
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    logger.exception("Failed to send entry-cross alert for %s", symbol.upper())
            _alert_zone_active[entry_key] = entry_in_zone
            _alert_level_sides[entry_key] = _price_side(current_price, entry)

            if not effective_tp1_hit and float(tp1) > 0:
                halfway = (entry + float(tp1)) / 2
                halfway_key = f"{trade_id}:cross:halfway_tp1"
                halfway_distance_pct = abs(current_price - halfway) / max(abs(halfway), 1e-10) * 100
                halfway_in_zone = halfway_distance_pct <= 0.5
                previous_halfway_side = _alert_level_sides.get(halfway_key)
                crossed_halfway = (
                    previous_halfway_side is not None
                    and (
                        (direction == "LONG" and previous_halfway_side < 0 and current_price >= halfway)
                        or (direction == "SHORT" and previous_halfway_side > 0 and current_price <= halfway)
                    )
                )
                _alert_state_seen[halfway_key] = now
                if (
                    crossed_halfway
                    and halfway_in_zone
                    and not _alert_zone_active.get(halfway_key, False)
                    and now - _alert_cooldowns.get(halfway_key, 0.0) > ALERT_COOLDOWN_SECONDS
                ):
                    _alert_cooldowns[halfway_key] = now
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=(
                                f"\U0001F4CD <b>{symbol_label}</b> halfway to TP1\n"
                                f"Price: {_format_trade_price(current_price)} ({pct_from_entry:+.1f}% from entry)"
                            ),
                            parse_mode="HTML",
                        )
                    except Exception:
                        logger.exception("Failed to send halfway alert for %s", symbol.upper())
                _alert_zone_active[halfway_key] = halfway_in_zone
                _alert_level_sides[halfway_key] = _price_side(current_price, halfway)
            else:
                halfway_key = f"{trade_id}:cross:halfway_tp1"
                _alert_zone_active[halfway_key] = False

        await connection.commit()

    _cleanup_trade_alert_state(now)


def _price_side(current_price: float, level: float) -> int:
    if current_price > level:
        return 1
    if current_price < level:
        return -1
    return 0


def _cleanup_trade_alert_state(now: float) -> None:
    cutoff = now - 86400
    stale_keys = {key for key, seen_at in _alert_state_seen.items() if seen_at < cutoff}
    for key in stale_keys:
        _alert_cooldowns.pop(key, None)
        _alert_zone_active.pop(key, None)
        _alert_level_sides.pop(key, None)
        _alert_state_seen.pop(key, None)


async def _asset_type_for_symbol(db: Database, symbol: str) -> str:
    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT asset_type
            FROM watchlist
            WHERE symbol = ? AND is_active = 1
            ORDER BY asset_type = 'crypto' DESC
            LIMIT 1
            """,
            (symbol.upper(),),
        )
        row = await cursor.fetchone()
    if row is None:
        return "crypto"
    asset_type = str(row[0]).lower()
    return asset_type if asset_type in {"stock", "crypto"} else "crypto"


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
        logger.info(
            "Reflect evaluator: using cached evaluator id=%s db_path=%s",
            id(evaluator),
            getattr(evaluator, "_db_path", None),
        )
        return evaluator
    aggregator = _aggregator_from_context(context)
    context.bot_data["reflection_evaluator"] = aggregator.reflection_evaluator
    db = context.bot_data.get("db")
    logger.info(
        "Reflect evaluator: caching aggregator evaluator id=%s eval_db_path=%s db.db_path=%s",
        id(aggregator.reflection_evaluator),
        getattr(aggregator.reflection_evaluator, "_db_path", None),
        getattr(db, "db_path", None),
    )
    return aggregator.reflection_evaluator


async def _latest_briefing_data(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    include_agent_analysis: bool = False,
) -> BriefingData:
    cached = context.bot_data.get("latest_briefing_data")
    if isinstance(cached, BriefingData) and (
        not include_agent_analysis or cached.agent_analysis is not None
    ):
        return cached

    aggregator = _aggregator_from_context(context)
    data = await aggregator.gather_briefing_data(include_agent_analysis=include_agent_analysis)
    _store_latest_briefing(context, data)
    return data


def _store_latest_briefing(context: ContextTypes.DEFAULT_TYPE, data: BriefingData) -> None:
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


def _portfolio_trade_notes(
    *,
    symbol: str,
    asset_type: str,
    signal: TradeSignal,
    portfolio_tracker: PortfolioTracker,
    correlation_engine: object,
) -> str:
    del asset_type
    notes: list[str] = []

    allowed, reason = portfolio_tracker.check_risk_budget(
        signal.suggested_risk_pct,
        replacing_symbol=symbol,
    )
    if not allowed:
        notes.append(f"🚫 <b>RISK LIMIT:</b> {html.escape(reason, quote=False)}")

    correlations: dict[str, float] = {}
    if correlation_engine is not None and hasattr(correlation_engine, "get_pair_analysis"):
        for existing_symbol in portfolio_tracker.positions:
            if existing_symbol == symbol.upper():
                continue
            pair = correlation_engine.get_pair_analysis(symbol.upper(), existing_symbol)
            if pair is not None and hasattr(pair, "corr_20d"):
                correlations[existing_symbol] = float(pair.corr_20d)

    overlap_warning = portfolio_tracker.check_correlation_overlap(symbol, correlations)
    if overlap_warning:
        notes.append(f"⚠️ <b>CORRELATION:</b> {html.escape(overlap_warning, quote=False)}")

    return "\n\n".join(notes)


def _format_playbook_message(
    *,
    generated_at: datetime,
    playbook_text: str,
    llm_calls: int,
    duration: float,
    llm_usage: list[LLMUsageSummary],
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
        lines = [f"<i>\U0001F9E0 {llm_calls} LLM {label} in {duration:.1f}s</i>"]
        for usage in llm_usage:
            call_label = "call" if usage.calls == 1 else "calls"
            lines.append(
                f"<i>   {html.escape(usage.role.title(), quote=False)}: "
                f"{html.escape(_provider_display_name(usage.provider), quote=False)} / "
                f"{html.escape(usage.model_display, quote=False)} ({usage.calls} {call_label})</i>"
            )
        footer = "\n".join(lines)
    return "\n\n".join(part for part in [header, body, footer] if part)


def _provider_display_name(provider: str) -> str:
    return {
        "openrouter": "OpenRouter",
        "groq": "Groq",
        "gemini": "Gemini",
    }.get(provider, provider.replace("_", " ").title())


def _format_trade_text(signal: TradeSignal) -> str:
    direction_emoji = "\U0001F7E2" if signal.direction == "LONG" else "\U0001F534"
    reasoning = html.escape(signal.reasoning, quote=False)
    manipulation = html.escape(signal.manipulation_check, quote=False)
    return (
        f"\U0001F3AF <b>TRADE SIGNAL: {html.escape(signal.symbol)}</b>\n\n"
        f"Direction: {direction_emoji} <b>{html.escape(signal.direction)}</b>\n"
        f"Confidence: {signal.confidence:.0%}\n\n"
        f"<b>Entry zone:</b> {_format_trade_price(signal.entry_low)} - {_format_trade_price(signal.entry_high)}\n"
        f"<b>Stop loss:</b> {_format_trade_price(signal.stop_loss)} (-{signal.stop_loss_pct:.1f}%)\n"
        f"<b>Risk:</b> {_format_trade_price(signal.risk_per_unit)} per unit\n\n"
        f"<b>Take profit targets:</b>\n"
        f"TP1: {_format_trade_price(signal.tp1)} (+{signal.tp1_pct:.1f}%)  conservative\n"
        f"TP2: {_format_trade_price(signal.tp2)} (+{signal.tp2_pct:.1f}%)  moderate\n"
        f"TP3: {_format_trade_price(signal.tp3)} (+{signal.tp3_pct:.1f}%)  aggressive\n\n"
        f"<b>Risk/Reward:</b> 1:{signal.risk_reward} (to TP2)\n"
        f"<b>Timeframe:</b> {html.escape(signal.timeframe)}\n\n"
        f"<i>{reasoning}</i>\n\n"
        f"{manipulation}\n\n"
        "<b>NOT financial advice.</b> Every trade carries risk of total loss. "
        "Never risk more than 2% of your portfolio on a single trade."
    )


def _format_crypto_trade_text(signal: TradeSignal) -> str:
    is_long = signal.direction == "LONG"
    direction_emoji = "\U0001F7E2" if is_long else "\U0001F534"
    direction_label = signal.direction if is_long else "SHORT BIAS"
    confluence_lines = "\n".join(_format_confluence_line(factor) for factor in signal.confluence_factors)
    if not confluence_lines:
        confluence_lines = "No detailed confluence breakdown available."

    zone_label = "Entry zone" if is_long else "Sell/reduce zone"
    stop_label = "Stop-loss sell" if is_long else "Invalidation level"
    risk_label = "Risk per coin if stopped" if is_long else "Giveback risk per coin"
    stop_pct = f"-{signal.stop_loss_pct:.1f}%" if is_long else f"+{signal.stop_loss_pct:.1f}%"
    spot_note = ""
    if not is_long:
        spot_note = (
            "<b>Spot note:</b> This is a bearish spot setup. Use it to reduce exposure, "
            "lock in gains, or wait for a better long entry.\n\n"
        )

    return (
        f"\U0001F3AF <b>SPOT TRADE SETUP: {html.escape(signal.symbol)}</b>\n\n"
        f"Direction: {direction_emoji} <b>{html.escape(direction_label)}</b>\n"
        f"Confidence: {signal.confidence:.0%} | "
        f"Confluence: {signal.confluence_count} factors {_trade_confluence_stars(signal.confluence_count)}\n\n"
        f"{spot_note}"
        f"<b>{zone_label}:</b> {_format_trade_price(signal.entry_low)} - {_format_trade_price(signal.entry_high)}\n"
        f"<b>{stop_label}:</b> {_format_trade_price(signal.stop_loss)} ({stop_pct})\n"
        f"<b>{risk_label}:</b> {_format_trade_price(signal.risk_per_unit)}\n\n"
        f"<b>{'Take-profit sell targets' if is_long else 'Downside checkpoints'}:</b>\n"
        f"TP1: {_format_trade_price(signal.tp1)} ({_trade_target_pct(signal.direction, signal.tp1_pct)})"
        f"{' - sell 1/3' if is_long else ' - first support'}\n"
        f"TP2: {_format_trade_price(signal.tp2)} ({_trade_target_pct(signal.direction, signal.tp2_pct)})"
        f"{' - sell another 1/3' if is_long else ' - deeper support'}\n"
        f"TP3: {_format_trade_price(signal.tp3)} ({_trade_target_pct(signal.direction, signal.tp3_pct)})"
        f"{' - trail the rest' if is_long else ' - stretch target if weakness accelerates'}\n\n"
        f"<b>R:R:</b> 1:{signal.risk_reward} | <b>Timeframe:</b> {html.escape(signal.timeframe)}\n"
        f"<b>Suggested risk:</b> {signal.suggested_risk_pct:.0f}% of portfolio\n\n"
        f"<b>Why this trade:</b>\n{confluence_lines}\n\n"
        f"{html.escape(signal.manipulation_check, quote=False)}\n\n"
        f"{_format_spot_position_sizing_text(signal)}\n\n"
        f"{_format_spot_execution_text(signal)}\n\n"
        "<b>NOT financial advice.</b> Risk no more than 2% of your portfolio per trade idea."
    )


def _format_spot_position_sizing_text(signal: TradeSignal) -> str:
    portfolio_example = 1_000.0
    risk_pct = signal.suggested_risk_pct / 100
    max_risk_amount = portfolio_example * risk_pct
    units = max_risk_amount / signal.risk_per_unit if signal.risk_per_unit > 0 else 0.0

    if signal.direction == "LONG":
        return (
            "<b>Position sizing example (£1,000 portfolio):</b>\n"
            f"Max risk ({signal.suggested_risk_pct:.0f}%): £{max_risk_amount:.2f}\n"
            f"Position size: {units:.4f} {html.escape(signal.symbol)}\n"
            f"Approx notional at entry: {_format_trade_price(units * signal.entry_high)}\n"
            f"If stopped out, you lose: £{max_risk_amount:.2f}\n"
            f"If TP2 hits, you gain: ~£{units * abs(signal.tp2 - signal.entry_high):.2f}"
        )

    return (
        "<b>Position sizing example (£1,000 portfolio):</b>\n"
        f"Max giveback to tolerate ({signal.suggested_risk_pct:.0f}%): £{max_risk_amount:.2f}\n"
        f"Spot size to cap that giveback: {units:.4f} {html.escape(signal.symbol)}\n"
        f"Approx spot value near the sell zone: {_format_trade_price(units * signal.entry_low)}\n"
        f"If price invalidates above {_format_trade_price(signal.stop_loss)}, giveback is about £{max_risk_amount:.2f}\n"
        f"If TP2 hits, you avoid roughly £{units * abs(signal.stop_loss - signal.tp2):.2f} of extra downside"
    )


def _format_spot_execution_text(signal: TradeSignal) -> str:
    if signal.direction == "LONG":
        return (
            "<b>How to execute (spot):</b>\n"
            f"1. Place a limit buy at {_format_trade_price(signal.entry_low)}\n"
            f"2. Set a stop-loss sell at {_format_trade_price(signal.stop_loss)}\n"
            f"3. When TP1 ({_format_trade_price(signal.tp1)}) hits, sell 1/3 of your position\n"
            "4. Move the stop-loss to your entry price (breakeven)\n"
            f"5. When TP2 ({_format_trade_price(signal.tp2)}) hits, sell another 1/3\n"
            "6. Let the last 1/3 ride with a trailing stop\n"
            "7. If nothing happens in 5-7 days, close the trade"
        )

    return (
        "<b>How to execute (spot):</b>\n"
        "1. Treat this as a bearish spot plan, not a fresh buy setup\n"
        f"2. If you already hold the coin, reduce into {_format_trade_price(signal.entry_low)} - {_format_trade_price(signal.entry_high)}\n"
        f"3. Keep a protective stop on any remaining spot at {_format_trade_price(signal.stop_loss)}\n"
        f"4. If price drops to TP1 ({_format_trade_price(signal.tp1)}), reduce another 1/3\n"
        f"5. If TP2 ({_format_trade_price(signal.tp2)}) hits, reduce again or wait for a base to form\n"
        f"6. Re-enter only if price reclaims {_format_trade_price(signal.stop_loss)} with strength\n"
        "7. If price stalls for 5-7 days, stay patient and reassess"
    )


def _format_no_trade_text(symbol: str, *, crypto_style: bool = False) -> str:
    if crypto_style:
        return (
            f"\u26A0\uFE0F <b>NO TRADE - {html.escape(symbol)}</b>\n\n"
            "Not enough confluence signals (need 3+) or risk/reward is too low.\n"
            "The best trade is often no trade. Wait for a better setup.\n\n"
            "<i>Patience is how the small minority who profit separate themselves from the crowd.</i>"
        )
    return (
        f"\u26A0\uFE0F <b>NO TRADE - {html.escape(symbol)}</b>\n\n"
        "Signals are too weak or too conflicting to produce a reliable setup.\n"
        "Wait for clearer conditions."
    )


def _format_confluence_line(factor: object) -> str:
    name = getattr(factor, "description", "")
    strength = float(getattr(factor, "strength", 0.0) or 0.0)
    filled = max(0, min(5, int(round(strength * 5))))
    bar = ("#" * filled) + ("-" * (5 - filled))
    return f"[{bar}] {html.escape(str(name), quote=False)}"


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


def _price_item_for_symbol(data: BriefingData, symbol: str) -> AssetPrice | None:
    symbol_upper = symbol.upper()
    for item in [*data.stock_prices, *data.crypto_prices]:
        if item.symbol.upper() == symbol_upper:
            return item
    return None


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


def _event_payload(event: DetectedEvent | None) -> dict[str, object] | None:
    if event is None:
        return None
    return {
        "event_type": event.event_type.value,
        "remaining_drift_pct": event.remaining_drift_pct,
        "expected_car_pct": event.expected_car_pct,
    }


def _btc_change_pct(data: BriefingData) -> float:
    for item in data.crypto_prices:
        if item.symbol.upper() == "BTC":
            return item.change_24h_pct
    return 0.0


def _event_type_for_symbol(data: BriefingData, symbol: str) -> str | None:
    event = _event_for_symbol(data, symbol)
    if event is None:
        return None
    return event.event_type.value


def _verdict_for_symbol(analysis: AgentAnalysis | None, symbol: str) -> DebateVerdict | None:
    if analysis is None:
        return None
    symbol_upper = symbol.upper()
    for verdict in analysis.debate_verdicts:
        if verdict.symbol.upper() == symbol_upper:
            return verdict
    return None


def _merge_agent_analysis(
    existing: AgentAnalysis | None,
    incoming: AgentAnalysis | None,
) -> AgentAnalysis | None:
    if incoming is None:
        return existing
    if existing is None:
        return incoming

    verdicts: dict[str, DebateVerdict] = {
        verdict.symbol.upper(): verdict
        for verdict in existing.debate_verdicts
    }
    for verdict in incoming.debate_verdicts:
        verdicts[verdict.symbol.upper()] = verdict

    return AgentAnalysis(
        analyst_reports=[*existing.analyst_reports, *incoming.analyst_reports],
        debate_verdicts=list(verdicts.values()),
        total_llm_calls=existing.total_llm_calls + incoming.total_llm_calls,
        pipeline_duration_seconds=(
            existing.pipeline_duration_seconds + incoming.pipeline_duration_seconds
        ),
    )


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


def _format_trade_price(value: float) -> str:
    if value >= 100:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:.2f}"
    return f"${value:.4f}"


def _trade_confluence_stars(count: int) -> str:
    filled = max(0, min(5, count))
    return ("*" * filled) + ("." * (5 - filled))


def _trade_target_pct(direction: str, pct: float) -> str:
    sign = "+" if direction.upper() == "LONG" else "-"
    return f"{sign}{pct:.1f}%"


def _trade_direction_to_prediction(direction: str) -> str:
    return "bullish" if direction.upper() == "LONG" else "bearish"
