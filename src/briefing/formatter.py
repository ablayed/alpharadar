from __future__ import annotations

import html
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.agents.state import AgentAnalysis, DebateVerdict
from src.briefing.models import BriefingData
from src.config import settings
from src.data.base import AssetPrice, DeFiProtocol, NewsItem, SECFiling

MAX_TELEGRAM_LENGTH = 4_096
AI_VERDICT_MAX_LENGTH = 700
INDEX_PROXY_NOTES: dict[str, str] = {
    "SPY": "S&P 500 proxy",
    "QQQ": "NASDAQ proxy",
    "VIXY": "VIX proxy",
}


def _sentiment_indicator(news_item: NewsItem) -> str:
    """Return an emoji indicator based on sentiment score and confidence."""
    if news_item.sentiment_label is None or news_item.sentiment_score is None:
        return ""

    score = news_item.sentiment_score
    confidence = news_item.sentiment_confidence or 0.0

    if score > 0.3 and confidence >= 0.7:
        return "📈 "
    if score < -0.3 and confidence >= 0.7:
        return "📉 "
    if abs(score) > 0.1:
        return "📊 "
    return ""


def _sentiment_summary(news_items: list[NewsItem]) -> str:
    """Compute a compact aggregate sentiment summary across the news set."""
    positive = sum(1 for item in news_items if item.sentiment_label == "positive")
    negative = sum(1 for item in news_items if item.sentiment_label == "negative")
    neutral = sum(
        1
        for item in news_items
        if item.sentiment_label == "neutral" or item.sentiment_label is None
    )

    parts: list[str] = []
    if positive > 0:
        parts.append(f"📈 {positive} pos")
    if negative > 0:
        parts.append(f"📉 {negative} neg")
    if neutral > 0:
        parts.append(f"{neutral} neu")

    if not parts:
        return ""
    return "Sentiment: " + "  ".join(parts)


def _format_event_annotation(news_item: NewsItem) -> str:
    """Format an event tag and calibrated drift summary beneath a headline."""

    if (
        news_item.event_type is None
        or news_item.event_type in {"general_news", "unknown"}
        or (news_item.event_confidence or 0.0) < 0.6
    ):
        return ""

    tag = f"[{html.escape(news_item.event_type, quote=False)}]"
    if news_item.expected_car_pct is None or news_item.expected_car_window is None:
        return f"    {tag}"

    status = "typical drift captured"
    if news_item.remaining_drift_pct is not None and abs(news_item.remaining_drift_pct) > 0.5:
        if (
            (news_item.expected_car_pct > 0 and news_item.remaining_drift_pct > 0)
            or (news_item.expected_car_pct < 0 and news_item.remaining_drift_pct < 0)
        ):
            direction = "upside" if news_item.remaining_drift_pct > 0 else "downside"
            status = f"remaining {abs(news_item.remaining_drift_pct):.1f}% {direction} expected"
        else:
            status = f"overshot by {abs(news_item.remaining_drift_pct):.1f}pp"

    car_line = (
        f"Hist. avg: {news_item.expected_car_pct:+.1f}% over {news_item.expected_car_window}  {status}"
    )
    car_line = BriefingFormatter._truncate(html.escape(car_line, quote=False), 80)
    return f"    {tag}\n    <i>{car_line}</i>"


def _market_context(
    index_prices: list[AssetPrice],
    total_tvl: float | None,
    tvl_change: float | None,
) -> str:
    """Generate a 1-line market context summary from the data."""
    del total_tvl
    parts: list[str] = []

    vix = next((price for price in index_prices if price.symbol.upper() == "VIX"), None)
    if vix and vix.current_price is not None:
        vix_val = vix.current_price
        if vix_val > 30:
            parts.append(f"VIX at {vix_val:.0f} signals high fear")
        elif vix_val > 20:
            parts.append(f"VIX at {vix_val:.0f} shows elevated caution")
        elif vix_val < 15:
            parts.append(f"VIX at {vix_val:.0f} reflects strong complacency")

    sp500 = next(
        (
            price
            for price in index_prices
            if "S&P" in price.symbol or price.symbol.upper() == "SPY"
        ),
        None,
    )
    if sp500:
        change = sp500.change_24h_pct
        if change > 1.5:
            parts.append("broad rally underway")
        elif change < -1.5:
            parts.append("broad selloff in progress")

    if tvl_change is not None:
        if tvl_change > 3:
            parts.append("DeFi capital inflows accelerating")
        elif tvl_change < -3:
            parts.append("DeFi capital outflows notable")

    if not parts:
        return ""

    context = "  ".join(parts)
    return context[0].upper() + context[1:]


def _move_qualifier(change_pct: float, asset_type: str) -> str:
    """Return a move intensity word."""
    abs_change = abs(change_pct)
    if asset_type == "stock":
        if abs_change > 5:
            return "surging" if change_pct > 0 else "plunging"
        if abs_change > 3:
            return "jumping" if change_pct > 0 else "sliding"
        if abs_change > 1.5:
            return "climbing" if change_pct > 0 else "dipping"
        return ""

    if abs_change > 10:
        return "surging" if change_pct > 0 else "crashing"
    if abs_change > 5:
        return "rallying" if change_pct > 0 else "tumbling"
    if abs_change > 3:
        return "climbing" if change_pct > 0 else "dipping"
    return ""


def _trend_context(change_7d: float | None) -> str:
    """Describe the 7-day trend."""
    if change_7d is None:
        return ""
    if change_7d > 15:
        return "strong weekly rally"
    if change_7d > 5:
        return "solid weekly gains"
    if change_7d < -15:
        return "heavy weekly losses"
    if change_7d < -5:
        return "weak weekly trend"
    return ""


def _split_long_block(block: str) -> list[str]:
    parts: list[str] = []
    current = ""

    for line in block.splitlines():
        if len(line) > MAX_TELEGRAM_LENGTH:
            if current:
                parts.append(current)
                current = ""
            remainder = line
            while len(remainder) > MAX_TELEGRAM_LENGTH:
                parts.append(remainder[:MAX_TELEGRAM_LENGTH])
                remainder = remainder[MAX_TELEGRAM_LENGTH:]
            if remainder:
                current = remainder
            continue

        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > MAX_TELEGRAM_LENGTH and current:
            parts.append(current)
            current = line
        else:
            current = candidate

    if current:
        parts.append(current)
    return parts or [block[:MAX_TELEGRAM_LENGTH]]


def _send_briefing_parts(message: str) -> list[str]:
    """Split a long briefing into multiple Telegram messages."""
    if len(message) <= MAX_TELEGRAM_LENGTH:
        return [message]

    sections = [section.strip() for section in message.split("\n\n") if section.strip()]
    parts: list[str] = []
    current = ""

    for section in sections:
        if len(section) > MAX_TELEGRAM_LENGTH:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_long_block(section))
            continue

        candidate = f"{current}\n\n{section}" if current else section
        if len(candidate) > MAX_TELEGRAM_LENGTH and current:
            parts.append(current)
            current = section
        else:
            current = candidate

    if current:
        parts.append(current)

    return parts if parts else [message[:MAX_TELEGRAM_LENGTH]]


class BriefingFormatter:
    def format_daily(self, data: BriefingData) -> str:
        return self._build_message(
            data=data,
            weekend=False,
            include_trending=True,
            include_headlines=True,
        )

    def format_weekend(self, data: BriefingData) -> str:
        return self._build_message(
            data=data,
            weekend=True,
            include_trending=True,
            include_headlines=False,
        )

    def format_on_demand(self, data: BriefingData) -> str:
        return self._build_message(
            data=data,
            weekend=data.is_weekend,
            include_trending=True,
            include_headlines=True,
        )

    def format_error(self, error_msg: str) -> str:
        escaped = self._escape(error_msg)
        return (
            "⚠️ <b>ALPHARADAR BRIEFING ERROR</b>\n"
            "A temporary issue prevented full briefing generation.\n"
            f"<i>{escaped[:250]}</i>"
        )

    def _build_message(
        self,
        data: BriefingData,
        weekend: bool,
        include_trending: bool,
        include_headlines: bool,
    ) -> str:
        sections: list[str] = [
            "📊 <b>ALPHARADAR DAILY BRIEFING</b>\n"
            f"{self._format_header_date(data.generated_at)}  {self._format_market_state(data.market_is_open)}",
            self._format_market_snapshot(data),
        ]

        if weekend:
            sections.extend(
                [
                    self._format_crypto_section(data.crypto_prices),
                    self._format_ai_deep_dive(data.agent_analysis),
                    self._format_defi_movers(data.defi_movers),
                ]
            )
        else:
            sections.extend(
                [
                    self._format_stocks_section(data.stock_prices, data.news_items, include_headlines),
                    self._format_crypto_section(data.crypto_prices),
                    self._format_ai_deep_dive(data.agent_analysis),
                    self._format_catalysts(data),
                ]
            )

        if include_trending and data.trending_crypto:
            sections.append(self._format_trending(data.trending_crypto))

        used, limit = data.coingecko_usage
        sections.append(
            f"<i>CG budget: {used:,}/{limit:,}  Next briefing: "
            f"{self._next_briefing_text(data.generated_at)}</i>"
        )

        if data.failed_sections:
            sections.append(
                "\n".join(
                    f"<i>{self._escape(line)}</i>"
                    for line in self._failed_section_messages(data.failed_sections)
                )
            )

        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _format_header_date(value: datetime) -> str:
        return f"{value.strftime('%A')}, {value.strftime('%B')} {value.day} {value.year}"

    @staticmethod
    def _format_market_state(market_is_open: bool) -> str:
        return "Market Open 🟢" if market_is_open else "Market Closed 🔴"

    def _format_market_snapshot(self, data: BriefingData) -> str:
        lines = ["📈 <b>MARKET SNAPSHOT</b>"]

        for item in data.index_prices:
            lines.append(self._format_index_line(item))

        btc = next((item for item in data.crypto_prices if item.symbol.upper() == "BTC"), None)
        eth = next((item for item in data.crypto_prices if item.symbol.upper() == "ETH"), None)
        if btc is not None or eth is not None:
            market_line_parts = []
            if btc is not None:
                market_line_parts.append(self._format_market_crypto_line("BTC", btc))
            if eth is not None:
                market_line_parts.append(self._format_market_crypto_line("ETH", eth))
            lines.append("  ".join(market_line_parts))

        if data.total_tvl is not None and data.tvl_change_24h_pct is not None:
            lines.append(
                f"Crypto TVL: {self._format_large_currency(data.total_tvl)} "
                f"{self._format_change(data.tvl_change_24h_pct)}"
            )

        summary = _sentiment_summary(data.news_items)
        if summary:
            lines.append(summary)

        context = _market_context(data.index_prices, data.total_tvl, data.tvl_change_24h_pct)
        if context:
            lines.append(f"<i>{self._escape(context)}</i>")

        return "\n".join(lines)

    def _format_stocks_section(
        self,
        stock_prices: list[AssetPrice],
        news_items: list[NewsItem],
        include_headlines: bool,
    ) -> str:
        lines = ["📈 <b>STOCKS</b>"]
        stocks = sorted(stock_prices, key=lambda item: item.change_24h_pct, reverse=True)
        if not stocks:
            lines.append("No stock data available.")
            return "\n".join(lines)

        for item in stocks:
            lines.append(self._format_asset_line(item, asset_type="stock"))
            if include_headlines and abs(item.change_24h_pct) > 1.5:
                news_item = self._news_for_symbol(item.symbol, news_items)
                if news_item:
                    lines.append(f"    {self._truncate(self._escape(news_item.headline), 90)}")
                    annotation = _format_event_annotation(news_item)
                    if annotation:
                        lines.append(annotation)
        return "\n".join(lines)

    def _format_crypto_section(self, crypto_prices: list[AssetPrice]) -> str:
        lines = ["🪙 <b>CRYPTO</b>"]
        if not crypto_prices:
            lines.append("No crypto data available.")
            return "\n".join(lines)

        for item in sorted(crypto_prices, key=lambda price: price.change_24h_pct, reverse=True):
            lines.append(self._format_asset_line(item, asset_type="crypto"))
        return "\n".join(lines)

    def _format_ai_deep_dive(self, analysis: AgentAnalysis | None) -> str:
        if analysis is None or not analysis.debate_verdicts:
            return ""

        lines = ["\U0001F916 <b>AI DEEP DIVE</b>"]
        for verdict in analysis.debate_verdicts[:2]:
            lines.append(
                "\n".join(
                    self._ai_verdict_lines(
                        verdict,
                        include_symbol_header=True,
                        max_length=AI_VERDICT_MAX_LENGTH,
                    )
                )
            )

        lines.append(
            f"<i>\U0001F9E0 Pipeline: {analysis.total_llm_calls} LLM calls in "
            f"{self._format_duration(analysis.pipeline_duration_seconds)}</i>"
        )
        return "\n\n".join(line for line in lines if line)

    def format_agent_verdict(self, symbol: str, analysis: AgentAnalysis) -> str | None:
        symbol_upper = symbol.upper()
        verdict = next(
            (item for item in analysis.debate_verdicts if item.symbol.upper() == symbol_upper),
            None,
        )
        if verdict is None:
            return None

        lines = [
            f"\U0001F916 <b>Agent Analysis: {self._escape(symbol_upper)}</b>",
            "",
            *self._ai_verdict_lines(
                verdict,
                include_symbol_header=False,
                max_length=1200,
            ),
            "",
            f"<i>\U0001F9E0 {analysis.total_llm_calls} LLM calls in "
            f"{self._format_duration(analysis.pipeline_duration_seconds)}</i>",
        ]
        return "\n".join(lines)

    def _format_catalysts(self, data: BriefingData) -> str:
        lines = ["🧭 <b>CATALYSTS</b>"]

        if data.earnings_today:
            for event in data.earnings_today[:5]:
                lines.append(
                    f"📅 {self._escape(event.symbol)} earnings call "
                    f"({self._escape(self._earnings_window(event.hour))})"
                )
        else:
            lines.append("📅 No major catalysts today.")

        for filing in data.sec_filings[:5]:
            lines.append(self._format_filing_line(filing))
        return "\n".join(lines)

    def _ai_verdict_lines(
        self,
        verdict: DebateVerdict,
        *,
        include_symbol_header: bool,
        max_length: int,
    ) -> list[str]:
        summary = self._sanitize_ai_text(verdict.summary, 140)
        bull = self._sanitize_ai_text(verdict.bull_thesis, 95)
        bear = self._sanitize_ai_text(verdict.bear_thesis, 95)
        risk = self._sanitize_ai_text(verdict.key_risk, 80)
        watch = self._sanitize_ai_text(verdict.action_context, 80)

        while True:
            lines: list[str] = []
            if include_symbol_header:
                lines.append(self._format_ai_symbol_header(verdict))
            lines.append(
                f"{self._view_emoji(verdict.prevailing_view)} <b>Verdict:</b> "
                f"{verdict.prevailing_view.title()} ({verdict.confidence:.0%} confidence)"
            )
            if summary:
                lines.extend(["", summary])
            if bull:
                lines.extend(["", f"\U0001F4C8 <b>Bulls say:</b> {bull}"])
            if bear:
                lines.append(f"\U0001F4C9 <b>Bears say:</b> {bear}")
            if risk:
                lines.append(f"\u26A0\uFE0F <b>Key risk:</b> {risk}")
            if watch:
                lines.append(f"\U0001F441 <b>Watch:</b> {watch}")

            block = "\n".join(lines)
            if len(block) <= max_length:
                return lines

            overflow = len(block) - max_length + 3
            if len(summary) > 40:
                summary = self._truncate(summary, max(40, len(summary) - overflow))
                continue
            if len(bull) > 32:
                bull = self._truncate(bull, max(32, len(bull) - overflow))
                continue
            if len(bear) > 32:
                bear = self._truncate(bear, max(32, len(bear) - overflow))
                continue
            if len(risk) > 24:
                risk = self._truncate(risk, max(24, len(risk) - overflow))
                continue
            if len(watch) > 24:
                watch = self._truncate(watch, max(24, len(watch) - overflow))
                continue
            return [self._truncate(block, max_length)]

    def _format_ai_symbol_header(self, verdict: DebateVerdict) -> str:
        selection_suffix = ""
        if verdict.selection_reason:
            selection_suffix = f" (selected: {self._escape(verdict.selection_reason)})"
        return f"\U0001F4CA <b>{self._escape(verdict.display_name)}</b>{selection_suffix}"

    def _sanitize_ai_text(self, value: str, limit: int) -> str:
        collapsed = " ".join(value.split())
        escaped = self._escape(collapsed)
        return self._truncate(escaped, limit)

    @staticmethod
    def _view_emoji(view: str) -> str:
        if view == "bullish":
            return "\U0001F4C8"
        if view == "bearish":
            return "\U0001F4C9"
        return "\u2696\uFE0F"

    @staticmethod
    def _format_duration(value: float) -> str:
        return f"{value:.1f}s"

    def _format_defi_movers(self, movers: list[DeFiProtocol]) -> str:
        lines = ["🏦 <b>DEFI MOVERS</b>"]
        if not movers:
            lines.append("No DeFi mover data available.")
            return "\n".join(lines)
        for mover in movers[:5]:
            direction = "📈" if mover.change_1d_pct >= 0 else "📉"
            lines.append(
                f"{direction} {self._escape(mover.name)} TVL {self._format_large_currency(mover.tvl)} "
                f"({self._format_pct(mover.change_1d_pct)})"
            )
        return "\n".join(lines)

    def _format_trending(self, trending: list[dict[str, object]]) -> str:
        entries: list[str] = []
        for idx, item in enumerate(trending[:5], start=1):
            symbol_obj = item.get("symbol")
            symbol = symbol_obj.upper() if isinstance(symbol_obj, str) else "?"
            entries.append(f"#{idx} {self._escape(symbol)}")
        return "🔥 <b>TRENDING</b>\n" + "  ".join(entries)

    def _news_for_symbol(self, symbol: str, news_items: list[NewsItem]) -> NewsItem | None:
        upper = symbol.upper()
        for item in news_items:
            if upper in {related.upper() for related in item.related_symbols}:
                return item
        return None

    def _format_index_line(self, item: AssetPrice) -> str:
        line = (
            f"{self._escape(item.symbol)}: {self._format_index_value(item)} "
            f"{self._format_change(item.change_24h_pct)}"
        )
        proxy_note = INDEX_PROXY_NOTES.get(item.symbol.upper())
        if proxy_note:
            line += f" ({self._escape(proxy_note)})"
        return line

    def _format_market_crypto_line(self, label: str, item: AssetPrice) -> str:
        return (
            f"{label}: {self._format_crypto_price(item.current_price)} "
            f"{self._format_change(item.change_24h_pct)}"
        )

    def _format_asset_line(self, item: AssetPrice, asset_type: str) -> str:
        pieces: list[str] = []
        indicator = self._indicator(item.change_24h_pct)
        if indicator:
            pieces.append(indicator)
        pieces.append(f"<b>{self._escape(item.symbol)}</b>")

        price = (
            self._format_stock_price(item.current_price)
            if asset_type == "stock"
            else self._format_crypto_price(item.current_price)
        )
        pieces.append(price)

        qualifier = _move_qualifier(item.change_24h_pct, asset_type)
        if qualifier:
            pieces.append(qualifier)
        pieces.append(self._format_pct(item.change_24h_pct))

        line = " ".join(pieces)
        if asset_type == "crypto" and item.change_7d_pct is not None:
            trend_parts = [f"7d: {self._format_pct(item.change_7d_pct)}"]
            trend = _trend_context(item.change_7d_pct)
            if trend:
                trend_parts.append(trend)
            line += f" ({'  '.join(trend_parts)})"
        return line

    def _format_filing_line(self, filing: SECFiling) -> str:
        return (
            f"📋 {self._escape(filing.form_type)}: {self._escape(filing.ticker)}  "
            f"{self._truncate(self._escape(filing.description), 70)} "
            f"(filed {self._escape(self._short_date(filing.filing_date))})"
        )

    @staticmethod
    def _earnings_window(value: str) -> str:
        mapping = {
            "amc": "after close",
            "bmo": "before open",
            "dmh": "during market hours",
        }
        return mapping.get(value.lower(), value.lower())

    @staticmethod
    def _short_date(value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        return f"{parsed.strftime('%b')} {parsed.day}"

    @staticmethod
    def _next_briefing_text(generated_at: datetime) -> str:
        tz = ZoneInfo(settings.timezone)
        localized = generated_at.astimezone(tz)
        next_run = localized.replace(
            hour=settings.briefing_hour,
            minute=settings.briefing_minute,
            second=0,
            microsecond=0,
        )
        if next_run <= localized:
            next_run += timedelta(days=1)

        delta_days = (next_run.date() - localized.date()).days
        day_label = "Today" if delta_days == 0 else "Tomorrow" if delta_days == 1 else next_run.strftime("%A")
        hour_12 = next_run.hour % 12 or 12
        return f"{day_label} {hour_12}:{next_run.strftime('%M %p')}"

    @staticmethod
    def _failed_section_messages(failed_sections: list[str]) -> list[str]:
        mapping = {
            "stock_prices": "Stock data temporarily unavailable",
            "crypto_prices": "Crypto data temporarily unavailable",
            "defi": "DeFi data temporarily unavailable",
            "earnings": "Earnings calendar temporarily unavailable",
            "sec_filings": "SEC filings temporarily unavailable",
            "trending": "Trending crypto temporarily unavailable",
            "index_prices": "Market snapshot temporarily unavailable",
            "market_status": "Market status temporarily unavailable",
            "news": "News feed temporarily unavailable",
            "coingecko_budget": "CoinGecko usage temporarily unavailable",
        }
        return [mapping[section] for section in failed_sections if section in mapping]

    @staticmethod
    def _indicator(change_pct: float) -> str:
        if change_pct >= 1:
            return "🟢"
        if change_pct <= -1:
            return "🔴"
        return ""

    def _format_change(self, value: float | None) -> str:
        if value is None:
            return "n/a"
        indicator = self._indicator(value)
        if indicator:
            return f"{indicator} {self._format_pct(value)}"
        return self._format_pct(value)

    @staticmethod
    def _format_stock_price(price: float) -> str:
        return f"${price:,.2f}"

    @staticmethod
    def _format_crypto_price(price: float) -> str:
        if price >= 1_000:
            return f"${price:,.0f}"
        if price >= 1:
            return f"${price:,.2f}"
        return f"${price:.4g}"

    @staticmethod
    def _format_index_value(item: AssetPrice) -> str:
        if item.symbol.upper() in INDEX_PROXY_NOTES:
            return BriefingFormatter._format_stock_price(item.current_price)
        if item.symbol.upper() == "VIX":
            return f"{item.current_price:,.2f}"
        return f"{item.current_price:,.0f}"

    @staticmethod
    def _format_large_currency(value: float) -> str:
        thresholds = [
            (1_000_000_000_000, "T"),
            (1_000_000_000, "B"),
            (1_000_000, "M"),
            (1_000, "K"),
        ]
        for threshold, suffix in thresholds:
            if abs(value) >= threshold:
                return f"${value / threshold:.1f}{suffix}"
        return f"${value:,.2f}"

    @staticmethod
    def _format_pct(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:+.1f}%"

    @staticmethod
    def _escape(value: str) -> str:
        return html.escape(value, quote=False)

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: max(0, limit - 3)]}..."
