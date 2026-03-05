from __future__ import annotations

import html
from datetime import datetime

from src.briefing.models import BriefingData
from src.data.base import AssetPrice, DeFiProtocol, NewsItem


class BriefingFormatter:
    def format_daily(self, data: BriefingData) -> str:
        # Phase 2: Insert sentiment indicators (🟢🔴) next to each headline
        message = self._build_message(
            data=data,
            include_time=True,
            weekend=False,
            include_trending=True,
            include_headlines=True,
        )
        return self._enforce_length(data, message, include_time=True, weekend=False)

    def format_weekend(self, data: BriefingData) -> str:
        message = self._build_message(
            data=data,
            include_time=True,
            weekend=True,
            include_trending=True,
            include_headlines=False,
        )
        return self._enforce_length(data, message, include_time=True, weekend=True)

    def format_on_demand(self, data: BriefingData) -> str:
        message = self._build_message(
            data=data,
            include_time=False,
            weekend=data.is_weekend,
            include_trending=True,
            include_headlines=True,
        )
        return self._enforce_length(data, message, include_time=False, weekend=data.is_weekend)

    def format_error(self, error_msg: str) -> str:
        escaped = self._escape(error_msg)
        return (
            "⚠️ <b>ALPHARADAR BRIEFING ERROR</b>\n"
            "A temporary issue prevented full briefing generation.\n"
            f"<i>{escaped[:250]}</i>"
        )

    def _enforce_length(
        self,
        data: BriefingData,
        message: str,
        include_time: bool,
        weekend: bool,
    ) -> str:
        if len(message) <= 3_800:
            return message
        message = self._build_message(
            data=data,
            include_time=include_time,
            weekend=weekend,
            include_trending=False,
            include_headlines=True,
        )
        if len(message) <= 3_800:
            return message
        message = self._build_message(
            data=data,
            include_time=include_time,
            weekend=weekend,
            include_trending=False,
            include_headlines=False,
        )
        if len(message) <= 4_096:
            return message
        return f"{message[:4_090]}..."

    def _build_message(
        self,
        data: BriefingData,
        include_time: bool,
        weekend: bool,
        include_trending: bool,
        include_headlines: bool,
    ) -> str:
        parts: list[str] = [
            "📊 <b>ALPHARADAR DAILY BRIEFING</b>",
            self._format_header_datetime(data.generated_at, include_time=include_time),
            "",
            self._format_market_snapshot(data),
        ]

        if weekend:
            parts.extend(
                [
                    "",
                    self._format_crypto_section(data.crypto_prices, include_all=True),
                    "",
                    self._format_defi_movers(data.defi_movers),
                ]
            )
        else:
            parts.extend(
                [
                    "",
                    self._format_stocks_section(data.stock_prices, data.news_items, include_headlines),
                    "",
                    self._format_crypto_section(data.crypto_prices, include_all=False),
                    "",
                    self._format_catalysts(data),
                ]
            )

        if include_trending and data.trending_crypto:
            parts.extend(["", self._format_trending(data.trending_crypto)])

        used, limit = data.coingecko_usage
        parts.extend(["", f"<i>CG budget: {used:,}/{limit:,} this month</i>"])

        if data.failed_sections:
            for line in self._failed_section_messages(data.failed_sections):
                parts.append(f"<i>{self._escape(line)}</i>")

        return "\n".join(parts)

    def _format_header_datetime(self, value: datetime, include_time: bool) -> str:
        date_part = f"{value.strftime('%A')}, {value.strftime('%B')} {value.day} {value.year}"
        if not include_time:
            return date_part
        hour_12 = value.hour % 12 or 12
        tz_name = value.strftime("%Z")
        time_part = f"{hour_12}:{value.strftime('%M %p')}"
        if tz_name:
            time_part = f"{time_part} {tz_name}"
        return f"{date_part} - {time_part}"

    def _format_market_snapshot(self, data: BriefingData) -> str:
        lines = ["📈 <b>MARKET SNAPSHOT</b>"]
        market_state = "Market Open 🟢" if data.market_is_open else "Market Closed 🔴"
        if data.market_session:
            market_state = f"{market_state} ({self._escape(data.market_session)})"
        lines.append(market_state)

        for item in data.index_prices:
            lines.append(
                f"{self._escape(item.symbol)}: {self._format_number(item.current_price)} "
                f"{self._indicator(item.change_24h_pct)} {self._format_pct(item.change_24h_pct)}"
            )

        btc = next((item for item in data.crypto_prices if item.symbol.upper() == "BTC"), None)
        eth = next((item for item in data.crypto_prices if item.symbol.upper() == "ETH"), None)
        if btc is not None or eth is not None:
            btc_str = (
                f"BTC: {self._format_crypto_price(btc.current_price)} "
                f"{self._indicator(btc.change_24h_pct)} {self._format_pct(btc.change_24h_pct)}"
                if btc
                else ""
            )
            eth_str = (
                f"ETH: {self._format_crypto_price(eth.current_price)} "
                f"{self._indicator(eth.change_24h_pct)} {self._format_pct(eth.change_24h_pct)}"
                if eth
                else ""
            )
            lines.append("  ".join(part for part in [btc_str, eth_str] if part))

        if data.total_tvl is not None and data.tvl_change_24h_pct is not None:
            lines.append(
                f"Crypto TVL: {self._format_large_currency(data.total_tvl)} "
                f"{self._indicator(data.tvl_change_24h_pct)} {self._format_pct(data.tvl_change_24h_pct)}"
            )
        return "\n".join(lines)

    def _format_stocks_section(
        self,
        stock_prices: list[AssetPrice],
        news_items: list[NewsItem],
        include_headlines: bool,
    ) -> str:
        lines = ["📈 <b>STOCKS</b>"]
        movers = sorted(stock_prices, key=lambda item: abs(item.change_24h_pct), reverse=True)[:5]
        if not movers:
            lines.append("No stock data available.")
            return "\n".join(lines)

        for item in movers:
            line = (
                f"{self._indicator(item.change_24h_pct)} "
                f"<b>{self._escape(item.symbol)}</b> "
                f"{self._format_stock_price(item.current_price)} "
                f"({self._format_pct(item.change_24h_pct)})"
            )
            if include_headlines and abs(item.change_24h_pct) >= 1:
                headline = self._headline_for_symbol(item.symbol, news_items)
                if headline:
                    line += f" - {self._truncate(self._escape(headline), 60)}"
                else:
                    line += " - No significant news"
            lines.append(line)
        return "\n".join(lines)

    def _format_crypto_section(self, crypto_prices: list[AssetPrice], include_all: bool) -> str:
        lines = ["🪙 <b>CRYPTO</b>"]
        if not crypto_prices:
            lines.append("No crypto data available.")
            return "\n".join(lines)

        selected = self._select_crypto_assets(crypto_prices, include_all=include_all)
        for item in selected:
            pct_7d = (
                f" / 7d: {self._format_pct(item.change_7d_pct)}"
                if item.change_7d_pct is not None
                else ""
            )
            lines.append(
                f"{self._indicator(item.change_24h_pct)} <b>{self._escape(item.symbol)}</b> "
                f"{self._format_crypto_price(item.current_price)} "
                f"({self._format_pct(item.change_24h_pct)}{pct_7d})"
            )
        return "\n".join(lines)

    def _format_catalysts(self, data: BriefingData) -> str:
        lines = ["🧭 <b>CATALYSTS</b>"]
        has_content = False

        for event in data.earnings_today[:5]:
            lines.append(
                f"📅 {self._escape(event.symbol)} earnings call ({self._escape(event.hour.lower())})"
            )
            has_content = True

        for filing in data.sec_filings[:5]:
            lines.append(
                f"📋 {self._escape(filing.form_type)}: {self._escape(filing.ticker)} - "
                f"{self._truncate(self._escape(filing.description), 80)}"
            )
            has_content = True

        if not has_content:
            lines.append("No major catalysts today.")
        return "\n".join(lines)

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
            if isinstance(symbol_obj, str):
                symbol = symbol_obj.upper()
            else:
                symbol = "?"
            entries.append(f"#{idx} {self._escape(symbol)}")
        return "🔥 <b>TRENDING</b>\n" + "  ".join(entries)

    def _select_crypto_assets(self, prices: list[AssetPrice], include_all: bool) -> list[AssetPrice]:
        sorted_prices = sorted(prices, key=lambda item: abs(item.change_24h_pct), reverse=True)
        if include_all:
            return sorted_prices

        selected = sorted_prices[:5]
        needed = {"BTC", "ETH"}
        selected_symbols = {item.symbol.upper() for item in selected}
        remaining = [item for item in sorted_prices if item.symbol.upper() in needed and item.symbol.upper() not in selected_symbols]
        for candidate in remaining:
            if len(selected) < 5:
                selected.append(candidate)
                continue
            replace_idx = next(
                (i for i, item in reversed(list(enumerate(selected))) if item.symbol.upper() not in needed),
                None,
            )
            if replace_idx is not None:
                selected[replace_idx] = candidate
        return selected

    def _headline_for_symbol(self, symbol: str, news_items: list[NewsItem]) -> str | None:
        upper = symbol.upper()
        for item in news_items:
            if upper in {related.upper() for related in item.related_symbols}:
                return item.headline
        return None

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
            "news": "News feed temporarily unavailable",
        }
        return [mapping[section] for section in failed_sections if section in mapping]

    @staticmethod
    def _indicator(change_pct: float) -> str:
        if change_pct >= 1:
            return "🟢"
        if change_pct <= -1:
            return "🔴"
        return "⚪"

    @staticmethod
    def _format_stock_price(price: float) -> str:
        return f"${price:,.2f}"

    @staticmethod
    def _format_crypto_price(price: float) -> str:
        if price >= 10_000:
            return f"${price:,.0f}"
        if price >= 1:
            return f"${price:,.2f}"
        return f"${price:.4g}"

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:,.2f}"

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
