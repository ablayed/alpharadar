from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar, cast
from zoneinfo import ZoneInfo

from src.briefing.models import BriefingData
from src.config import Settings
from src.data.base import (
    AssetPrice,
    AssetType,
    DataFreshness,
    DeFiProtocol,
    EarningsEvent,
    NewsItem,
    SECFiling,
)
from src.data.coingecko_client import CoinGeckoClient
from src.data.defillama_client import DeFiLlamaClient
from src.data.edgar_client import EdgarClient
from src.data.finnhub_client import FinnhubClient
from src.data.market_indices import MarketIndicesFetcher
from src.events.classifier import EventClassifier
from src.events.models import DetectedEvent
from src.reflection.evaluator import ReflectionEvaluator
from src.reflection.reflector import get_latest_lessons, run_weekly_reflection
from src.reflection.tracker import PredictionTracker
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.cache import SentimentCache
from src.storage.database import Database, WatchlistItem
from src.ta.candles import fetch_crypto_candles, fetch_stock_candles
from src.ta.signals import TASignal, compute_signals

if TYPE_CHECKING:
    from src.agents.graph import AgentPipeline
    from src.agents.llm import GeminiClient
    from src.agents.state import AgentAnalysis

logger = logging.getLogger(__name__)
T = TypeVar("T")
PRICE_TARGET_RE = re.compile(r"\$([0-9][0-9,]*(?:\.\d+)?)")

PLAYBOOK_SYSTEM_PROMPT = """You are a concise financial analyst creating a daily playbook.
For each asset provided, write a brief actionable paragraph in this exact format:

SYMBOL (current_price, change%)
What happened: [1 sentence on the key event or move]
What it means: [1 sentence on why this matters]
Historical context: [1 sentence using CAR data if available]
Watch for: [1 specific trigger - price level, event, or date]
Scenario: If [X happens], expect [Y]. If [Z happens], expect [W].

Rules:
- Be specific with numbers such as price levels, percentages, and dates.
- Reference the detected event type and CAR data when available.
- Do NOT give buy or sell advice. Frame everything as information and scenarios.
- Keep each asset to exactly 6 lines as shown above.
- Include both stocks and crypto in the same response.
- Do not use markdown bullets, code fences, or extra commentary."""


class DataAggregator:
    """
    Fetches data from all providers concurrently.
    If any provider fails, the briefing still generates with available data.
    """

    # Phase 5: Add WebSocket manager for real-time alerts

    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.finnhub = FinnhubClient(
            api_key=settings.finnhub_api_key.get_secret_value(),
            db=db,
            rpm=settings.finnhub_rpm,
        )
        self.coingecko = CoinGeckoClient(
            api_key=settings.coingecko_api_key.get_secret_value(),
            db=db,
            rpm=settings.coingecko_rpm,
            monthly_limit=settings.coingecko_monthly_limit,
            warning_threshold=settings.coingecko_warning_threshold,
        )
        self.defillama = DeFiLlamaClient(db=db, rpm=settings.defillama_rpm)
        self.edgar = EdgarClient(db=db, rps=settings.edgar_rps)
        self.market_indices = MarketIndicesFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_cache = SentimentCache(db=db)
        self.tracker = PredictionTracker(settings.db_path)
        self.reflection_evaluator = ReflectionEvaluator(settings.db_path)
        self._gemini_client: GeminiClient | None = None
        self._agent_pipeline: AgentPipeline | None = None
        self._agent_pipeline_init_error: str | None = None

    async def initialize(self) -> None:
        await asyncio.gather(
            self.finnhub.initialize(),
            self.coingecko.initialize(),
            self.defillama.initialize(),
            self.edgar.initialize(),
        )
        try:
            await self.sentiment_analyzer.initialize()
        except Exception:
            logger.exception("Sentiment analyzer initialization failed; continuing without sentiment")
        await self._ensure_agent_pipeline()

    async def close(self) -> None:
        await asyncio.gather(
            self.finnhub.close(),
            self.coingecko.close(),
            self.defillama.close(),
            self.edgar.close(),
        )
        self.sentiment_analyzer.close()

    async def fetch_stock_prices(self) -> list[AssetPrice]:
        watchlist = await self.db.get_watchlist("stock")
        symbols = [item.symbol for item in watchlist]
        return await self._fetch_stock_prices(symbols)

    async def fetch_crypto_prices(self) -> list[AssetPrice]:
        watchlist = await self.db.get_watchlist("crypto")
        crypto_items = [item for item in watchlist if item.coingecko_id]
        return await self._fetch_crypto_prices(crypto_items)

    async def get_current_price(self, symbol: str, asset_type: str) -> float | None:
        symbol_upper = symbol.upper()
        if asset_type == "stock":
            try:
                quote = await self.finnhub.get_quote(symbol_upper)
            except Exception:
                logger.exception("Failed to fetch current stock price for %s", symbol_upper)
                return None
            return quote.current_price if quote.current_price > 0 else None

        if asset_type == "crypto":
            coin_map = await self._coin_id_map()
            coin_id = coin_map.get(symbol_upper)
            if coin_id is None:
                return None
            try:
                prices = await self.coingecko.get_market_data([coin_id])
            except Exception:
                logger.exception("Failed to fetch current crypto price for %s", symbol_upper)
                return None
            return prices[0].current_price if prices else None

        return None

    async def get_ta_signal(self, symbol: str) -> TASignal | None:
        symbol_upper = symbol.upper()
        stock_symbols = {item.symbol.upper() for item in await self.db.get_watchlist("stock")}
        if symbol_upper in stock_symbols:
            candles = await fetch_stock_candles(
                symbol_upper,
                self.settings.finnhub_api_key.get_secret_value(),
                db_path=self.settings.db_path,
            )
            if candles is None:
                return None
            close, high, low, _volume = candles
            return compute_signals(symbol_upper, close, high, low, timeframe="daily")

        coin_map = await self._coin_id_map()
        coin_id = coin_map.get(symbol_upper)
        if coin_id is None:
            return None
        candles = await fetch_crypto_candles(coin_id, db_path=self.settings.db_path)
        if candles is None:
            return None
        close, high, low, _volume = candles
        return compute_signals(symbol_upper, close, high, low, timeframe="daily")

    async def get_ta_signal_payload(self, symbol: str) -> dict[str, object] | None:
        try:
            signal = await self.get_ta_signal(symbol)
        except Exception:
            logger.exception("TA signal computation failed for %s", symbol.upper())
            return None
        if signal is None:
            return None
        return cast(dict[str, object], asdict(signal))

    async def record_prediction_outcomes(self) -> int:
        return await self.tracker.record_outcomes(self.get_current_price)

    async def run_weekly_reflection(self) -> str | None:
        llm = await self._ensure_llm_client()
        return await run_weekly_reflection(self.settings.db_path, llm)

    async def get_latest_lessons(self) -> str:
        return await get_latest_lessons(self.settings.db_path)

    def agent_pipeline_status_text(self) -> str:
        if self.settings.gemini_api_key is None:
            return "No GEMINI_API_KEY set"
        if self._agent_pipeline is not None:
            return "Ready (Groq Llama 3.3 70B)"
        if self._agent_pipeline_init_error is not None:
            return "Configured but unavailable"
        return "Initializing"

    async def provider_health(self) -> dict[str, bool]:
        results = await asyncio.gather(
            self.finnhub.health_check(),
            self.coingecko.health_check(),
            self.defillama.health_check(),
            self.edgar.health_check(),
            return_exceptions=True,
        )
        return {
            "finnhub": bool(results[0]) if not isinstance(results[0], BaseException) else False,
            "coingecko": bool(results[1]) if not isinstance(results[1], BaseException) else False,
            "defillama": bool(results[2]) if not isinstance(results[2], BaseException) else False,
            "edgar": bool(results[3]) if not isinstance(results[3], BaseException) else False,
        }

    async def gather_briefing_data(self, *, include_agent_analysis: bool = True) -> BriefingData:
        tz = ZoneInfo(self.settings.timezone)
        now = datetime.now(tz)
        is_weekend = now.weekday() >= 5
        today_str = now.date().isoformat()

        watchlist = await self.db.get_watchlist()
        stock_items = [item for item in watchlist if item.asset_type == "stock"]
        crypto_items = [item for item in watchlist if item.asset_type == "crypto" and item.coingecko_id]
        stock_symbols = [item.symbol for item in stock_items]

        market_is_open = False
        market_session: str | None = None
        failed_sections: list[str] = []

        try:
            market_status = await self.finnhub.get_market_status()
            market_is_open = bool(market_status.get("is_open", False))
            market_session = (
                str(market_status["session"]) if isinstance(market_status.get("session"), str) else None
            )
        except Exception:
            logger.exception("Failed to fetch market status")
            failed_sections.append("market_status")

        results = await asyncio.gather(
            self._fetch_stock_prices(stock_symbols),
            self._fetch_crypto_prices(crypto_items),
            self.market_indices.get_index_prices(self.finnhub),
            self._fetch_earnings(today_str, stock_symbols),
            self.edgar.get_recent_8k_filings(stock_symbols),
            self._fetch_defi_data(),
            self.coingecko.get_trending(),
            return_exceptions=True,
        )

        stock_prices = self._result_or_default(
            cast(list[AssetPrice] | BaseException, results[0]),
            [],
            "stock_prices",
            failed_sections,
        )
        crypto_prices = self._result_or_default(
            cast(list[AssetPrice] | BaseException, results[1]),
            [],
            "crypto_prices",
            failed_sections,
        )
        index_prices = self._result_or_default(
            cast(list[AssetPrice] | BaseException, results[2]),
            [],
            "index_prices",
            failed_sections,
        )
        earnings_today = self._result_or_default(
            cast(list[EarningsEvent] | BaseException, results[3]),
            [],
            "earnings",
            failed_sections,
        )
        sec_filings = self._result_or_default(
            cast(list[SECFiling] | BaseException, results[4]),
            [],
            "sec_filings",
            failed_sections,
        )
        defi_default: tuple[float | None, float | None, list[DeFiProtocol]] = (None, None, [])
        defi_data = self._result_or_default(
            cast(tuple[float | None, float | None, list[DeFiProtocol]] | BaseException, results[5]),
            defi_default,
            "defi",
            failed_sections,
        )
        trending = self._result_or_default(
            cast(list[dict[str, object]] | BaseException, results[6]),
            [],
            "trending",
            failed_sections,
        )

        company_names = {item.symbol.upper(): item.display_name for item in stock_items}
        news_items: list[NewsItem] = []
        event_data: list[DetectedEvent] = []
        if stock_symbols:
            try:
                news_items = await self._fetch_news(stock_symbols, company_names)
                if news_items:
                    logger.info(
                        "Events: classifying %s relevant headline(s) across the watchlist (weekend=%s)",
                        len(news_items),
                        is_weekend,
                    )
                    news_items = await self._enrich_with_sentiment(news_items)
                    news_items, event_data = await self._enrich_with_events(
                        news_items,
                        [*stock_prices, *crypto_prices],
                    )
                else:
                    logger.info("Events: no relevant headlines found for the watchlist")
            except Exception:
                logger.exception("Failed to fetch company news")
                failed_sections.append("news")

        try:
            coingecko_usage = await self.coingecko.get_remaining_budget()
        except Exception:
            logger.exception("Failed to fetch CoinGecko usage")
            coingecko_usage = (0, self.settings.coingecko_monthly_limit)
            failed_sections.append("coingecko_budget")

        total_tvl: float | None
        tvl_change_24h_pct: float | None
        defi_movers: list[DeFiProtocol]
        total_tvl, tvl_change_24h_pct, defi_movers = defi_data

        briefing_data = BriefingData(
            generated_at=now,
            is_weekend=is_weekend,
            market_is_open=market_is_open,
            market_session=market_session,
            index_prices=index_prices,
            stock_prices=stock_prices,
            crypto_prices=crypto_prices,
            news_items=news_items,
            earnings_today=earnings_today,
            sec_filings=sec_filings,
            total_tvl=total_tvl,
            tvl_change_24h_pct=tvl_change_24h_pct,
            defi_movers=defi_movers,
            trending_crypto=trending,
            coingecko_usage=coingecko_usage,
            event_data=event_data,
            agent_analysis=None,
            failed_sections=sorted(set(failed_sections)),
        )

        if include_agent_analysis:
            briefing_data.agent_analysis = await self.run_agent_analysis(briefing_data)

        return briefing_data

    async def _fetch_stock_prices(self, symbols: list[str]) -> list[AssetPrice]:
        if not symbols:
            return []
        quotes = await self.finnhub.get_quotes_batch(symbols)
        return [quote for quote in quotes if quote.current_price > 0 or quote.previous_close is not None]

    async def _fetch_crypto_prices(self, crypto_items: list[WatchlistItem]) -> list[AssetPrice]:
        if not crypto_items:
            return []

        coin_ids = [item.coingecko_id for item in crypto_items if item.coingecko_id is not None]
        id_to_symbol = {
            item.coingecko_id: item.symbol
            for item in crypto_items
            if item.coingecko_id is not None
        }

        prices = await self.coingecko.get_market_data(coin_ids)
        if prices:
            return prices

        fallback_prices = await self.defillama.get_coingecko_prices(coin_ids)
        mapped: list[AssetPrice] = []
        for coin_id, price in fallback_prices.items():
            symbol = id_to_symbol.get(coin_id, coin_id.upper())
            mapped.append(
                AssetPrice(
                    symbol=symbol,
                    asset_type=AssetType.CRYPTO,
                    current_price=price,
                    change_24h_pct=0.0,
                    change_7d_pct=None,
                    high_24h=None,
                    low_24h=None,
                    market_cap=None,
                    volume_24h=None,
                    previous_close=None,
                    freshness=DataFreshness.HOURLY,
                    source="defillama",
                )
            )
        return mapped

    async def _fetch_news(
        self,
        symbols: list[str],
        company_names: dict[str, str],
    ) -> list[NewsItem]:
        coroutines = [
            self.finnhub.get_company_news(
                symbol,
                company_name=company_names.get(symbol.upper()),
            )
            for symbol in symbols
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        news: list[NewsItem] = []
        for result in results:
            if isinstance(result, BaseException):
                continue
            news.extend(result)
        news.sort(key=lambda item: item.published_at, reverse=True)
        return news

    async def _enrich_with_sentiment(self, news_items: list[NewsItem]) -> list[NewsItem]:
        if not news_items:
            return news_items

        headlines = [item.headline for item in news_items]
        cached = await self.sentiment_cache.get_cached(headlines)

        uncached_texts: list[str] = []
        seen_hashes: set[str] = set()
        for headline in headlines:
            if cached.get(headline) is not None:
                continue
            headline_hash = self.sentiment_cache.hash_headline(headline)
            if headline_hash in seen_hashes:
                continue
            seen_hashes.add(headline_hash)
            uncached_texts.append(headline)

        logger.info(
            "Sentiment: %s headlines, %s cached, %s to analyze",
            len(headlines),
            len(headlines) - len(uncached_texts),
            len(uncached_texts),
        )

        if uncached_texts:
            try:
                new_results = await self.sentiment_analyzer.analyze(uncached_texts)
                await self.sentiment_cache.store_results(uncached_texts, new_results)
                for headline, result in zip(uncached_texts, new_results, strict=True):
                    cached[headline] = result
            except Exception:
                logger.exception("Sentiment analysis failed; continuing without sentiment")
                return news_items

        for item in news_items:
            sentiment_result = cached.get(item.headline)
            if sentiment_result is None:
                continue
            item.sentiment_label = sentiment_result.label
            item.sentiment_score = sentiment_result.score
            item.sentiment_confidence = sentiment_result.confidence

        return news_items

    async def _enrich_with_events(
        self,
        news_items: list[NewsItem],
        all_prices: list[AssetPrice],
    ) -> tuple[list[NewsItem], list[DetectedEvent]]:
        """Classify headlines into event types and attach CAR calibration."""

        if not news_items:
            return news_items, []

        classifier = EventClassifier(db=self.db, gemini_client=self._gemini_client)
        headline_data: list[dict[str, object]] = []
        for item in news_items:
            actual_move = 0.0
            for symbol in item.related_symbols:
                price = next((candidate for candidate in all_prices if candidate.symbol == symbol), None)
                if price is None:
                    continue
                actual_move = price.change_24h_pct
                break

            headline_data.append(
                {
                    "headline": item.headline,
                    "symbols": item.related_symbols,
                    "actual_move_pct": actual_move,
                }
            )

        try:
            events = await classifier.classify_headlines(headline_data)
            for item, event in zip(news_items, events, strict=True):
                item.event_type = event.event_type.value
                item.event_confidence = event.confidence
                item.expected_car_pct = event.expected_car_pct
                item.expected_car_window = event.expected_car_window
                item.remaining_drift_pct = event.remaining_drift_pct
            return news_items, events
        except Exception:
            logger.exception("Event classification failed; continuing without event data")
            return news_items, []

    async def _fetch_earnings(self, date_value: str, watchlist_symbols: list[str]) -> list[EarningsEvent]:
        earnings = await self.finnhub.get_earnings_calendar(date_value)
        watch = {symbol.upper() for symbol in watchlist_symbols}
        return [event for event in earnings if event.symbol.upper() in watch]

    async def _fetch_defi_data(self) -> tuple[float | None, float | None, list[DeFiProtocol]]:
        total_tvl, change = await self.defillama.get_total_tvl()
        movers = await self.defillama.get_top_movers(5)
        return total_tvl, change, movers

    async def run_agent_analysis(
        self,
        briefing_data: BriefingData,
        *,
        forced_symbols: list[str] | None = None,
        source: str = "briefing",
    ) -> AgentAnalysis | None:
        pipeline = await self._ensure_agent_pipeline()
        if pipeline is None:
            return None

        try:
            analysis = await pipeline.run(briefing_data, forced_symbols=forced_symbols)
        except Exception:
            logger.exception("Agent pipeline failed; continuing without agent analysis")
            return None

        logger.info(
            "Agent pipeline complete: %s calls, %.1fs, %s verdict(s)",
            analysis.total_llm_calls,
            analysis.pipeline_duration_seconds,
            len(analysis.debate_verdicts),
        )
        await self._log_analysis_predictions(analysis, briefing_data, source=source)
        return analysis

    async def generate_playbook(self, briefing_data: BriefingData) -> tuple[str, int, float]:
        notable_assets = self._notable_assets_for_playbook(briefing_data)
        if not notable_assets:
            return ("No assets met today's notable threshold.", 0, 0.0)

        llm = await self._ensure_llm_client()
        if llm is None:
            raise RuntimeError("Groq client is not available")

        asset_blocks: list[str] = []
        for asset in notable_assets:
            asset_blocks.append(
                "\n".join(
                    [
                        f"{asset['symbol']} ({asset['asset_type']}):",
                        f"  Price: {asset['price_text']}",
                        f"  24h change: {asset['change_24h_pct']:+.1f}%",
                        f"  7d change: {asset['change_7d_text']}",
                        f"  Sentiment: {asset['sentiment_label']} ({asset['sentiment_score_text']})",
                        f"  Event: {asset['event_type']} (confidence: {asset['event_confidence_text']})",
                        f"  Historical CAR: {asset['expected_car_text']} over {asset['expected_car_window']}",
                        f"  Remaining drift: {asset['remaining_drift_text']}",
                        f"  Latest headline: {asset['headline']}",
                    ]
                )
            )

        user_prompt = (
            "Create a daily playbook for these assets:\n\n"
            + "\n\n".join(asset_blocks)
            + "\n\nGenerate the playbook following the exact format specified."
        )

        started_at = time.perf_counter()
        response = await llm.generate(
            system_prompt=PLAYBOOK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=max(900, min(2600, 220 * len(notable_assets))),
        )
        duration = round(time.perf_counter() - started_at, 1)
        playbook_text = response.strip()
        await self._log_playbook_predictions(playbook_text, notable_assets)
        return playbook_text, 1, duration

    def _notable_assets_for_playbook(self, briefing_data: BriefingData) -> list[dict[str, object]]:
        latest_news_by_symbol: dict[str, NewsItem] = {}
        for item in briefing_data.news_items:
            for symbol in item.related_symbols:
                latest_news_by_symbol.setdefault(symbol.upper(), item)

        strongest_event_by_symbol: dict[str, DetectedEvent] = {}
        for event in briefing_data.event_data:
            if event.event_type.value in {"general_news", "unknown"}:
                continue
            for symbol in event.related_symbols:
                existing = strongest_event_by_symbol.get(symbol.upper())
                if existing is None or event.confidence > existing.confidence:
                    strongest_event_by_symbol[symbol.upper()] = event

        notable_assets: list[dict[str, object]] = []
        for price in [*briefing_data.stock_prices, *briefing_data.crypto_prices]:
            symbol_upper = price.symbol.upper()
            news_item = latest_news_by_symbol.get(symbol_upper)
            event = strongest_event_by_symbol.get(symbol_upper)
            has_event = event is not None
            if price.asset_type == AssetType.STOCK:
                is_notable = abs(price.change_24h_pct) > 1.0 or has_event
            else:
                is_notable = abs(price.change_24h_pct) > 3.0 or has_event
            if not is_notable:
                continue

            notable_assets.append(
                {
                    "symbol": symbol_upper,
                    "asset_type": price.asset_type.value,
                    "current_price": price.current_price,
                    "price_text": self._playbook_price_text(price),
                    "change_24h_pct": price.change_24h_pct,
                    "change_7d_text": self._playbook_pct_text(price.change_7d_pct),
                    "sentiment_label": news_item.sentiment_label if news_item and news_item.sentiment_label else "N/A",
                    "sentiment_score_text": self._playbook_pct_score(news_item.sentiment_score if news_item else None),
                    "event_type": event.event_type.value if event is not None else "none",
                    "event_confidence_text": f"{event.confidence:.0%}" if event is not None else "N/A",
                    "expected_car_text": self._playbook_pct_text(event.expected_car_pct if event is not None else None),
                    "expected_car_window": event.expected_car_window if event and event.expected_car_window else "N/A",
                    "remaining_drift_text": self._playbook_pct_text(event.remaining_drift_pct if event is not None else None),
                    "headline": news_item.headline if news_item is not None else "N/A",
                    "sort_key": (
                        1 if has_event else 0,
                        abs(price.change_24h_pct),
                        abs(price.change_7d_pct or 0.0),
                    ),
                }
            )

        notable_assets.sort(key=lambda asset: cast(tuple[float, float, float], asset["sort_key"]), reverse=True)
        return notable_assets

    @staticmethod
    def _playbook_pct_text(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:+.1f}%"

    @staticmethod
    def _playbook_pct_score(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:+.2f}"

    @staticmethod
    def _playbook_price_text(price: AssetPrice) -> str:
        if price.asset_type == AssetType.STOCK:
            return f"${price.current_price:,.2f}"
        if price.current_price >= 1_000:
            return f"${price.current_price:,.0f}"
        if price.current_price >= 1:
            return f"${price.current_price:,.2f}"
        return f"${price.current_price:.4g}"

    async def _ensure_agent_pipeline(self) -> AgentPipeline | None:
        if self.settings.gemini_api_key is None:
            return None
        if self._agent_pipeline is not None:
            return self._agent_pipeline

        try:
            from src.agents.graph import AgentPipeline
            gemini = await self._ensure_llm_client()
            if gemini is None:
                return None
            self._agent_pipeline = AgentPipeline(
                gemini_client=gemini,
                ta_signal_fetcher=self.get_ta_signal_payload,
                lessons_provider=self.get_latest_lessons,
            )
            self._agent_pipeline_init_error = None
        except Exception as exc:
            self._agent_pipeline = None
            self._gemini_client = None
            self._agent_pipeline_init_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Agent pipeline initialization failed; continuing without Groq")

        return self._agent_pipeline

    async def _ensure_llm_client(self) -> GeminiClient | None:
        if self.settings.gemini_api_key is None:
            return None
        if self._gemini_client is not None:
            return self._gemini_client

        from src.agents.llm import GeminiClient

        api_key = self.settings.gemini_api_key.get_secret_value()
        gemini = GeminiClient(api_key=api_key)
        await gemini.initialize()
        self._gemini_client = gemini
        return gemini

    async def _coin_id_map(self) -> dict[str, str]:
        watchlist = await self.db.get_watchlist("crypto")
        return {
            item.symbol.upper(): item.coingecko_id
            for item in watchlist
            if item.coingecko_id is not None
        }

    async def _log_analysis_predictions(
        self,
        analysis: AgentAnalysis,
        briefing_data: BriefingData,
        *,
        source: str,
    ) -> None:
        price_lookup = {
            price.symbol.upper(): price
            for price in [*briefing_data.stock_prices, *briefing_data.crypto_prices]
        }
        strongest_event_by_symbol: dict[str, str | None] = {}
        for event in briefing_data.event_data:
            for symbol in event.related_symbols:
                current_event = strongest_event_by_symbol.get(symbol.upper())
                if current_event is None or event.confidence >= 0.6:
                    strongest_event_by_symbol[symbol.upper()] = event.event_type.value

        for verdict in analysis.debate_verdicts:
            symbol_upper = verdict.symbol.upper()
            price = price_lookup.get(symbol_upper)
            if price is None or price.current_price <= 0:
                continue
            predicted_direction = self._direction_from_view(verdict.prevailing_view)
            await self.tracker.log_prediction(
                symbol=symbol_upper,
                asset_type=price.asset_type.value,
                predicted_direction=predicted_direction,
                confidence=verdict.confidence,
                price_at_prediction=price.current_price,
                source=source,
                reasoning=verdict.summary,
                event_type=strongest_event_by_symbol.get(symbol_upper),
                predicted_target=self._extract_price_target(
                    verdict.summary,
                    verdict.bull_thesis,
                    verdict.bear_thesis,
                    verdict.action_context,
                ),
            )

    async def _log_playbook_predictions(
        self,
        playbook_text: str,
        notable_assets: list[dict[str, object]],
    ) -> None:
        if not playbook_text.strip():
            return

        asset_lookup = {
            str(asset["symbol"]).upper(): asset
            for asset in notable_assets
            if isinstance(asset.get("symbol"), str)
        }

        for block in self._split_playbook_sections(playbook_text):
            first_line = block.splitlines()[0] if block.splitlines() else ""
            symbol = first_line.split(" ", 1)[0].strip().upper()
            asset = asset_lookup.get(symbol)
            if asset is None:
                continue
            current_price = float(asset.get("current_price", 0.0) or 0.0)
            if current_price <= 0:
                continue
            direction, confidence = self._infer_playbook_prediction(block)
            await self.tracker.log_prediction(
                symbol=symbol,
                asset_type=str(asset.get("asset_type", "unknown")),
                predicted_direction=direction,
                confidence=confidence,
                price_at_prediction=current_price,
                source="playbook",
                reasoning=block,
                event_type=str(asset.get("event_type", "")) or None,
                predicted_target=self._extract_price_target(block),
            )

    @staticmethod
    def _direction_from_view(view: str) -> str:
        if view in {"bullish", "bearish", "neutral"}:
            return view
        return "neutral"

    @staticmethod
    def _extract_price_target(*texts: str) -> float | None:
        for text in texts:
            for match in PRICE_TARGET_RE.findall(text):
                try:
                    return float(match.replace(",", ""))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _split_playbook_sections(playbook_text: str) -> list[str]:
        return [section.strip() for section in playbook_text.split("\n\n") if section.strip()]

    @staticmethod
    def _infer_playbook_prediction(block: str) -> tuple[str, float]:
        normalized = block.lower()
        bullish_terms = (
            "upside",
            "break above",
            "higher",
            "rally",
            "gains",
            "bounce",
            "surge",
        )
        bearish_terms = (
            "downside",
            "break below",
            "lower",
            "selloff",
            "losses",
            "drop",
            "fade",
        )
        bullish_hits = sum(term in normalized for term in bullish_terms)
        bearish_hits = sum(term in normalized for term in bearish_terms)
        if bullish_hits > bearish_hits:
            return "bullish", 0.55
        if bearish_hits > bullish_hits:
            return "bearish", 0.55
        return "neutral", 0.5

    @staticmethod
    def _result_or_default(
        result: T | BaseException,
        default: T,
        section: str,
        failed_sections: list[str],
    ) -> T:
        if isinstance(result, BaseException):
            logger.exception("Section '%s' failed", section, exc_info=result)
            failed_sections.append(section)
            return default
        return result
