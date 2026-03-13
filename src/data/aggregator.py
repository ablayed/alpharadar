from __future__ import annotations

import asyncio
import logging
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
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.cache import SentimentCache
from src.storage.database import Database, WatchlistItem

if TYPE_CHECKING:
    from src.agents.graph import AgentPipeline
    from src.agents.llm import GeminiClient
    from src.agents.state import AgentAnalysis

logger = logging.getLogger(__name__)
T = TypeVar("T")


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

    def agent_pipeline_status_text(self) -> str:
        if self.settings.gemini_api_key is None:
            return "No GEMINI_API_KEY set"
        if self._agent_pipeline is not None:
            return "Ready (Gemini Flash-Lite)"
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
        headline_symbols = [item.symbol for item in stock_prices if abs(item.change_24h_pct) > 1.5]
        news_items: list[NewsItem] = []
        if headline_symbols and not is_weekend:
            try:
                news_items = await self._fetch_news(headline_symbols, company_names)
                news_items = await self._enrich_with_sentiment(news_items)
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
        return analysis

    async def _ensure_agent_pipeline(self) -> AgentPipeline | None:
        if self.settings.gemini_api_key is None:
            return None
        if self._agent_pipeline is not None:
            return self._agent_pipeline

        try:
            from src.agents.graph import AgentPipeline
            from src.agents.llm import GeminiClient

            api_key = self.settings.gemini_api_key.get_secret_value()
            gemini = GeminiClient(api_key=api_key)
            await gemini.initialize()
            self._gemini_client = gemini
            self._agent_pipeline = AgentPipeline(gemini_client=gemini)
            self._agent_pipeline_init_error = None
        except Exception as exc:
            self._agent_pipeline = None
            self._gemini_client = None
            self._agent_pipeline_init_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Agent pipeline initialization failed; continuing without Gemini")

        return self._agent_pipeline

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
