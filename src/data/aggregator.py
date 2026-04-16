from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Awaitable
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar, cast
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from src.briefing.models import BriefingData
from src.config import Settings
from src.agents.providers import LLMRole
from src.correlation import (
    CorrelationEngine,
    CorrelationSnapshot,
    LeadLagResult,
    PairCorrelation,
    RegimeState,
    compute_lead_lag,
    detect_regime,
)
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
from src.data.derivatives import (
    CMEGap,
    FundingData,
    OpenInterestData,
    detect_cme_gaps,
    fetch_funding_rate,
    fetch_open_interest,
)
from src.data.edgar_client import EdgarClient
from src.data.finnhub_client import FinnhubClient
from src.data.macro import (
    BTCDominanceData,
    DXYData,
    FearGreedData,
    fetch_btc_dominance,
    fetch_dxy_data,
    fetch_fear_greed,
)
from src.data.market_indices import MarketIndicesFetcher
from src.data.hot_list import get_active_hot_list
from src.events.classifier import EventClassifier
from src.events.models import DetectedEvent
from src.reflection.evaluator import ReflectionEvaluator
from src.reflection.reflector import get_latest_lessons, run_weekly_reflection
from src.reflection.tracker import PredictionTracker
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.cache import SentimentCache
from src.signals import CompositeScore, SignalInput, compute_composite
from src.storage.database import Database, WatchlistItem
from src.ta.candles import fetch_crypto_candles, fetch_stock_candles
from src.ta.signals import TASignal, compute_signals

if TYPE_CHECKING:
    from src.agents.graph import AgentPipeline
    from src.agents.llm import AgentLLM
    from src.agents.state import AgentAnalysis, LLMUsageSummary
    from src.ta.signals import TASignal

logger = logging.getLogger(__name__)
T = TypeVar("T")
PRICE_TARGET_RE = re.compile(r"\$([0-9][0-9,]*(?:\.\d+)?)")
SIGNAL_CONTEXT_PROXIES = ("SPY", "QQQ", "VIXY")
CRYPTO_SYMBOL_ALIASES = {
    "RENDER": "RNDR",
}
PLAYBOOK_MAX_HEADLINE_CHARS = 160
PLAYBOOK_MAX_TA_SUMMARY_CHARS = 120

PLAYBOOK_SYSTEM_PROMPT = """You are a financial analyst writing a daily playbook for someone
who is not a trader and does not know technical jargon.

For each asset, write exactly this format:

SYMBOL $PRICE (EMOJI DIRECTION)
[one plain-English paragraph of 4 to 6 sentences]

What the paragraph must cover:
1. What happened today in plain English
2. Whether our signals lean bullish, bearish, or neutral, and why in simple terms
3. What price levels matter and what to watch for next
4. A clear if-X-then-Y scenario

Critical rules:
- Never use jargon like RSI, MACD, Bollinger bands, histogram, CAR, correlation coefficient, or composite score.
- Never show raw analytics such as RSI 38.0, correlation 0.63, composite +0.09, or agreement percentages.
- You still receive all of that data internally. Translate it into simple phrases like "momentum is weak", "signals are mixed", or "it is moving with the broader market".
- Use one of these signal headers only: 🟢 Bullish, 🔴 Bearish, ⚪ Neutral.
- The direction in the header must match the provided consensus bias exactly.
- Always include concrete price levels when the data provides them.
- If the inputs conflict, say so plainly, for example: "news is supportive but the trend is still weak."
- Do not give direct buy or sell advice. Frame everything as scenarios and what to watch.
- Keep the tone conversational, like a smart friend explaining the market.
- Include both stocks and crypto in the same response.
- Do not use bullets, code fences, or extra commentary outside the per-asset format."""


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
        self.correlation_engine = CorrelationEngine()
        self.current_regime: RegimeState | None = None
        self._llm_client: AgentLLM | None = None
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
        try:
            self.current_regime = await self.db.latest_regime_state()
        except Exception:
            logger.exception("Failed to load latest stored regime state")
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

    async def fetch_crypto_prices(self, *, include_hot_list: bool = False) -> list[AssetPrice]:
        crypto_items = await self._crypto_watchlist_items(include_hot_list=include_hot_list)
        return await self._fetch_crypto_prices(crypto_items)

    async def _crypto_watchlist_items(self, *, include_hot_list: bool) -> list[WatchlistItem]:
        watchlist = await self.db.get_watchlist("crypto")
        fixed_items = [item for item in watchlist if item.coingecko_id]
        if not include_hot_list:
            return fixed_items

        merged: dict[str, WatchlistItem] = {
            item.symbol.upper(): item
            for item in fixed_items
        }
        settings_obj = getattr(self, "settings", None)
        db_path = getattr(settings_obj, "db_path", None)
        if not db_path:
            return list(merged.values())

        for coin in await get_active_hot_list(db_path):
            symbol = str(coin.get("symbol", "") or "").upper()
            coin_id = str(coin.get("coingecko_id", "") or "")
            if not symbol or not coin_id or symbol in merged:
                continue
            merged[symbol] = WatchlistItem(
                symbol=symbol,
                asset_type="crypto",
                display_name=str(coin.get("name", "") or symbol),
                coingecko_id=coin_id,
            )
        return list(merged.values())

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
            close, high, low, volume, open_ = candles
            return compute_signals(
                symbol_upper,
                close,
                high,
                low,
                timeframe="daily",
                volume=volume,
                open_=open_,
            )

        coin_map = await self._coin_id_map()
        coin_id = coin_map.get(symbol_upper)
        if coin_id is None:
            return None
        candles = await fetch_crypto_candles(coin_id, db_path=self.settings.db_path)
        if candles is None:
            return None
        close, high, low, volume, open_ = candles
        return compute_signals(
            symbol_upper,
            close,
            high,
            low,
            timeframe="daily",
            volume=volume,
            open_=open_,
        )

    async def get_ta_signal_payload(self, symbol: str) -> dict[str, object] | None:
        try:
            signal = await self.get_ta_signal(symbol)
        except Exception:
            logger.exception("TA signal computation failed for %s", symbol.upper())
            return None
        if signal is None:
            return None
        payload = cast(dict[str, object], asdict(signal))
        for key in ("_close", "_high", "_low", "_volume", "_open"):
            payload.pop(key, None)
        return payload

    async def update_correlation_engine(self, *, persist: bool = True) -> dict[int, CorrelationSnapshot]:
        watchlist = await self.db.get_watchlist()
        stock_symbols = [item.symbol for item in watchlist if item.asset_type == "stock"]
        crypto_items = [item for item in watchlist if item.asset_type == "crypto" and item.coingecko_id]
        candle_map = await self._load_signal_candles(stock_symbols, crypto_items, include_proxies=True)
        return await self._refresh_correlation_from_candles(candle_map, persist=persist)

    async def refresh_regime_state(
        self,
        *,
        index_prices: list[AssetPrice] | None = None,
        crypto_prices: list[AssetPrice] | None = None,
        snapshots: dict[int, CorrelationSnapshot] | None = None,
        persist: bool = True,
    ) -> RegimeState | None:
        if snapshots is None:
            snapshots = self.correlation_engine.compute_all_windows()
            if not snapshots:
                snapshots = await self.update_correlation_engine(persist=persist)

        if index_prices is None or crypto_prices is None:
            watchlist = await self.db.get_watchlist()
            crypto_items = [item for item in watchlist if item.asset_type == "crypto" and item.coingecko_id]
            index_prices, crypto_prices = await asyncio.gather(
                self.market_indices.get_index_prices(self.finnhub),
                self._fetch_crypto_prices(crypto_items),
            )

        btc_qqq_pair = self.correlation_engine.get_pair_analysis("BTC", "QQQ")
        regime_state = detect_regime(
            btc_qqq_corr_20d=(btc_qqq_pair.corr_20d if btc_qqq_pair is not None else None),
            btc_qqq_corr_60d=(btc_qqq_pair.corr_60d if btc_qqq_pair is not None else None),
            vix_proxy_change=self._price_change_for_symbols(index_prices, {"VIX", "VIXY"}),
            spy_change_pct=self._price_change_for_symbols(index_prices, {"SPY", "S&P 500"}),
            btc_change_pct=self._price_change_for_symbols(crypto_prices, {"BTC"}),
        )
        self.current_regime = regime_state
        if persist:
            await self.db.store_regime_state(regime_state)
        return regime_state

    async def _build_phase6_context(
        self,
        *,
        stock_items: list[WatchlistItem],
        crypto_items: list[WatchlistItem],
        stock_prices: list[AssetPrice],
        crypto_prices: list[AssetPrice],
        index_prices: list[AssetPrice],
        news_items: list[NewsItem],
        event_data: list[DetectedEvent],
        agent_analysis: AgentAnalysis | None = None,
        persist: bool = False,
    ) -> tuple[
        dict[str, TASignal],
        dict[str, PairCorrelation],
        dict[str, LeadLagResult],
        list[PairCorrelation],
        RegimeState | None,
        dict[str, CompositeScore],
    ]:
        candle_map = await self._load_signal_candles(
            [item.symbol for item in stock_items],
            crypto_items,
            include_proxies=True,
        )
        ta_signals = self._ta_signal_map_from_candles(
            candle_map,
            {item.symbol.upper() for item in [*stock_items, *crypto_items]},
        )
        snapshots = await self._refresh_correlation_from_candles(candle_map, persist=persist)
        correlation_pairs = self._build_asset_correlation_pairs([*stock_prices, *crypto_prices])
        lead_lag_pairs = self._build_lead_lag_pairs()
        correlation_top_movers = self.correlation_engine.get_top_movers(3)
        regime_state = await self.refresh_regime_state(
            index_prices=index_prices,
            crypto_prices=crypto_prices,
            snapshots=snapshots,
            persist=persist,
        )
        composite_scores = self._compute_composite_scores(
            stock_prices=stock_prices,
            crypto_prices=crypto_prices,
            news_items=news_items,
            event_data=event_data,
            ta_signals=ta_signals,
            correlation_pairs=correlation_pairs,
            regime_state=regime_state,
            agent_analysis=agent_analysis,
        )
        return (
            ta_signals,
            correlation_pairs,
            lead_lag_pairs,
            correlation_top_movers,
            regime_state,
            composite_scores,
        )

    async def _load_signal_candles(
        self,
        stock_symbols: list[str],
        crypto_items: list[WatchlistItem],
        *,
        include_proxies: bool,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        candle_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        stock_universe = {symbol.upper() for symbol in stock_symbols}
        if include_proxies:
            stock_universe.update(SIGNAL_CONTEXT_PROXIES)

        stock_jobs = [
            fetch_stock_candles(
                symbol,
                self.settings.finnhub_api_key.get_secret_value(),
                db_path=self.settings.db_path,
            )
            for symbol in sorted(stock_universe)
        ]
        crypto_symbols: list[str] = []
        crypto_jobs: list[
            asyncio.Future[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None]
            | asyncio.Task[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None]
            | object
        ] = []
        for item in crypto_items:
            if item.coingecko_id is None:
                continue
            crypto_symbols.append(item.symbol.upper())
            crypto_jobs.append(fetch_crypto_candles(item.coingecko_id, db_path=self.settings.db_path))

        stock_results = await asyncio.gather(*stock_jobs, return_exceptions=True)
        for symbol, result in zip(sorted(stock_universe), stock_results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Signal candle fetch failed for %s: %s", symbol, result)
                continue
            if result is not None:
                candle_map[symbol] = result

        crypto_results = await asyncio.gather(*crypto_jobs, return_exceptions=True)
        for symbol, result in zip(crypto_symbols, crypto_results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Signal candle fetch failed for %s: %s", symbol, result)
                continue
            if result is not None:
                candle_map[symbol] = result

        return candle_map

    def _ta_signal_map_from_candles(
        self,
        candle_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        symbols: set[str],
    ) -> dict[str, TASignal]:
        ta_signals: dict[str, TASignal] = {}
        for symbol in sorted(symbols):
            candles = candle_map.get(symbol.upper())
            if candles is None:
                continue
            try:
                close, high, low, volume, open_ = candles
                ta_signals[symbol.upper()] = compute_signals(
                    symbol.upper(),
                    close,
                    high,
                    low,
                    timeframe="daily",
                    volume=volume,
                    open_=open_,
                )
            except Exception:
                logger.exception("TA signal build failed for %s", symbol.upper())
        return ta_signals

    async def _refresh_correlation_from_candles(
        self,
        candle_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        *,
        persist: bool,
    ) -> dict[int, CorrelationSnapshot]:
        close_prices = {
            symbol: candles[0]
            for symbol, candles in candle_map.items()
            if len(candles[0]) >= 20
        }
        self.correlation_engine.update_prices(close_prices)
        snapshots = self.correlation_engine.compute_all_windows()
        if persist and snapshots:
            await self._store_correlation_snapshots(snapshots)
        return snapshots

    async def _store_correlation_snapshots(self, snapshots: dict[int, CorrelationSnapshot]) -> None:
        rows: list[tuple[int, str, str, float, float | None, str | None]] = []
        for window, snapshot in snapshots.items():
            symbols = snapshot.symbols
            for idx_a, symbol_a in enumerate(symbols):
                for idx_b in range(idx_a + 1, len(symbols)):
                    symbol_b = symbols[idx_b]
                    correlation = float(snapshot.matrix[idx_a, idx_b])
                    z_score: float | None = None
                    regime: str | None = None
                    if window == 20:
                        pair = self.correlation_engine.get_pair_analysis(symbol_a, symbol_b)
                        if pair is not None:
                            z_score = pair.z_score
                            regime = pair.regime_signal
                    rows.append((window, symbol_a, symbol_b, correlation, z_score, regime))
        await self.db.store_correlation_rows(rows)

    def _build_asset_correlation_pairs(self, prices: list[AssetPrice]) -> dict[str, PairCorrelation]:
        pairs: dict[str, PairCorrelation] = {}
        for price in prices:
            symbol = price.symbol.upper()
            partner = self._correlation_partner(symbol, price.asset_type)
            pair = self.correlation_engine.get_pair_analysis(symbol, partner)
            if pair is None and price.asset_type == AssetType.STOCK:
                pair = self.correlation_engine.get_pair_analysis(symbol, "QQQ")
            if pair is None and price.asset_type == AssetType.CRYPTO:
                pair = self.correlation_engine.get_pair_analysis(symbol, "ETH")
            if pair is not None:
                pairs[symbol] = pair
        return pairs

    def _build_lead_lag_pairs(self) -> dict[str, LeadLagResult]:
        lead_lag: dict[str, LeadLagResult] = {}
        for symbol_a, symbol_b in [("QQQ", "BTC"), ("SPY", "BTC"), ("BTC", "ETH"), ("ETH", "SOL"), ("NVDA", "BTC")]:
            returns_a = self.correlation_engine.get_returns(symbol_a)
            returns_b = self.correlation_engine.get_returns(symbol_b)
            if returns_a is None or returns_b is None:
                continue
            lead_lag[f"{symbol_a}:{symbol_b}"] = compute_lead_lag(returns_a, returns_b, symbol_a, symbol_b)
        return lead_lag

    def _compute_composite_scores(
        self,
        *,
        stock_prices: list[AssetPrice],
        crypto_prices: list[AssetPrice],
        news_items: list[NewsItem],
        event_data: list[DetectedEvent],
        ta_signals: dict[str, TASignal],
        correlation_pairs: dict[str, PairCorrelation],
        regime_state: RegimeState | None,
        agent_analysis: AgentAnalysis | None,
    ) -> dict[str, CompositeScore]:
        prices = [*stock_prices, *crypto_prices]
        regime_key = regime_state.regime.value if regime_state is not None else "default"
        verdicts = {
            verdict.symbol.upper(): verdict
            for verdict in (agent_analysis.debate_verdicts if agent_analysis is not None else [])
        }

        composite_scores: dict[str, CompositeScore] = {}
        for price in prices:
            symbol = price.symbol.upper()
            inputs = self._build_signal_inputs(
                price=price,
                news_items=news_items,
                event_data=event_data,
                ta_signal=ta_signals.get(symbol),
                correlation_pair=correlation_pairs.get(symbol),
                regime_state=regime_state,
                agent_verdict=verdicts.get(symbol),
            )
            composite_scores[symbol] = compute_composite(symbol, inputs, regime=regime_key)
        return composite_scores

    def apply_agent_analysis(
        self,
        briefing_data: BriefingData,
        agent_analysis: AgentAnalysis | None,
    ) -> BriefingData:
        briefing_data.agent_analysis = agent_analysis
        if agent_analysis is None:
            return briefing_data

        briefing_data.composite_scores = self._compute_composite_scores(
            stock_prices=briefing_data.stock_prices,
            crypto_prices=briefing_data.crypto_prices,
            news_items=briefing_data.news_items,
            event_data=briefing_data.event_data,
            ta_signals=briefing_data.ta_signals,
            correlation_pairs=briefing_data.correlation_pairs,
            regime_state=briefing_data.regime_state,
            agent_analysis=agent_analysis,
        )
        return briefing_data

    def _build_signal_inputs(
        self,
        *,
        price: AssetPrice,
        news_items: list[NewsItem],
        event_data: list[DetectedEvent],
        ta_signal: TASignal | None,
        correlation_pair: PairCorrelation | None,
        regime_state: RegimeState | None,
        agent_verdict: object | None,
    ) -> list[SignalInput]:
        now = time.time()
        symbol = price.symbol.upper()
        symbol_news = [
            item for item in news_items if symbol in {related.upper() for related in item.related_symbols}
        ]
        symbol_events = [
            item for item in event_data if symbol in {related.upper() for related in item.related_symbols}
        ]
        signals: list[SignalInput] = []

        if symbol_news:
            strongest_news = max(
                symbol_news,
                key=lambda item: (
                    abs(item.sentiment_score or 0.0),
                    self._timestamp_from_iso(item.published_at),
                ),
            )
            if strongest_news.sentiment_score is not None:
                signals.append(
                    SignalInput(
                        name="sentiment",
                        value=float(np.clip(strongest_news.sentiment_score, -1.0, 1.0)),
                        confidence=float(strongest_news.sentiment_confidence or 0.5),
                        timestamp=self._timestamp_from_iso(strongest_news.published_at),
                        half_life_hours=12.0,
                    )
                )

        if ta_signal is not None:
            signals.append(
                SignalInput(
                    name="ta",
                    value=self._normalize_ta_signal(ta_signal.overall_signal),
                    confidence=float(ta_signal.confidence),
                    timestamp=now,
                    half_life_hours=72.0,
                )
            )

        if symbol_events:
            strongest_event = max(
                symbol_events,
                key=lambda item: abs(item.remaining_drift_pct or item.expected_car_pct or 0.0) * item.confidence,
            )
            event_value = strongest_event.remaining_drift_pct
            if event_value is None:
                event_value = strongest_event.expected_car_pct
            if event_value is not None:
                signals.append(
                    SignalInput(
                        name="event",
                        value=float(np.clip(event_value / 5.0, -1.0, 1.0)),
                        confidence=float(strongest_event.confidence),
                        timestamp=self._timestamp_for_event(strongest_event, symbol_news, now),
                        half_life_hours=48.0,
                    )
                )

        correlation_signal = self._correlation_signal_input(price, correlation_pair, regime_state, now)
        if correlation_signal is not None:
            signals.append(correlation_signal)

        if agent_verdict is not None:
            prevailing_view = getattr(agent_verdict, "prevailing_view", "neutral")
            confidence = getattr(agent_verdict, "confidence", 0.5)
            signals.append(
                SignalInput(
                    name="agent",
                    value=self._normalize_agent_view(str(prevailing_view)),
                    confidence=float(confidence),
                    timestamp=now,
                    half_life_hours=72.0,
                )
            )

        return signals

    def _correlation_signal_input(
        self,
        price: AssetPrice,
        pair: PairCorrelation | None,
        regime_state: RegimeState | None,
        now: float,
    ) -> SignalInput | None:
        regime_key = regime_state.regime.value if regime_state is not None else "default"
        if regime_key == "risk_on":
            base_value = 0.25
        elif regime_key == "risk_off":
            base_value = -0.25
        elif regime_key == "divergent":
            base_value = 0.30 if price.asset_type == AssetType.CRYPTO else -0.10
        elif regime_key == "volatile":
            base_value = 0.0
        else:
            base_value = 0.05

        pair_bias = 0.0
        confidence = 0.35
        if pair is not None:
            if pair.regime_signal == "coupling":
                pair_bias = 0.15
            elif pair.regime_signal == "decoupling":
                pair_bias = 0.15 if price.asset_type == AssetType.CRYPTO else -0.10
            elif pair.regime_signal == "anomalous":
                sign = math.copysign(1.0, pair.z_score if pair.z_score != 0 else pair.divergence or 1.0)
                pair_bias = 0.15 * sign
            confidence = min(0.95, 0.45 + min(0.25, abs(pair.divergence)) + min(0.20, abs(pair.z_score) * 0.05))

        value = float(np.clip(base_value + pair_bias, -1.0, 1.0))
        return SignalInput(
            name="correlation",
            value=value,
            confidence=confidence,
            timestamp=now,
            half_life_hours=168.0,
        )

    @staticmethod
    def _correlation_partner(symbol: str, asset_type: AssetType) -> str:
        if asset_type == AssetType.STOCK:
            return "QQQ" if symbol in {"SPY", "QQQ"} else "SPY"
        return "QQQ" if symbol == "BTC" else "BTC"

    @staticmethod
    def _price_change_for_symbols(prices: list[AssetPrice], symbols: set[str]) -> float:
        target_symbols = {symbol.upper() for symbol in symbols}
        for price in prices:
            if price.symbol.upper() in target_symbols:
                return price.change_24h_pct
        return 0.0

    @staticmethod
    def _timestamp_from_iso(value: str) -> float:
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return time.time()

    @staticmethod
    def _timestamp_for_event(
        event: DetectedEvent,
        symbol_news: list[NewsItem],
        fallback: float,
    ) -> float:
        for item in symbol_news:
            if item.headline == event.headline:
                return DataAggregator._timestamp_from_iso(item.published_at)
        return fallback

    @staticmethod
    def _normalize_ta_signal(signal: str) -> float:
        mapping = {
            "strong_buy": 1.0,
            "buy": 0.5,
            "neutral": 0.0,
            "sell": -0.5,
            "strong_sell": -1.0,
        }
        return mapping.get(signal, 0.0)

    @staticmethod
    def _normalize_agent_view(view: str) -> float:
        mapping = {
            "bullish": 1.0,
            "neutral": 0.0,
            "bearish": -1.0,
        }
        return mapping.get(view, 0.0)

    async def record_prediction_outcomes(self, ensemble: object | None = None) -> int:
        return await self.tracker.record_outcomes(self.get_current_price, ensemble=ensemble)

    async def run_weekly_reflection(self) -> str | None:
        llm = await self._ensure_llm_client()
        return await run_weekly_reflection(self.settings.db_path, llm)

    async def get_latest_lessons(self) -> str:
        return await get_latest_lessons(self.settings.db_path)

    def agent_pipeline_status_text(self) -> str:
        if self.settings.llm_api_key is None:
            return "No OPENROUTER_API_KEY or GROQ_API_KEY set"
        llm_status = self.llm_provider_status()
        if self._agent_pipeline is not None:
            if llm_status["openrouter"] and llm_status["groq"]:
                return "Ready | OpenRouter primary with Groq fallback"
            if llm_status["openrouter"]:
                return "Ready | OpenRouter only"
            if llm_status["groq"]:
                return "Ready | Groq-only fallback mode"
            return "Ready"
        if self._agent_pipeline_init_error is not None:
            return "Configured but unavailable"
        return "Initializing"

    def llm_provider_status(self) -> dict[str, bool]:
        return {
            "openrouter": self.settings.has_openrouter_api_key,
            "groq": self.settings.has_groq_api_key,
        }

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
        fixed_crypto_items = [item for item in watchlist if item.asset_type == "crypto" and item.coingecko_id]
        hot_list = await get_active_hot_list(self.settings.db_path)
        crypto_items_by_symbol: dict[str, WatchlistItem] = {
            item.symbol.upper(): item
            for item in fixed_crypto_items
        }
        for coin in hot_list:
            symbol = str(coin.get("symbol", "") or "").upper()
            coin_id = str(coin.get("coingecko_id", "") or "")
            if not symbol or not coin_id or symbol in crypto_items_by_symbol:
                continue
            crypto_items_by_symbol[symbol] = WatchlistItem(
                symbol=symbol,
                asset_type="crypto",
                display_name=str(coin.get("name", "") or symbol),
                coingecko_id=coin_id,
            )
        crypto_items = list(crypto_items_by_symbol.values())
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

        ta_signals: dict[str, TASignal] = {}
        correlation_pairs: dict[str, PairCorrelation] = {}
        lead_lag_pairs: dict[str, LeadLagResult] = {}
        correlation_top_movers: list[PairCorrelation] = []
        regime_state: RegimeState | None = None
        composite_scores: dict[str, CompositeScore] = {}
        funding: dict[str, FundingData] = {}
        oi: dict[str, OpenInterestData] = {}
        fear_greed: FearGreedData | None = None
        dxy: DXYData | None = None
        cme_gap: CMEGap | None = None
        btc_dominance: BTCDominanceData | None = None
        try:
            (
                ta_signals,
                correlation_pairs,
                lead_lag_pairs,
                correlation_top_movers,
                regime_state,
                composite_scores,
            ) = await self._build_phase6_context(
                stock_items=stock_items,
                crypto_items=crypto_items,
                stock_prices=stock_prices,
                crypto_prices=crypto_prices,
                index_prices=index_prices,
                news_items=news_items,
                event_data=event_data,
                persist=False,
            )
        except Exception:
            logger.exception("Failed to compute Phase 6 signal context")
            failed_sections.append("signal_context")

        crypto_symbols = sorted({item.symbol.upper() for item in crypto_items})
        btc_price = next(
            (price.current_price for price in crypto_prices if price.symbol.upper() == "BTC"),
            0.0,
        )
        async_jobs: list[tuple[str, str | None, Awaitable[object]]] = []
        results: list[object | BaseException] = []
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                for symbol in crypto_symbols:
                    async_jobs.append(("funding", symbol, fetch_funding_rate(client, symbol)))
                    async_jobs.append(("oi", symbol, fetch_open_interest(client, symbol)))
                async_jobs.extend(
                    [
                        ("fear_greed", None, fetch_fear_greed(client)),
                        ("btc_dominance", None, fetch_btc_dominance(client)),
                        ("dxy", None, fetch_dxy_data()),
                        ("cme_gap", None, detect_cme_gaps(btc_price)),
                    ]
                )
                results = cast(
                    list[object | BaseException],
                    await asyncio.gather(
                        *(job[2] for job in async_jobs),
                        return_exceptions=True,
                    ),
                )
        except Exception:
            logger.exception("Failed to fetch derivatives and macro context")
            failed_sections.extend(["derivatives", "macro_context"])

        derivative_failed = False
        macro_failed = False
        for (kind, symbol, _job), result in zip(async_jobs, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "Context task %s failed for %s",
                    kind,
                    symbol or "global",
                    exc_info=result,
                )
                if kind in {"funding", "oi"}:
                    derivative_failed = True
                else:
                    macro_failed = True
                continue

            if kind == "funding" and symbol is not None and isinstance(result, FundingData):
                funding[symbol] = result
            elif kind == "oi" and symbol is not None and isinstance(result, OpenInterestData):
                oi[symbol] = result
            elif kind == "fear_greed" and isinstance(result, FearGreedData):
                fear_greed = result
            elif kind == "btc_dominance" and isinstance(result, BTCDominanceData):
                btc_dominance = result
            elif kind == "dxy" and isinstance(result, DXYData):
                dxy = result
            elif kind == "cme_gap" and isinstance(result, CMEGap):
                cme_gap = result

        if derivative_failed:
            failed_sections.append("derivatives")
        if macro_failed:
            failed_sections.append("macro_context")

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
            hot_list=hot_list,
            coingecko_usage=coingecko_usage,
            event_data=event_data,
            ta_signals=ta_signals,
            correlation_pairs=correlation_pairs,
            lead_lag_pairs=lead_lag_pairs,
            correlation_top_movers=correlation_top_movers,
            regime_state=regime_state,
            composite_scores=composite_scores,
            agent_analysis=None,
            funding=funding,
            oi=oi,
            fear_greed=fear_greed,
            dxy=dxy,
            cme_gap=cme_gap,
            btc_dominance=btc_dominance,
            failed_sections=sorted(set(failed_sections)),
        )

        if include_agent_analysis:
            analysis = await self.run_agent_analysis(briefing_data)
            if analysis is not None:
                try:
                    briefing_data = self.apply_agent_analysis(briefing_data, analysis)
                except Exception:
                    logger.exception("Failed to recompute composite scores with agent context")

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

        classifier = EventClassifier(db=self.db, gemini_client=self._llm_client)
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

    async def generate_playbook(
        self,
        briefing_data: BriefingData,
    ) -> tuple[str, int, float, list[LLMUsageSummary]]:
        notable_assets = self._notable_assets_for_playbook(briefing_data)
        if not notable_assets:
            return ("No assets met today's notable threshold.", 0, 0.0, [])

        llm = await self._ensure_llm_client()
        if llm is None:
            raise RuntimeError("No LLM provider is available")

        asset_blocks = [
            self._build_playbook_asset_block(asset, briefing_data)
            for asset in notable_assets
        ]

        user_prompt = (
            "Create a daily playbook for these assets:\n\n"
            + "\n\n".join(asset_blocks)
            + "\n\nGenerate the playbook following the exact format specified."
        )

        started_at = time.perf_counter()
        usage_marker = llm.mark_usage()
        response = await llm.generate(
            system_prompt=PLAYBOOK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=max(900, min(2600, 220 * len(notable_assets))),
            role=LLMRole.PLAYBOOK,
        )
        duration = round(time.perf_counter() - started_at, 1)
        llm_usage = llm.usage_since(usage_marker)
        playbook_text = response.strip()
        await self._log_playbook_predictions(playbook_text, notable_assets)
        return playbook_text, 1, duration, llm_usage

    def _build_playbook_asset_block(
        self,
        asset: dict[str, object],
        briefing_data: BriefingData,
    ) -> str:
        symbol = str(asset.get("symbol", "")).upper()
        ta_signal = briefing_data.ta_signals.get(symbol)
        headline = self._truncate_playbook_field(asset.get("headline"), PLAYBOOK_MAX_HEADLINE_CHARS)
        ta_summary = self._truncate_playbook_field(
            ta_signal.summary if ta_signal is not None else "N/A",
            PLAYBOOK_MAX_TA_SUMMARY_CHARS,
        )
        agent_view = str(asset.get("agent_view", "N/A"))
        agent_text = (
            agent_view
            if agent_view == "N/A"
            else f"{agent_view} ({asset.get('agent_confidence_text', 'N/A')})"
        )

        context_parts = [f"Regime={asset.get('regime', 'N/A')}"]
        pair_label = str(asset.get("pair_label", "N/A"))
        pair_regime = str(asset.get("pair_regime", "N/A"))
        if pair_label != "N/A" or pair_regime != "N/A":
            context_parts.append(f"Pair={pair_label} {pair_regime}".strip())
        btc_qqq_corr = str(asset.get("btc_qqq_corr_text", "N/A"))
        if btc_qqq_corr != "N/A":
            context_parts.append(f"BTC-QQQ corr={btc_qqq_corr}")

        return "\n".join(
            [
                f"{asset['symbol']} ({asset['asset_type']})",
                f"Price: {asset['price_text']} | 24h: {asset['change_24h_pct']:+.1f}% | 7d: {asset['change_7d_text']}",
                f"Consensus: {asset['consensus_bias']} | Agent: {agent_text} | Sentiment: {asset['sentiment_label']} ({asset['sentiment_score_text']})",
                f"Headline: {headline}",
                f"Event: {self._playbook_event_summary(asset)}",
                f"Signals: {ta_summary}",
                f"Levels: {self._playbook_levels_summary(ta_signal)}",
                f"Context: {' | '.join(context_parts)}",
            ]
        )

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

        verdicts = {
            verdict.symbol.upper(): verdict
            for verdict in (
                briefing_data.agent_analysis.debate_verdicts
                if briefing_data.agent_analysis is not None
                else []
            )
        }

        notable_assets: list[dict[str, object]] = []
        for price in [*briefing_data.stock_prices, *briefing_data.crypto_prices]:
            symbol_upper = price.symbol.upper()
            news_item = latest_news_by_symbol.get(symbol_upper)
            event = strongest_event_by_symbol.get(symbol_upper)
            verdict = verdicts.get(symbol_upper)
            composite = briefing_data.composite_scores.get(symbol_upper)
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
                    "consensus_bias": self._playbook_bias_label(composite.direction if composite is not None else "neutral"),
                    "agent_view": verdict.prevailing_view if verdict is not None else "N/A",
                    "agent_confidence_text": (
                        f"{verdict.confidence:.0%}"
                        if verdict is not None
                        else "N/A"
                    ),
                    "rsi_value_text": self._playbook_number_text(
                        briefing_data.ta_signals.get(symbol_upper).rsi_value
                        if symbol_upper in briefing_data.ta_signals
                        else None,
                        precision=1,
                    ),
                    "rsi_signal": (
                        briefing_data.ta_signals[symbol_upper].rsi_signal
                        if symbol_upper in briefing_data.ta_signals
                        else "N/A"
                    ),
                    "macd_signal": (
                        briefing_data.ta_signals[symbol_upper].macd_signal
                        if symbol_upper in briefing_data.ta_signals
                        else "N/A"
                    ),
                    "macd_histogram_text": self._playbook_number_text(
                        briefing_data.ta_signals.get(symbol_upper).macd_histogram
                        if symbol_upper in briefing_data.ta_signals
                        else None,
                        precision=3,
                    ),
                    "bb_position": (
                        briefing_data.ta_signals[symbol_upper].bb_position
                        if symbol_upper in briefing_data.ta_signals
                        else "N/A"
                    ),
                    "bb_squeeze_text": (
                        "(SQUEEZE!)"
                        if symbol_upper in briefing_data.ta_signals and briefing_data.ta_signals[symbol_upper].bb_squeeze
                        else ""
                    ),
                    "trend": (
                        briefing_data.ta_signals[symbol_upper].trend
                        if symbol_upper in briefing_data.ta_signals
                        else "N/A"
                    ),
                    "atr_pct_text": self._playbook_pct_text(
                        briefing_data.ta_signals.get(symbol_upper).atr_pct
                        if symbol_upper in briefing_data.ta_signals
                        else None,
                    ),
                    "volatility": (
                        briefing_data.ta_signals[symbol_upper].volatility
                        if symbol_upper in briefing_data.ta_signals
                        else "N/A"
                    ),
                    "ta_overall": (
                        briefing_data.ta_signals[symbol_upper].overall_signal
                        if symbol_upper in briefing_data.ta_signals
                        else "N/A"
                    ),
                    "ta_confidence_text": self._playbook_confidence_text(
                        briefing_data.ta_signals.get(symbol_upper).confidence
                        if symbol_upper in briefing_data.ta_signals
                        else None,
                    ),
                    "pair_label": self._pair_label(briefing_data.correlation_pairs.get(symbol_upper)),
                    "pair_corr_20d_text": self._playbook_number_text(
                        briefing_data.correlation_pairs.get(symbol_upper).corr_20d
                        if symbol_upper in briefing_data.correlation_pairs
                        else None,
                        precision=2,
                    ),
                    "pair_corr_60d_text": self._playbook_number_text(
                        briefing_data.correlation_pairs.get(symbol_upper).corr_60d
                        if symbol_upper in briefing_data.correlation_pairs
                        else None,
                        precision=2,
                    ),
                    "pair_regime": (
                        briefing_data.correlation_pairs[symbol_upper].regime_signal
                        if symbol_upper in briefing_data.correlation_pairs
                        else "N/A"
                    ),
                    "btc_qqq_corr_text": self._playbook_number_text(
                        briefing_data.regime_state.btc_qqq_corr if briefing_data.regime_state is not None else None,
                        precision=2,
                    ),
                    "regime": (
                        briefing_data.regime_state.regime.value
                        if briefing_data.regime_state is not None
                        else "N/A"
                    ),
                    "composite_score_text": self._playbook_number_text(
                        briefing_data.composite_scores.get(symbol_upper).score
                        if symbol_upper in briefing_data.composite_scores
                        else None,
                        precision=2,
                        signed=True,
                    ),
                    "composite_direction": (
                        briefing_data.composite_scores[symbol_upper].direction
                        if symbol_upper in briefing_data.composite_scores
                        else "N/A"
                    ),
                    "signal_agreement_text": self._playbook_confidence_text(
                        briefing_data.composite_scores.get(symbol_upper).agreement
                        if symbol_upper in briefing_data.composite_scores
                        else None,
                    ),
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
    def _truncate_playbook_field(value: object, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return "N/A"
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def _playbook_event_summary(asset: dict[str, object]) -> str:
        event_type = str(asset.get("event_type", "none"))
        if event_type == "none":
            return "none"

        parts = [f"{event_type} ({asset.get('event_confidence_text', 'N/A')})"]
        expected_car = str(asset.get("expected_car_text", "N/A"))
        expected_window = str(asset.get("expected_car_window", "N/A"))
        if expected_car != "N/A" and expected_window != "N/A":
            parts.append(f"hist {expected_car} over {expected_window}")
        remaining = str(asset.get("remaining_drift_text", "N/A"))
        if remaining != "N/A":
            parts.append(f"remaining {remaining}")
        return " | ".join(parts)

    @classmethod
    def _playbook_levels_summary(cls, ta_signal: TASignal | None) -> str:
        if ta_signal is None:
            return "current price only"

        parts = [
            f"support ~{cls._playbook_price_level(ta_signal.bb_lower)}",
            f"EMA20 ~{cls._playbook_price_level(ta_signal.ema_20)}",
            f"resistance ~{cls._playbook_price_level(ta_signal.bb_upper)}",
        ]
        return ", ".join(parts)

    @staticmethod
    def _playbook_price_level(value: float | None) -> str:
        if value is None:
            return "N/A"
        if value >= 1_000:
            return f"${value:,.0f}"
        if value >= 1:
            return f"${value:,.2f}"
        return f"${value:.4g}"

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
    def _playbook_number_text(
        value: float | None,
        *,
        precision: int,
        signed: bool = False,
    ) -> str:
        if value is None:
            return "N/A"
        sign = "+" if signed else ""
        return f"{value:{sign}.{precision}f}"

    @staticmethod
    def _playbook_confidence_text(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.0%}"

    @staticmethod
    def _pair_label(pair: PairCorrelation | None) -> str:
        if pair is None:
            return "N/A"
        return f"{pair.symbol_a}-{pair.symbol_b}"

    @staticmethod
    def _playbook_price_text(price: AssetPrice) -> str:
        if price.asset_type == AssetType.STOCK:
            return f"${price.current_price:,.2f}"
        if price.current_price >= 1_000:
            return f"${price.current_price:,.0f}"
        if price.current_price >= 1:
            return f"${price.current_price:,.2f}"
        return f"${price.current_price:.4g}"

    @staticmethod
    def _playbook_bias_label(direction: str) -> str:
        if direction in {"strong_buy", "buy"}:
            return "Bullish"
        if direction in {"strong_sell", "sell"}:
            return "Bearish"
        return "Neutral"

    async def _ensure_agent_pipeline(self) -> AgentPipeline | None:
        if self.settings.llm_api_key is None:
            return None
        if self._agent_pipeline is not None:
            return self._agent_pipeline

        try:
            from src.agents.graph import AgentPipeline
            llm_client = await self._ensure_llm_client()
            if llm_client is None:
                return None
            self._agent_pipeline = AgentPipeline(
                gemini_client=llm_client,
                ta_signal_fetcher=self.get_ta_signal_payload,
                lessons_provider=self.get_latest_lessons,
            )
            self._agent_pipeline_init_error = None
        except Exception as exc:
            self._agent_pipeline = None
            self._llm_client = None
            self._agent_pipeline_init_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Agent pipeline initialization failed; continuing without LLM access")

        return self._agent_pipeline

    async def _ensure_llm_client(self) -> AgentLLM | None:
        if self.settings.llm_api_key is None:
            return None
        if self._llm_client is not None:
            return self._llm_client

        from src.agents.llm import AgentLLM

        llm_client = AgentLLM()
        await llm_client.initialize()
        self._llm_client = llm_client
        return llm_client

    async def _coin_id_map(self) -> dict[str, str]:
        watchlist = await self._crypto_watchlist_items(include_hot_list=True)
        coin_map = {
            item.symbol.upper(): item.coingecko_id
            for item in watchlist
            if item.coingecko_id is not None
        }
        for alias, canonical in CRYPTO_SYMBOL_ALIASES.items():
            coin_id = coin_map.get(canonical)
            if coin_id is not None:
                coin_map.setdefault(alias, coin_id)
        return coin_map

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
