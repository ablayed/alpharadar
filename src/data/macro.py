from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FearGreedData:
    value: int
    classification: str
    signal: str
    description: str


@dataclass(slots=True)
class DXYData:
    current: float
    change_7d_pct: float
    btc_correlation_30d: float
    signal: str
    description: str


@dataclass(slots=True)
class BTCDominanceData:
    dominance_pct: float
    change_7d: float
    season: str
    description: str


async def fetch_fear_greed(client: httpx.AsyncClient) -> FearGreedData | None:
    try:
        response = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        records = payload.get("data", [])
        if not isinstance(records, list) or not records:
            return None
        entry = records[0] if isinstance(records[0], dict) else {}
        value = int(entry.get("value", 50))
        classification = str(entry.get("value_classification", "Neutral"))
        return _classify_fear_greed(value, classification)
    except Exception:
        logger.exception("Failed to fetch Fear & Greed index")
        return None


def _classify_fear_greed(value: int, classification: str) -> FearGreedData:
    if value <= 10:
        signal = "bullish"
        description = f"Extreme Fear ({value}) - historically a strong contrarian buy zone"
    elif value <= 25:
        signal = "bullish"
        description = f"Fear ({value}) - contrarian setup favors patience on the long side"
    elif value >= 90:
        signal = "bearish"
        description = f"Extreme Greed ({value}) - froth is high and correction risk elevated"
    elif value >= 75:
        signal = "bearish"
        description = f"Greed ({value}) - market looks overconfident, pullback risk rising"
    else:
        signal = "neutral"
        description = f"Neutral sentiment ({value}) - no strong contrarian edge"

    return FearGreedData(
        value=value,
        classification=classification,
        signal=signal,
        description=description,
    )


async def fetch_dxy_data() -> DXYData | None:
    return await asyncio.to_thread(_fetch_dxy_data_sync)


def _fetch_dxy_data_sync() -> DXYData | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance is not installed; DXY data unavailable")
        return None

    try:
        dxy_history = yf.Ticker("DX-Y.NYB").history(period="60d", interval="1d", auto_adjust=False)
        btc_history = yf.Ticker("BTC-USD").history(period="60d", interval="1d", auto_adjust=False)
    except Exception:
        logger.exception("Failed to fetch DXY history")
        return None

    if dxy_history.empty or "Close" not in dxy_history or len(dxy_history) < 10:
        return None

    close = dxy_history["Close"].dropna()
    if close.empty:
        return None
    current = float(close.iloc[-1])
    week_ago = float(close.iloc[-5]) if len(close) >= 5 else current
    change_7d_pct = ((current - week_ago) / week_ago) * 100 if week_ago > 0 else 0.0

    correlation = 0.0
    if not btc_history.empty and "Close" in btc_history:
        aligned = (
            dxy_history[["Close"]]
            .rename(columns={"Close": "dxy"})
            .join(
                btc_history[["Close"]].rename(columns={"Close": "btc"}),
                how="inner",
            )
            .dropna()
        )
        if len(aligned) >= 31:
            returns = np.log(aligned[["dxy", "btc"]]).diff().dropna().tail(30)
            if len(returns) >= 10:
                correlation = float(np.corrcoef(returns["dxy"], returns["btc"])[0, 1])

    return _classify_dxy(current, change_7d_pct, correlation)


def _classify_dxy(current: float, change_7d_pct: float, correlation: float) -> DXYData:
    if current > 104:
        signal = "bearish"
        description = f"DXY at {current:.1f} - strong dollar remains a crypto headwind"
    elif current < 100:
        signal = "bullish"
        description = f"DXY at {current:.1f} - weak dollar is a tailwind for crypto"
    else:
        signal = "neutral"
        description = f"DXY at {current:.1f} - dollar backdrop is neutral"

    if change_7d_pct <= -1.0:
        signal = "bullish"
        description += f". Dollar down {change_7d_pct:.1f}% this week - supportive for crypto"
    elif change_7d_pct >= 1.0:
        signal = "bearish"
        description += f". Dollar up {change_7d_pct:+.1f}% this week - a fresh headwind"

    return DXYData(
        current=round(current, 2),
        change_7d_pct=round(change_7d_pct, 2),
        btc_correlation_30d=round(correlation, 2),
        signal=signal,
        description=description,
    )


async def fetch_btc_dominance(client: httpx.AsyncClient) -> BTCDominanceData | None:
    try:
        response = await client.get("https://api.coingecko.com/api/v3/global", timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        data = payload.get("data", {})
        if not isinstance(data, dict):
            return None
        market_cap_percentage = data.get("market_cap_percentage", {})
        if not isinstance(market_cap_percentage, dict):
            return None
        dominance_pct = float(market_cap_percentage.get("btc", 0.0) or 0.0)
        mcap_change = float(data.get("market_cap_change_percentage_24h_usd", 0.0) or 0.0)
        return _classify_btc_dominance(dominance_pct, mcap_change)
    except Exception:
        logger.exception("Failed to fetch BTC dominance")
        return None


def _classify_btc_dominance(dominance_pct: float, market_cap_change_24h_pct: float) -> BTCDominanceData:
    if dominance_pct > 55:
        season = "btc_season"
        description = f"BTC dominance at {dominance_pct:.1f}% - Bitcoin is still leading the tape"
    elif dominance_pct < 45 and market_cap_change_24h_pct > 0:
        season = "alt_season"
        description = f"BTC dominance at {dominance_pct:.1f}% with market cap rising - alt rotation improving"
    else:
        season = "neutral"
        description = f"BTC dominance at {dominance_pct:.1f}% - no clear rotation edge"

    return BTCDominanceData(
        dominance_pct=round(dominance_pct, 1),
        change_7d=0.0,
        season=season,
        description=description,
    )
