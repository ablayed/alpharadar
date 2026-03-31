from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
import numpy as np

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
PREMIUM_INDEX_URL = f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex"
FUNDING_HISTORY_URL = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
OPEN_INTEREST_URL = f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
OI_HISTORY_URL = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"

FUTURES_PAIRS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "LINK": "LINKUSDT",
    "FET": "FETUSDT",
    "TAO": "TAOUSDT",
    "RNDR": "RENDERUSDT",
    "RENDER": "RENDERUSDT",
    "AKT": "AKTUSDT",
    "ONDO": "ONDOUSDT",
    "HNT": "HNTUSDT",
}


@dataclass(slots=True)
class FundingData:
    symbol: str
    current_rate: float
    rate_annualized: float
    z_score: float
    signal: str
    description: str


@dataclass(slots=True)
class OpenInterestData:
    symbol: str
    current_oi: float
    oi_change_4h_pct: float
    oi_change_24h_pct: float
    signal: str
    description: str


@dataclass(slots=True)
class CMEGap:
    gap_low: float
    gap_high: float
    gap_direction: str
    distance_pct: float
    filled: bool
    description: str


async def fetch_funding_rate(client: httpx.AsyncClient, symbol: str) -> FundingData | None:
    symbol_upper = symbol.upper()
    pair = FUTURES_PAIRS.get(symbol_upper)
    if pair is None:
        return None

    try:
        response = await client.get(PREMIUM_INDEX_URL, params={"symbol": pair}, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        current_rate = float(payload.get("lastFundingRate", 0.0) or 0.0)
    except Exception:
        logger.exception("Failed to fetch funding rate for %s", symbol_upper)
        return None

    z_score = 0.0
    try:
        response = await client.get(
            FUNDING_HISTORY_URL,
            params={"symbol": pair, "limit": 90},
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            rates = np.asarray(
                [
                    float(item.get("fundingRate", 0.0) or 0.0)
                    for item in payload
                    if isinstance(item, dict)
                ],
                dtype=np.float64,
            )
            rates = rates[np.isfinite(rates)]
            if rates.size > 10:
                std = float(np.std(rates))
                if std > 1e-10:
                    z_score = (current_rate - float(np.mean(rates))) / std
    except Exception:
        logger.warning("Funding history unavailable for %s", symbol_upper, exc_info=True)

    return _classify_funding(symbol_upper, current_rate, z_score)


def _classify_funding(symbol: str, rate: float, z_score: float) -> FundingData:
    annualized = rate * 3 * 365 * 100
    rate_pct = rate * 100

    if rate > 0.001:
        signal = "bearish"
        description = f"Funding at {rate_pct:+.3f}% - market heavily long, correction risk high"
    elif rate > 0.0005:
        signal = "bearish"
        description = f"Funding at {rate_pct:+.3f}% - long positioning stretched, squeeze risk elevated"
    elif rate < -0.001:
        signal = "bullish"
        description = f"Funding at {rate_pct:+.3f}% - extreme short crowding, squeeze risk high"
    elif rate < -0.0005:
        signal = "bullish"
        description = f"Funding at {rate_pct:+.3f}% - shorts paying longs, bullish pressure building"
    else:
        signal = "neutral"
        description = f"Funding at {rate_pct:+.3f}% - positioning looks balanced"

    if abs(z_score) > 2.0:
        description += f" (z-score {z_score:+.1f}, extreme versus recent history)"

    return FundingData(
        symbol=symbol,
        current_rate=rate,
        rate_annualized=annualized,
        z_score=z_score,
        signal=signal,
        description=description,
    )


async def fetch_open_interest(client: httpx.AsyncClient, symbol: str) -> OpenInterestData | None:
    symbol_upper = symbol.upper()
    pair = FUTURES_PAIRS.get(symbol_upper)
    if pair is None:
        return None

    try:
        response = await client.get(OPEN_INTEREST_URL, params={"symbol": pair}, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        current_oi = float(payload.get("openInterest", 0.0) or 0.0)
    except Exception:
        logger.exception("Failed to fetch open interest for %s", symbol_upper)
        return None

    change_4h = 0.0
    change_24h = 0.0
    try:
        response = await client.get(
            OI_HISTORY_URL,
            params={"symbol": pair, "period": "4h", "limit": 7},
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            values = [
                float(item.get("sumOpenInterest", 0.0) or 0.0)
                for item in payload
                if isinstance(item, dict)
            ]
            if values:
                oi_4h_ago = values[-2] if len(values) >= 2 else current_oi
                oi_24h_ago = values[0]
                if oi_4h_ago > 0:
                    change_4h = ((current_oi - oi_4h_ago) / oi_4h_ago) * 100
                if oi_24h_ago > 0:
                    change_24h = ((current_oi - oi_24h_ago) / oi_24h_ago) * 100
    except Exception:
        logger.warning("Open interest history unavailable for %s", symbol_upper, exc_info=True)

    return _classify_open_interest(symbol_upper, current_oi, change_4h, change_24h)


def _classify_open_interest(
    symbol: str,
    current_oi: float,
    oi_change_4h_pct: float,
    oi_change_24h_pct: float,
) -> OpenInterestData:
    if abs(oi_change_4h_pct) > 10:
        signal = "warning"
        description = f"OI changed {oi_change_4h_pct:+.1f}% in 4h - liquidation cascade risk elevated"
    elif oi_change_24h_pct > 5:
        signal = "rising"
        description = f"OI up {oi_change_24h_pct:+.1f}% in 24h - fresh positions entering"
    elif oi_change_24h_pct < -5:
        signal = "falling"
        description = f"OI down {oi_change_24h_pct:+.1f}% in 24h - positions closing out"
    else:
        signal = "neutral"
        description = f"OI stable ({oi_change_24h_pct:+.1f}% in 24h)"

    return OpenInterestData(
        symbol=symbol,
        current_oi=current_oi,
        oi_change_4h_pct=round(oi_change_4h_pct, 1),
        oi_change_24h_pct=round(oi_change_24h_pct, 1),
        signal=signal,
        description=description,
    )


async def detect_cme_gaps(current_btc_price: float) -> CMEGap | None:
    if current_btc_price <= 0:
        return None
    return await asyncio.to_thread(_detect_cme_gaps_sync, current_btc_price)


def _detect_cme_gaps_sync(current_btc_price: float) -> CMEGap | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance is not installed; CME gap detection unavailable")
        return None

    try:
        history = yf.Ticker("BTC=F").history(period="45d", interval="1d", auto_adjust=False)
    except Exception:
        logger.exception("Failed to fetch CME futures history")
        return None

    return _detect_cme_gaps_from_history(history, current_btc_price)


def _detect_cme_gaps_from_history(history: object, current_btc_price: float) -> CMEGap | None:
    if current_btc_price <= 0 or not hasattr(history, "empty") or getattr(history, "empty"):
        return None
    if not hasattr(history, "sort_index") or len(history) < 5:  # type: ignore[arg-type]
        return None

    sorted_history = history.sort_index()
    gaps: list[CMEGap] = []

    for idx in range(1, len(sorted_history)):  # type: ignore[arg-type]
        prev_row = sorted_history.iloc[idx - 1]
        curr_row = sorted_history.iloc[idx]
        prev_close = float(prev_row["Close"])
        curr_open = float(curr_row["Open"])
        if not np.isfinite(prev_close) or not np.isfinite(curr_open) or prev_close <= 0:
            continue

        prev_date = sorted_history.index[idx - 1]
        curr_date = sorted_history.index[idx]
        if (curr_date - prev_date).days <= 1:
            continue

        gap_pct = (abs(curr_open - prev_close) / prev_close) * 100
        if gap_pct <= 0.5:
            continue

        gap_low = min(prev_close, curr_open)
        gap_high = max(prev_close, curr_open)
        gap_direction = "up" if curr_open > prev_close else "down"

        filled = False
        for fill_idx in range(idx, len(sorted_history)):  # type: ignore[arg-type]
            fill_row = sorted_history.iloc[fill_idx]
            if gap_direction == "up" and float(fill_row["Low"]) <= gap_low:
                filled = True
                break
            if gap_direction == "down" and float(fill_row["High"]) >= gap_high:
                filled = True
                break

        if filled:
            continue

        gap_mid = (gap_low + gap_high) / 2
        distance_pct = (abs(current_btc_price - gap_mid) / current_btc_price) * 100
        gaps.append(
            CMEGap(
                gap_low=gap_low,
                gap_high=gap_high,
                gap_direction=gap_direction,
                distance_pct=round(distance_pct, 1),
                filled=False,
                description=(
                    f"Unfilled CME gap at ${gap_low:,.0f}-${gap_high:,.0f} "
                    f"({distance_pct:.1f}% away) - historically acts as a price magnet"
                ),
            )
        )

    if not gaps:
        return None

    gaps.sort(key=lambda gap: gap.distance_pct)
    return gaps[0]
