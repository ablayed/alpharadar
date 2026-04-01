"""
Weekly training pipeline for the lightweight crypto ensemble.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from src.ml.ensemble import MLEnsemble
from src.ml.features import engineer_features
from src.ta.candles import fetch_crypto_candles

logger = logging.getLogger(__name__)

YFINANCE_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "LINK": "LINK-USD",
    "FET": "FET-USD",
    "TAO": "TAO-USD",
    "RNDR": "RNDR-USD",
    "AKT": "AKT-USD",
    "ONDO": "ONDO-USD",
    "HNT": "HNT-USD",
}

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "LINK": "chainlink",
    "FET": "fetch-ai",
    "TAO": "bittensor",
    "RNDR": "render-token",
    "AKT": "akash-network",
    "ONDO": "ondo-finance",
    "HNT": "helium",
}


async def train_all_models(
    ensemble: MLEnsemble,
    crypto_symbols: list[str],
    *,
    db_path: str | None = None,
) -> dict[str, bool]:
    """Fetch daily OHLCV data and retrain the ensemble for each crypto symbol."""
    results: dict[str, bool] = {}

    for raw_symbol in crypto_symbols:
        symbol = raw_symbol.strip().upper()
        try:
            candles = await _fetch_ohlcv(symbol, days=120, db_path=db_path)
            if candles[0] is None:
                logger.warning("Skipping ML training for %s due to missing candles", symbol)
                results[symbol] = False
                continue

            close, high, low, volume, open_ = candles
            if close is None or high is None or low is None or volume is None or open_ is None:
                logger.warning("Skipping ML training for %s due to incomplete OHLCV arrays", symbol)
                results[symbol] = False
                continue

            X, feature_names = engineer_features(close, high, low, volume, open_)
            y = np.full(len(close), np.nan, dtype=np.float64)
            y[:-1] = (close[1:] > close[:-1]).astype(np.float64)

            success = ensemble.train(symbol, X, y, feature_names=feature_names)
            results[symbol] = success
            if success:
                logger.info(
                    "ML training succeeded for %s: %d candles, %d features",
                    symbol,
                    len(close),
                    X.shape[1],
                )
        except Exception:
            logger.exception("ML training failed for %s", symbol)
            results[symbol] = False

    return results


async def _fetch_ohlcv(
    symbol: str,
    *,
    days: int = 120,
    db_path: str | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    candles = await _fetch_yfinance_ohlcv(symbol, days)
    if candles[0] is not None:
        return candles

    coin_id = COINGECKO_IDS.get(symbol.upper())
    if coin_id is None:
        return None, None, None, None, None

    fallback = await fetch_crypto_candles(coin_id, days=days, db_path=db_path)
    if fallback is None:
        return None, None, None, None, None
    return fallback


async def _fetch_yfinance_ohlcv(
    symbol: str,
    days: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    ticker = YFINANCE_TICKERS.get(symbol.upper(), f"{symbol.upper()}-USD")
    try:
        return await asyncio.to_thread(_download_yfinance_ohlcv, ticker, days)
    except Exception:
        logger.exception("Failed to download yfinance OHLCV for %s", symbol.upper())
        return None, None, None, None, None


def _download_yfinance_ohlcv(
    ticker: str,
    days: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance is unavailable for ML OHLCV download")
        return None, None, None, None, None

    history = yf.Ticker(ticker).history(period=f"{days}d", interval="1d", auto_adjust=False)
    required_columns = ("Open", "High", "Low", "Close", "Volume")
    if history.empty or any(column not in history for column in required_columns):
        return None, None, None, None, None

    open_array = np.asarray(history["Open"], dtype=np.float64)
    high_array = np.asarray(history["High"], dtype=np.float64)
    low_array = np.asarray(history["Low"], dtype=np.float64)
    close_array = np.asarray(history["Close"], dtype=np.float64)
    volume_array = np.asarray(history["Volume"], dtype=np.float64)

    valid_mask = (
        np.isfinite(open_array)
        & np.isfinite(high_array)
        & np.isfinite(low_array)
        & np.isfinite(close_array)
        & np.isfinite(volume_array)
    )
    open_array = open_array[valid_mask]
    high_array = high_array[valid_mask]
    low_array = low_array[valid_mask]
    close_array = close_array[valid_mask]
    volume_array = volume_array[valid_mask]

    if len(close_array) < 60:
        return None, None, None, None, None

    return close_array, high_array, low_array, volume_array, open_array
