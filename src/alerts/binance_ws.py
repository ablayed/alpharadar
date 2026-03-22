from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

CryptoPriceCallback = Callable[[str, float], Awaitable[None]]

COINGECKO_TO_BINANCE: dict[str, str] = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "LINK": "linkusdt",
    "RNDR": "renderusdt",
    "RENDER": "renderusdt",
    "FET": "fetusdt",
    "TAO": "taousdt",
    "ONDO": "ondousdt",
    "AKT": "aktusdt",
    "HNT": "hntusdt",
}


def build_stream_url(symbols: list[str]) -> str:
    stream_pairs = []
    seen_pairs: set[str] = set()
    for symbol in symbols:
        pair = COINGECKO_TO_BINANCE.get(symbol.upper())
        if pair is None or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        stream_pairs.append(f"{pair}@miniTicker")
    streams = "/".join(stream_pairs)
    return f"wss://stream.binance.com:9443/stream?streams={streams}" if streams else ""


async def binance_ws_loop(
    symbols: list[str],
    on_price: CryptoPriceCallback,
) -> None:
    """
    Maintain a Binance combined miniTicker connection for crypto updates.
    """

    stream_url = build_stream_url(symbols)
    if not stream_url:
        logger.info("Binance WebSocket not started: no supported crypto symbols")
        return

    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    pair_to_symbol: dict[str, str] = {}
    for symbol in symbols:
        pair = COINGECKO_TO_BINANCE.get(symbol.upper())
        if pair is None:
            continue
        if pair.upper() not in pair_to_symbol:
            pair_to_symbol[pair.upper()] = symbol.upper()

    reconnect_interval_seconds = 23 * 3600

    while True:
        connected_at = time.time()
        try:
            async for websocket in connect(
                stream_url,
                ping_interval=20,
                ping_timeout=30,
                max_size=2**16,
                compression=None,
            ):
                try:
                    logger.info(
                        "Binance WebSocket connected for %d crypto symbols",
                        len(pair_to_symbol),
                    )
                    async for message in websocket:
                        if (time.time() - connected_at) > reconnect_interval_seconds:
                            logger.info("Binance reconnect interval reached, reconnecting")
                            await websocket.close()
                            break

                        payload = json.loads(message)
                        inner = payload.get("data", payload)
                        pair = inner.get("s")
                        last_price = inner.get("c")
                        if not isinstance(pair, str) or not isinstance(last_price, str | int | float):
                            continue

                        mapped_symbol = pair_to_symbol.get(pair.upper())
                        if mapped_symbol is None:
                            continue

                        await on_price(mapped_symbol, float(last_price))

                except asyncio.CancelledError:
                    raise
                except ConnectionClosed:
                    logger.warning("Binance WebSocket disconnected, reconnecting")
                    continue
                except Exception:
                    logger.exception("Binance WebSocket error")
                    continue

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Binance WebSocket outer loop failed, retrying in 10s")
            await asyncio.sleep(10)
