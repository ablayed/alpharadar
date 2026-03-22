from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
StockTradeCallback = Callable[[str, float, int], Awaitable[None]]


def _is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


async def finnhub_ws_loop(
    api_key: str,
    symbols: list[str],
    on_trade: StockTradeCallback,
) -> None:
    """
    Maintain a single Finnhub WebSocket connection for stock trade updates.
    """

    if not api_key or not symbols:
        logger.info("Finnhub WebSocket not started: missing API key or symbols")
        return

    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    uri = f"wss://ws.finnhub.io?token={api_key}"
    subscribed_symbols = [symbol.upper() for symbol in symbols]

    while True:
        if not _is_market_hours():
            await asyncio.sleep(60)
            continue

        try:
            async for websocket in connect(
                uri,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**16,
                compression=None,
            ):
                pause_until_market_open = False
                try:
                    logger.info(
                        "Finnhub WebSocket connected, subscribing to %d stock symbols",
                        len(subscribed_symbols),
                    )
                    for symbol in subscribed_symbols:
                        await websocket.send(json.dumps({"type": "subscribe", "symbol": symbol}))

                    async for message in websocket:
                        if not _is_market_hours():
                            pause_until_market_open = True
                            await websocket.close()
                            break

                        payload = json.loads(message)
                        if payload.get("type") != "trade":
                            continue

                        for trade in payload.get("data", []):
                            trade_symbol = str(trade.get("s", "")).upper()
                            trade_price = trade.get("p")
                            trade_volume = trade.get("v", 0)
                            if not trade_symbol or not isinstance(trade_price, int | float):
                                continue
                            await on_trade(trade_symbol, float(trade_price), int(trade_volume))

                except asyncio.CancelledError:
                    raise
                except ConnectionClosed:
                    logger.warning("Finnhub WebSocket disconnected, reconnecting")
                    continue
                except Exception:
                    logger.exception("Finnhub WebSocket error")
                    continue

                if pause_until_market_open:
                    break

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Finnhub WebSocket outer loop failed, retrying in 30s")
            await asyncio.sleep(30)
