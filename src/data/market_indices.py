from __future__ import annotations

from dataclasses import replace

from src.data.base import AssetPrice
from src.data.finnhub_client import FinnhubClient

INDEX_SYMBOLS: dict[str, str] = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
}

FALLBACK_INDEX_SYMBOLS: dict[str, str] = {
    "S&P 500": "SPY",
    "NASDAQ": "QQQ",
    "VIX": "VIXY",
}


class MarketIndicesFetcher:
    def __init__(self) -> None:
        self._resolved_symbols: dict[str, str] = dict(INDEX_SYMBOLS)
        self._validated = False

    async def get_index_prices(self, finnhub_client: FinnhubClient) -> list[AssetPrice]:
        if not self._validated:
            await self._validate_symbols(finnhub_client)

        prices: list[AssetPrice] = []
        for index_name, symbol in self._resolved_symbols.items():
            quote = await finnhub_client.get_quote(symbol)
            prices.append(replace(quote, symbol=index_name))
        return prices

    async def _validate_symbols(self, finnhub_client: FinnhubClient) -> None:
        for index_name, symbol in INDEX_SYMBOLS.items():
            quote = await finnhub_client.get_quote(symbol)
            if quote.current_price != 0:
                continue
            fallback_symbol = FALLBACK_INDEX_SYMBOLS.get(index_name)
            if fallback_symbol is None:
                continue
            fallback_quote = await finnhub_client.get_quote(fallback_symbol)
            if fallback_quote.current_price > 0:
                self._resolved_symbols[index_name] = fallback_symbol
        self._validated = True

