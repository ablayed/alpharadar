"""
Rule-based mover selection for the debate stage.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping


async def mover_selector_node(state: Mapping[str, object]) -> dict[str, object]:
    """Select up to two assets for Bull/Bear debate without using an LLM."""
    stocks = _dict_list(state.get("stock_prices"))
    crypto = _dict_list(state.get("crypto_prices"))
    filings = _dict_list(state.get("sec_filings"))
    news_items = _dict_list(state.get("news_items"))
    is_weekend = bool(state.get("is_weekend", False))
    forced_symbols = {symbol.upper() for symbol in _string_list(state.get("forced_symbols"))}

    assets: list[dict[str, object]] = []
    symbol_sentiment = _aggregate_symbol_sentiment(news_items)
    filing_symbols = {
        str(filing.get("ticker", "")).upper()
        for filing in filings
        if isinstance(filing.get("ticker"), str)
    }

    if not is_weekend or forced_symbols:
        for stock in stocks:
            assets.append(
                _build_candidate(
                    stock,
                    asset_type="stock",
                    has_filing=str(stock.get("symbol", "")).upper() in filing_symbols,
                    avg_sentiment=symbol_sentiment.get(str(stock.get("symbol", "")).upper()),
                )
            )

    for coin in crypto:
        assets.append(
            _build_candidate(
                coin,
                asset_type="crypto",
                has_filing=False,
                avg_sentiment=symbol_sentiment.get(str(coin.get("symbol", "")).upper()),
            )
        )

    filtered_assets = [asset for asset in assets if asset.get("symbol")]
    if forced_symbols:
        forced_selected = [
            {
                "symbol": str(asset["symbol"]),
                "asset_type": str(asset["asset_type"]),
                "reason": "Requested via /analyze",
            }
            for asset in filtered_assets
            if str(asset["symbol"]).upper() in forced_symbols
        ]
        return {"selected_movers": forced_selected[:2]}

    filtered_assets.sort(
        key=lambda asset: (
            _float_or_zero(asset.get("abs_change")),
            bool(asset.get("has_filing", False)),
            bool(asset.get("sentiment_divergence", False)),
        ),
        reverse=True,
    )

    selected_movers: list[dict[str, object]] = []
    for asset in filtered_assets[:2]:
        abs_change = _float_or_zero(asset.get("abs_change"))
        if abs_change < 1.0 and not bool(asset.get("has_filing", False)):
            continue
        selected_movers.append(
            {
                "symbol": str(asset["symbol"]),
                "asset_type": str(asset["asset_type"]),
                "reason": _selection_reason(asset),
            }
        )

    if not selected_movers and filtered_assets:
        top = filtered_assets[0]
        selected_movers.append(
            {
                "symbol": str(top["symbol"]),
                "asset_type": str(top["asset_type"]),
                "reason": f"Largest mover ({_float_or_zero(top.get('raw_change')):+.1f}%)",
            }
        )

    return {"selected_movers": selected_movers}


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _build_candidate(
    asset: dict[str, object],
    *,
    asset_type: str,
    has_filing: bool,
    avg_sentiment: float | None,
) -> dict[str, object]:
    symbol = str(asset.get("symbol", "")).upper()
    raw_change = _float_or_zero(asset.get("change_24h_pct"))
    divergence = avg_sentiment is not None and raw_change != 0 and (
        (avg_sentiment > 0.35 and raw_change < 0)
        or (avg_sentiment < -0.35 and raw_change > 0)
    )
    return {
        "symbol": symbol,
        "asset_type": asset_type,
        "abs_change": abs(raw_change),
        "raw_change": raw_change,
        "has_filing": has_filing,
        "sentiment_divergence": divergence,
        "avg_sentiment": avg_sentiment,
    }


def _aggregate_symbol_sentiment(news_items: list[dict[str, object]]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for item in news_items:
        score = item.get("sentiment_score")
        if not isinstance(score, int | float):
            continue
        for symbol in _string_list(item.get("related_symbols")):
            scores[symbol.upper()].append(float(score))
    return {
        symbol: sum(values) / len(values)
        for symbol, values in scores.items()
        if values
    }


def _selection_reason(asset: Mapping[str, object]) -> str:
    if bool(asset.get("has_filing", False)):
        return "SEC 8-K filing"
    if bool(asset.get("sentiment_divergence", False)):
        return "Strong sentiment divergence"
    return f"{_float_or_zero(asset.get('raw_change')):+.1f}% move"


def _float_or_zero(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
