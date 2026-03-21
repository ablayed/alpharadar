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

    if is_weekend:
        selected_movers: list[dict[str, object]] = []
        stock_candidates = sorted(
            [asset for asset in filtered_assets if asset.get("asset_type") == "stock"],
            key=_weekend_sort_key,
            reverse=True,
        )
        crypto_candidates = sorted(
            [asset for asset in filtered_assets if asset.get("asset_type") == "crypto"],
            key=_weekend_sort_key,
            reverse=True,
        )
        if stock_candidates:
            top_stock = stock_candidates[0]
            selected_movers.append(
                {
                    "symbol": str(top_stock["symbol"]),
                    "asset_type": str(top_stock["asset_type"]),
                    "reason": _weekend_selection_reason(top_stock),
                }
            )
        if crypto_candidates:
            top_crypto = crypto_candidates[0]
            selected_movers.append(
                {
                    "symbol": str(top_crypto["symbol"]),
                    "asset_type": str(top_crypto["asset_type"]),
                    "reason": _weekend_selection_reason(top_crypto),
                }
            )
        if selected_movers:
            return {"selected_movers": selected_movers[:2]}

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
    weekly_change = asset.get("change_7d_pct")
    weekly_numeric = _float_or_zero(weekly_change) if isinstance(weekly_change, int | float) else None
    divergence = avg_sentiment is not None and raw_change != 0 and (
        (avg_sentiment > 0.35 and raw_change < 0)
        or (avg_sentiment < -0.35 and raw_change > 0)
    )
    return {
        "symbol": symbol,
        "asset_type": asset_type,
        "abs_change": abs(raw_change),
        "raw_change": raw_change,
        "weekly_change": weekly_numeric,
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


def _weekend_sort_key(asset: Mapping[str, object]) -> tuple[float, bool, bool]:
    weekly_change = asset.get("weekly_change")
    selection_change = (
        _float_or_zero(weekly_change)
        if isinstance(weekly_change, int | float)
        else _float_or_zero(asset.get("raw_change"))
    )
    return (
        abs(selection_change),
        bool(asset.get("has_filing", False)),
        bool(asset.get("sentiment_divergence", False)),
    )


def _weekend_selection_reason(asset: Mapping[str, object]) -> str:
    weekly_change = asset.get("weekly_change")
    move = (
        _float_or_zero(weekly_change)
        if isinstance(weekly_change, int | float)
        else _float_or_zero(asset.get("raw_change"))
    )
    asset_label = "stock" if str(asset.get("asset_type")) == "stock" else "crypto"
    return f"Largest weekly {asset_label} move ({move:+.1f}%)"


def _float_or_zero(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
