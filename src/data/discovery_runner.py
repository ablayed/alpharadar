"""
Scheduled discovery pipeline for dynamic crypto scanning.
"""
from __future__ import annotations

import logging

from src.data.discovery import (
    apply_safety_filters,
    build_trigger_reason,
    enrich_candidates,
    fetch_binance_volume_spikes,
    fetch_coingecko_top_gainers,
    fetch_coingecko_trending,
    merge_candidate_sources,
    score_candidate,
)
from src.data.hot_list import (
    add_to_hot_list,
    expire_old_entries,
    get_active_hot_list,
    record_discovery_run,
)

logger = logging.getLogger(__name__)


async def run_discovery(
    db_path: str,
    coingecko_api_key: str,
    fixed_watchlist: set[str],
    bot=None,
    chat_id: int | str | None = None,
) -> list[dict]:
    """Run the full discovery flow and return newly added hot-list coins."""
    logger.info("Discovery: starting scan")

    expired_count = await expire_old_entries(db_path)

    trending, top_gainers, binance_volume = await _fetch_discovery_sources(
        api_key=coingecko_api_key,
        db_path=db_path,
    )
    merged_candidates = merge_candidate_sources(trending, binance_volume, top_gainers)

    fixed_symbols = {symbol.upper() for symbol in fixed_watchlist}
    prelim_candidates = [
        candidate
        for candidate in merged_candidates
        if str(candidate.get("symbol", "")).upper() not in fixed_symbols
        and float(candidate.get("volume_24h", 0) or 0) >= 500_000
    ]
    prelim_candidates.sort(key=lambda item: _preliminary_rank(item), reverse=True)
    enriched_candidates = await enrich_candidates(prelim_candidates[:25], coingecko_api_key, db_path=db_path)
    safe_candidates = apply_safety_filters(enriched_candidates, fixed_watchlist)

    for coin in safe_candidates:
        coin["signal_score"] = score_candidate(coin)
        coin["trigger_reason"] = build_trigger_reason(coin)

    safe_candidates.sort(
        key=lambda item: (
            int(item.get("signal_score", 0) or 0),
            float(item.get("volume_24h", 0) or 0),
            abs(float(item.get("price_change_24h", 0) or 0)),
        ),
        reverse=True,
    )

    newly_added: list[dict] = []
    for coin in safe_candidates[:20]:
        added = await add_to_hot_list(
            db_path=db_path,
            symbol=str(coin.get("symbol", "")).upper(),
            coingecko_id=str(coin.get("id", "") or ""),
            name=str(coin.get("name", "") or coin.get("symbol", "")),
            price=float(coin.get("price_usd", 0) or 0),
            volume=float(coin.get("volume_24h", 0) or 0),
            market_cap=float(coin.get("market_cap", 0) or 0),
            trigger_reason=str(coin.get("trigger_reason", coin.get("source", "unknown"))),
            signal_score=int(coin.get("signal_score", 0) or 0),
            price_change_24h=float(coin.get("price_change_24h", 0) or 0),
            binance_symbol=(
                str(coin.get("binance_symbol"))
                if coin.get("binance_symbol") is not None
                else None
            ),
        )
        if added:
            newly_added.append(coin)

    await record_discovery_run(
        db_path,
        candidates_scanned=len(merged_candidates),
        candidates_safe=len(safe_candidates),
        added_count=len(newly_added),
        expired_count=expired_count,
    )

    if newly_added and bot is not None and chat_id is not None:
        await _send_discovery_alert(bot, chat_id, db_path, newly_added)

    logger.info(
        "Discovery: complete scanned=%d safe=%d added=%d expired=%d",
        len(merged_candidates),
        len(safe_candidates),
        len(newly_added),
        expired_count,
    )
    return newly_added


async def _fetch_discovery_sources(*, api_key: str, db_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    trending = await fetch_coingecko_trending(api_key, db_path=db_path)
    top_gainers = await fetch_coingecko_top_gainers(api_key, limit=120, db_path=db_path)
    binance_volume = await fetch_binance_volume_spikes()
    return trending, top_gainers, binance_volume


def _preliminary_rank(candidate: dict) -> float:
    rough_score = 0.0
    rough_score += abs(float(candidate.get("price_change_24h", 0) or 0))
    rough_score += min(float(candidate.get("volume_24h", 0) or 0) / 5_000_000, 20.0)
    sources = {
        str(source)
        for source in candidate.get("sources", [candidate.get("source", "unknown")])
        if source
    }
    if "trending_coingecko" in sources:
        rough_score += 10
    if "binance_volume" in sources:
        rough_score += 8
    if len(sources) > 1:
        rough_score += 5
    return rough_score


async def _send_discovery_alert(bot, chat_id: int | str, db_path: str, newly_added: list[dict]) -> None:
    lines = ["🔥 <b>NEW TRENDING COINS DETECTED</b>", ""]
    for coin in newly_added:
        pct_change = float(coin.get("price_change_24h", 0) or 0)
        direction = "🟢" if pct_change >= 0 else "🔴"
        lines.append(f"{direction} <b>{coin['symbol']}</b> ({coin.get('name', '')})")
        lines.append(
            f"   {pct_change:+.1f}% | Vol: ${float(coin.get('volume_24h', 0) or 0):,.0f} | "
            f"Score: {int(coin.get('signal_score', 0) or 0)}/100 | "
            f"{coin.get('trigger_reason', coin.get('source', 'unknown'))}"
        )

    active_count = len(await get_active_hot_list(db_path))
    lines.extend(
        [
            "",
            "Added to the hot list. Use /trade <SYMBOL> for analysis.",
            f"Active hot list: {active_count} coin(s)",
        ]
    )

    try:
        await bot.send_message(
            chat_id=chat_id,
            text="\n".join(lines),
            parse_mode="HTML",
        )
    except Exception:
        logger.exception("Discovery: failed to send Telegram alert")
