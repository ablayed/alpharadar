"""
Scheduled discovery pipeline for dynamic crypto scanning.
"""
from __future__ import annotations

import logging

from src.bot.handlers import save_signal_to_log
from src.data.discovery import (
    apply_safety_filters,
    build_trigger_reason,
    enrich_and_score,
    enrich_candidates,
    fetch_binance_volume_spikes,
    fetch_coingecko_top_gainers,
    fetch_coingecko_trending,
    merge_candidate_sources,
)
from src.data.hot_list import (
    add_to_hot_list,
    expire_old_entries,
    get_active_hot_list,
    record_discovery_run,
)
from src.ta.candles import fetch_crypto_candles
from src.ta.signals import compute_signals
from src.trade.generator import generate_crypto_trade_signal

logger = logging.getLogger(__name__)


async def run_discovery(
    db_path: str,
    coingecko_api_key: str,
    fixed_watchlist: set[str],
    bot=None,
    chat_id: int | str | None = None,
    ensemble=None,
    tracker=None,
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
    prelim_candidates.sort(key=_preliminary_rank, reverse=True)
    enriched_candidates = await enrich_candidates(prelim_candidates[:25], coingecko_api_key, db_path=db_path)
    safe_candidates = apply_safety_filters(enriched_candidates, fixed_watchlist)
    scored_candidates = await enrich_and_score(safe_candidates, coingecko_api_key)

    for coin in scored_candidates:
        coin["trigger_reason"] = build_trigger_reason(coin)

    newly_added: list[dict] = []
    for coin in scored_candidates[:20]:
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
            direction=str(coin.get("direction", "neutral") or "neutral"),
            breakout_description=str(coin.get("breakout_description", "") or ""),
            signals_firing=int(coin.get("signals_firing", 0) or 0),
            key_level=_optional_float(coin.get("key_level")),
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

    if newly_added:
        if bot is not None and chat_id is not None:
            await _send_discovery_alert(bot, chat_id, db_path, newly_added)
        for coin in newly_added:
            if (
                int(coin.get("signal_score", 0) or 0) >= 70
                and int(coin.get("signals_firing", 0) or 0) >= 3
                and str(coin.get("direction", "neutral")) in {"long_setup", "short_setup"}
            ):
                await _run_instant_trade_analysis(
                    db_path=db_path,
                    coin=coin,
                    bot=bot,
                    chat_id=chat_id,
                    ensemble=ensemble,
                    tracker=tracker,
                )

    logger.info(
        "Discovery: complete scanned=%d safe=%d added=%d expired=%d",
        len(merged_candidates),
        len(scored_candidates),
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
    lines = ["🔥 <b>NEW BREAKOUT CANDIDATES DETECTED</b>", ""]
    for coin in newly_added:
        direction = str(coin.get("direction", "neutral") or "neutral")
        emoji, label = _setup_badge(direction)
        lines.append(
            f"{emoji} <b>{label}: {coin['symbol']}</b> "
            f"(Score: {int(coin.get('signal_score', 0) or 0)}/100, "
            f"{int(coin.get('signals_firing', 0) or 0)} signals)"
        )
        lines.append(f"   {str(coin.get('breakout_description', 'No clear setup yet'))}")
        lines.append(f"   Vol: ${float(coin.get('volume_24h', 0) or 0):,.0f}")
        lines.append("")

    active_count = len(await get_active_hot_list(db_path))
    lines.extend(
        [
            "Use /trade <SYMBOL> for full analysis.",
            f"Active hot list: {active_count} coin(s)",
        ]
    )

    try:
        await bot.send_message(
            chat_id=chat_id,
            text="\n".join(lines).strip(),
            parse_mode="HTML",
        )
    except Exception:
        logger.exception("Discovery: failed to send Telegram alert")


async def _run_instant_trade_analysis(
    *,
    db_path: str,
    coin: dict,
    bot,
    chat_id: int | str | None,
    ensemble,
    tracker,
) -> None:
    symbol = str(coin.get("symbol", "") or "").upper()
    logger.info(
        "Discovery: high-score setup %s %d/100 (%d signals), running instant analysis",
        symbol,
        int(coin.get("signal_score", 0) or 0),
        int(coin.get("signals_firing", 0) or 0),
    )

    try:
        ta_signal = await _instant_ta_signal(coin, db_path=db_path)
        if ta_signal is None:
            return

        current_price = float(coin.get("price_usd", 0) or ta_signal.current_price or 0)
        if current_price <= 0:
            current_price = ta_signal.current_price

        ml_prediction = None
        if ensemble is not None and hasattr(ensemble, "predict_from_ohlcv"):
            try:
                ml_prediction = ensemble.predict_from_ohlcv(
                    symbol,
                    ta_signal._close,
                    ta_signal._high,
                    ta_signal._low,
                    ta_signal._volume,
                    ta_signal._open,
                )
            except Exception:
                logger.exception("Discovery: ML prediction failed for %s", symbol)

        signal = generate_crypto_trade_signal(
            symbol=symbol,
            current_price=current_price,
            ta_signal=ta_signal,
            composite=None,
            verdict=None,
            event_data=None,
            regime="default",
            ml_prediction=ml_prediction,
            breakout_score=int(coin.get("signal_score", 0) or 0),
            breakout_direction=str(coin.get("direction", "neutral") or "neutral"),
            breakout_description=str(coin.get("breakout_description", "") or ""),
        )
        if signal is None or signal.confluence_count < 3:
            return

        prediction_id: int | None = None
        if tracker is not None and hasattr(tracker, "log_prediction"):
            prediction_id = await tracker.log_prediction(
                symbol=signal.symbol,
                asset_type="crypto",
                predicted_direction="bullish" if signal.direction == "LONG" else "bearish",
                confidence=signal.confidence,
                price_at_prediction=current_price,
                source="discovery_breakout",
                reasoning=signal.reasoning,
                predicted_target=signal.tp2,
            )
        await save_signal_to_log(signal, db_path, signal_id=prediction_id)

        if bot is None or chat_id is None:
            return

        direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
        direction_label = "LONG" if signal.direction == "LONG" else "SHORT BIAS"
        lines = [
            "🔥 <b>BREAKOUT TRADE ALERT</b>",
            "",
            (
                f"{direction_emoji} <b>{signal.symbol} {direction_label}</b> "
                f"(Breakout Score: {int(coin.get('signal_score', 0) or 0)}/100, "
                f"{int(coin.get('signals_firing', 0) or 0)} signals)"
            ),
            "",
            f"Entry: ${signal.entry_low:,.4g} - ${signal.entry_high:,.4g}",
            f"SL: ${signal.stop_loss:,.4g} ({signal.stop_loss_pct:.1f}%)",
            f"TP1: ${signal.tp1:,.4g} | TP2: ${signal.tp2:,.4g} | TP3: ${signal.tp3:,.4g}",
            "",
            "<b>Breakout signals:</b>",
            str(coin.get("breakout_description", "N/A")),
            "",
            "<b>Trade confluence:</b>",
        ]
        for factor in signal.confluence_factors[:5]:
            lines.append(_format_confluence_line(factor.description, factor.strength))
        lines.extend(
            [
                "",
                "Discovered just now. This is a time-sensitive setup.",
                f"Use /trade {signal.symbol} for full details.",
                f"Log with /log {signal.symbol} <ENTRY> <SIZE> if you enter.",
            ]
        )
        await bot.send_message(
            chat_id=chat_id,
            text="\n".join(lines),
            parse_mode="HTML",
        )
    except Exception:
        logger.exception("Discovery: instant trade analysis failed for %s", symbol)


async def _instant_ta_signal(coin: dict, *, db_path: str):
    close = coin.get("_close")
    high = coin.get("_high")
    low = coin.get("_low")
    volume = coin.get("_volume")
    open_ = coin.get("_open")
    symbol = str(coin.get("symbol", "") or "").upper()

    if all(series is not None for series in (close, high, low, volume, open_)) and len(close) >= 60:
        return compute_signals(
            symbol,
            close,
            high,
            low,
            timeframe="daily",
            volume=volume,
            open_=open_,
        )

    coin_id = str(coin.get("id", "") or coin.get("coingecko_id", "") or "")
    if not coin_id:
        return None

    candles = await fetch_crypto_candles(coin_id, db_path=db_path)
    if candles is None:
        return None
    close, high, low, volume, open_ = candles
    return compute_signals(
        symbol,
        close,
        high,
        low,
        timeframe="daily",
        volume=volume,
        open_=open_,
    )


def _format_confluence_line(description: str, strength: float) -> str:
    filled = max(0, min(5, int(round(strength * 5))))
    bar = ("#" * filled) + ("-" * (5 - filled))
    return f"[{bar}] {description}"


def _setup_badge(direction: str) -> tuple[str, str]:
    if direction == "long_setup":
        return "🟢", "LONG SETUP"
    if direction == "short_setup":
        return "🔴", "SHORT SETUP"
    return "⚪", "WATCHING"


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
