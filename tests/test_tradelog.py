from __future__ import annotations

import os

import aiosqlite
import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "123")
os.environ.setdefault("FINNHUB_API_KEY", "test-key")
os.environ.setdefault("COINGECKO_API_KEY", "test-key")

from src.bot.handlers import (  # noqa: E402
    activate_logged_trade,
    check_trade_levels,
    close_logged_trade,
    load_trade_journal_snapshot,
    save_signal_to_log,
)
from src.trade import OpenPosition, PortfolioTracker, TradeSignal  # noqa: E402


def _trade_signal(
    *,
    symbol: str = "BTC",
    direction: str = "LONG",
    stop_loss: float = 68_000.0,
    tp1: float = 69_500.0,
    tp2: float = 70_500.0,
    tp3: float = 71_500.0,
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        direction=direction,
        confidence=0.82,
        entry_low=68_200.0,
        entry_high=68_500.0,
        stop_loss=stop_loss,
        stop_loss_pct=0.7,
        risk_per_unit=500.0,
        tp1=tp1,
        tp1_pct=1.5,
        tp2=tp2,
        tp2_pct=2.9,
        tp3=tp3,
        tp3_pct=4.4,
        risk_reward=2.0,
        timeframe="1-3 days",
        reasoning="Test setup",
        manipulation_check="No manipulation flags.",
        confluence_count=5,
        suggested_risk_pct=1.5,
    )


class _FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send_message(self, *, chat_id: int | str, text: str, parse_mode: str | None = None) -> None:
        self.messages.append(
            {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        )


@pytest.mark.asyncio
async def test_trade_log_lifecycle_and_journal_snapshot(db) -> None:
    signal = _trade_signal()

    trade_id = await save_signal_to_log(signal, db.db_path, signal_id=42)
    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            "SELECT signal_id, status FROM trade_log WHERE id = ?",
            (trade_id,),
        )
        row = await cursor.fetchone()
    assert row == (42, "pending")

    activated = await activate_logged_trade(db.db_path, "BTC", 68_500.0, 0.015)
    assert activated is not None
    assert activated["id"] == trade_id
    assert activated["risk_amount"] == pytest.approx(7.5)
    assert activated["entry_cost"] == pytest.approx(1_027.5)

    closed = await close_logged_trade(db.db_path, "BTC", 69_500.0, "tp1")
    assert closed is not None
    assert closed["status"] == "closed_tp1"
    assert closed["realized_pnl"] == pytest.approx(15.0)
    assert closed["r_multiple"] == pytest.approx(2.0)

    snapshot = await load_trade_journal_snapshot(db.db_path)
    assert snapshot["total_trades"] == 1
    assert snapshot["wins"] == 1
    assert snapshot["losses"] == 0
    assert snapshot["active_trades"] == 0
    assert snapshot["net_pnl"] == pytest.approx(15.0)
    assert snapshot["tp1_hit_rate"] == pytest.approx(100.0)
    assert snapshot["profit_factor"] == float("inf")
    assert snapshot["recent_trades"][0]["symbol"] == "BTC"


@pytest.mark.asyncio
async def test_check_trade_levels_marks_hits_and_notifies(db) -> None:
    signal = _trade_signal(stop_loss=90.0, tp1=110.0, tp2=120.0, tp3=130.0)
    await save_signal_to_log(signal, db.db_path)
    activated = await activate_logged_trade(db.db_path, "BTC", 100.0, 1.0)
    assert activated is not None

    tracker = PortfolioTracker()
    tracker.add_position(
        OpenPosition(
            symbol="BTC",
            asset_type="crypto",
            direction="LONG",
            entry_price=100.0,
            stop_loss=90.0,
            tp1=110.0,
            risk_pct=1.5,
            opened_at=0.0,
        )
    )
    bot = _FakeBot()

    await check_trade_levels(
        "BTC",
        110.0,
        db.db_path,
        bot,
        123,
        portfolio_tracker=tracker,
    )

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            "SELECT tp1_hit, sl_hit FROM trade_log WHERE symbol = 'BTC' AND status = 'active'"
        )
        row = await cursor.fetchone()
    assert row == (1, 0)
    assert tracker.positions["BTC"].tp1_hit is True
    assert len(bot.messages) == 1
    assert "TP1 HIT" in str(bot.messages[0]["text"])

    await check_trade_levels(
        "BTC",
        89.0,
        db.db_path,
        bot,
        123,
        portfolio_tracker=tracker,
    )
    await check_trade_levels(
        "BTC",
        88.0,
        db.db_path,
        bot,
        123,
        portfolio_tracker=tracker,
    )

    async with aiosqlite.connect(db.db_path) as connection:
        cursor = await connection.execute(
            "SELECT tp1_hit, sl_hit FROM trade_log WHERE symbol = 'BTC' AND status = 'active'"
        )
        row = await cursor.fetchone()
    assert row == (1, 1)
    assert len(bot.messages) == 2
    assert "STOP LOSS HIT" in str(bot.messages[1]["text"])
