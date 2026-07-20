from __future__ import annotations

from types import SimpleNamespace

import aiosqlite
import pytest

from src.agents.state import DebateVerdict
from src.bot import scheduler
from src.bot.scheduler import setup_scheduler
from src.reflection.tracker import PredictionTracker


class _FakeJob:
    def __init__(self) -> None:
        self.next_t = "scheduled"

    def schedule_removal(self) -> None:
        raise AssertionError("fallback scheduling should not run in this test")


class _FakeJobQueue:
    def __init__(self) -> None:
        self.daily_calls: list[dict] = []
        self.repeating_calls: list[dict] = []

    def run_daily(self, callback, time, days=None, name=None, chat_id=None):
        self.daily_calls.append(
            {
                "callback": callback,
                "time": time,
                "days": days,
                "name": name,
                "chat_id": chat_id,
            }
        )
        return _FakeJob()

    def run_repeating(self, callback, interval, first, name=None):
        self.repeating_calls.append(
            {
                "callback": callback,
                "interval": interval,
                "first": first,
                "name": name,
            }
        )
        return _FakeJob()


@pytest.mark.asyncio
async def test_setup_scheduler_registers_weekly_reflection_for_sunday() -> None:
    job_queue = _FakeJobQueue()
    application = SimpleNamespace(job_queue=job_queue)

    await setup_scheduler(application)

    weekly_reflection = next(call for call in job_queue.daily_calls if call["name"] == "weekly_reflection")
    assert weekly_reflection["days"] == (0,)
    assert weekly_reflection["time"].hour == 8
    assert weekly_reflection["time"].minute == 0
    assert getattr(weekly_reflection["time"].tzinfo, "key", None) == "UTC"


@pytest.mark.asyncio
async def test_setup_scheduler_registers_weekly_ml_training_for_saturday() -> None:
    job_queue = _FakeJobQueue()
    application = SimpleNamespace(job_queue=job_queue)

    await setup_scheduler(application)

    weekly_ml = next(call for call in job_queue.daily_calls if call["name"] == "weekly_ml_training")
    assert weekly_ml["days"] == (6,)
    assert weekly_ml["time"].hour == 6
    assert weekly_ml["time"].minute == 0


@pytest.mark.asyncio
async def test_setup_scheduler_registers_outcome_jobs_for_10_and_22_utc() -> None:
    job_queue = _FakeJobQueue()
    application = SimpleNamespace(job_queue=job_queue)

    await setup_scheduler(application)

    outcome_jobs = [
        call for call in job_queue.daily_calls
        if call["name"] in {"record_prediction_outcomes_10", "record_prediction_outcomes_22"}
    ]

    assert len(outcome_jobs) == 2
    assert {call["time"].hour for call in outcome_jobs} == {10, 22}
    assert all(call["time"].minute == 0 for call in outcome_jobs)
    assert all(getattr(call["time"].tzinfo, "key", None) == "UTC" for call in outcome_jobs)
    assert all(call["days"] == tuple(range(7)) for call in outcome_jobs)


@pytest.mark.asyncio
async def test_setup_scheduler_registers_discovery_scans_for_630_london() -> None:
    job_queue = _FakeJobQueue()
    application = SimpleNamespace(job_queue=job_queue)

    await setup_scheduler(application)

    discovery_jobs = [
        call for call in job_queue.daily_calls
        if call["name"] in {"discovery_scan_06", "discovery_scan_18"}
    ]

    assert len(discovery_jobs) == 2
    assert {call["time"].hour for call in discovery_jobs} == {6, 18}
    assert all(call["time"].minute == 30 for call in discovery_jobs)
    assert all(getattr(call["time"].tzinfo, "key", None) == "Europe/London" for call in discovery_jobs)


@pytest.mark.asyncio
async def test_auto_trade_scan_keeps_neutral_prediction_history_but_promotes_no_trade(
    db,
    monkeypatch,
) -> None:
    tracker = PredictionTracker(db.db_path)
    verdict = DebateVerdict(
        symbol="ETH",
        display_name="ETH",
        prevailing_view="neutral",
        confidence=0.9,
        summary="Evidence is split.",
        bull_thesis="Bull case",
        bear_thesis="Bear case",
        key_risk="Uncertain direction",
        action_context="Wait for confirmation",
    )
    briefing_data = SimpleNamespace(
        regime_state=SimpleNamespace(regime=SimpleNamespace(value="risk_on")),
        crypto_prices=[SimpleNamespace(symbol="ETH", current_price=2_500.0, change_24h_pct=0.0)],
        ta_signals={},
        composite_scores={"ETH": SimpleNamespace(signal_breakdown={"sentiment": 0.2})},
        funding={},
        oi={},
        cme_gap=None,
        fear_greed=SimpleNamespace(signal="bullish", value=50, description="Constructive sentiment"),
        dxy=None,
        agent_analysis=SimpleNamespace(debate_verdicts=[verdict]),
        event_data=[],
    )

    class FakeAggregator:
        async def gather_briefing_data(self, *, include_agent_analysis: bool):
            assert include_agent_analysis is True
            await tracker.log_prediction(
                symbol="ETH",
                asset_type="crypto",
                predicted_direction=verdict.prevailing_view,
                confidence=verdict.confidence,
                price_at_prediction=2_500.0,
                source="analysis",
                reasoning=verdict.summary,
            )
            return briefing_data

    class FakeBot:
        def __init__(self) -> None:
            self.sent_messages: list[dict] = []

        async def send_message(self, **kwargs) -> None:
            self.sent_messages.append(kwargs)

    monkeypatch.setattr(scheduler, "DataAggregator", FakeAggregator)
    bot = FakeBot()
    context = SimpleNamespace(
        job=SimpleNamespace(data={"hour": 7, "minute": 0}),
        bot_data={"aggregator": FakeAggregator(), "tracker": tracker, "db": db, "ml_ensemble": None},
        bot=bot,
        job_queue=None,
    )

    await scheduler.auto_trade_scan(context)

    assert bot.sent_messages == []
    async with aiosqlite.connect(db.db_path) as connection:
        prediction_rows = await (await connection.execute(
            "SELECT predicted_direction, source FROM predictions"
        )).fetchall()
        trade_log_rows = await (await connection.execute("SELECT id FROM trade_log")).fetchall()

    assert prediction_rows == [("neutral", "analysis")]
    assert trade_log_rows == []
