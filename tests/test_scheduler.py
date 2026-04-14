from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.bot.scheduler import setup_scheduler


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
