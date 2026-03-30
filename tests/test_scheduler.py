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
