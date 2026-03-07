from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from telegram.constants import ParseMode
    from telegram.ext import Application, Defaults, ContextTypes

    from src.config import settings

    tz = ZoneInfo(settings.timezone)
    now = datetime.datetime.now(tz)
    fire_at = now + datetime.timedelta(seconds=90)
    fire_time = fire_at.time().replace(tzinfo=tz)

    print(f"Current time:  {now.strftime('%H:%M:%S %Z')}")
    print(f"Will fire at:  {fire_at.strftime('%H:%M:%S %Z')} (90 seconds from now)")
    print(f"Chat ID:       {settings.telegram_chat_id}")
    print()

    async def test_job(context: ContextTypes.DEFAULT_TYPE) -> None:
        print("\nSCHEDULER FIRED SUCCESSFULLY!")
        await context.bot.send_message(
            chat_id=context.job.chat_id,
            text="AlphaRadar scheduler test: job fired on time!",
        )
        print("Message sent to Telegram!")
        print("\nScheduler works. You can Ctrl+C now.")

    async def post_init(app: Application) -> None:
        jq = app.job_queue
        if jq is None:
            raise RuntimeError("Job queue unavailable.")

        job = jq.run_daily(
            callback=test_job,
            time=fire_time,
            days=tuple(range(7)),
            name="scheduler_test",
            chat_id=int(settings.telegram_chat_id),
        )

        next_run = getattr(job, "next_t", None) or getattr(job, "next_run_time", None)
        print(f"run_daily next_run = {next_run}")

        if next_run is None:
            print("next_run is None - run_daily has the ZoneInfo bug!")
            print("Testing fallback with run_once...")
            job.schedule_removal()

            jq.run_once(
                callback=test_job,
                when=90,
                chat_id=int(settings.telegram_chat_id),
                name="scheduler_test_fallback",
            )
            print("run_once fallback scheduled (90 seconds)")
        else:
            print(f"run_daily works! Next fire: {next_run}")

        print("\nWaiting for job to fire... (check Telegram in ~90 seconds)")

    app = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .post_init(post_init)
        .build()
    )

    print("Starting bot (Ctrl+C to stop after test completes)...")
    app.run_polling()


if __name__ == "__main__":
    main()
