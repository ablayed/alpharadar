import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    #  1. Check .env loads 
    header("1. Configuration Check")
    try:
        from src.config import settings

        print(f"  TELEGRAM_CHAT_ID = '{settings.telegram_chat_id}'")
        print(f"  FINNHUB_API_KEY  = {'SET' if settings.finnhub_api_key.get_secret_value() else 'EMPTY'}")
        print(f"  COINGECKO_KEY    = {'SET' if settings.coingecko_api_key.get_secret_value() else 'EMPTY'}")
        print(f"  BRIEFING_HOUR    = {settings.briefing_hour}")
        print(f"  BRIEFING_MINUTE  = {settings.briefing_minute}")
        print(f"  TIMEZONE         = {settings.timezone}")
        print(f"  DB_PATH          = {settings.db_path}")

        # Check chat_id is numeric
        if not settings.telegram_chat_id.strip().lstrip("-").isdigit():
            errors.append(
                f"TELEGRAM_CHAT_ID is '{settings.telegram_chat_id}'  must be a numeric ID like 123456789"
            )
        else:
            print(f"   Chat ID is numeric: {settings.telegram_chat_id}")

    except Exception as e:
        errors.append(f"Config load failed: {e}")
        print(f"\n  FATAL: {e}")
        _print_summary(errors, warnings)
        return

    #  2. Check Telegram Bot Token 
    header("2. Telegram Bot Token Check")
    import httpx

    token = settings.telegram_bot_token.get_secret_value()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
        if resp.status_code == 200:
            bot_info = resp.json()["result"]
            print(f"   Bot name: {bot_info['first_name']}")
            print(f"   Bot username: @{bot_info.get('username', 'N/A')}")
        else:
            errors.append(f"Bot token invalid: HTTP {resp.status_code}")
            print(f"   Token invalid: {resp.text}")

    #  3. Check Chat ID by sending a test message 
    header("3. Chat ID Delivery Test")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": settings.telegram_chat_id,
                "text": "🔧 AlphaRadar diagnostic: If you see this, your Chat ID is correct!",
            },
        )
        if resp.status_code == 200:
            print(f"   Message sent successfully to chat_id={settings.telegram_chat_id}")
            print(f"   Check Telegram now  you should see the diagnostic message")
        else:
            data = resp.json()
            errors.append(f"Cannot send to chat_id={settings.telegram_chat_id}: {data.get('description', resp.text)}")
            print(f"   FAILED: {data.get('description', resp.text)}")
            print()
            print(f"  LIKELY FIX: Your TELEGRAM_CHAT_ID is wrong.")
            print(f"  Run: python scripts/get_chat_id.py")
            print(f"  Send a message to your bot, copy the ID it prints,")
            print(f"  then update .env with the correct ID.")

    #  4. Check Finnhub API 
    header("4. Finnhub API Check")
    async with httpx.AsyncClient() as client:
        fh_key = settings.finnhub_api_key.get_secret_value()
        resp = await client.get(
            f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={fh_key}",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("c", 0)
            if price > 0:
                print(f"   AAPL current price: ${price}")
            else:
                warnings.append("Finnhub returned price=0 for AAPL  possibly outside market hours")
                print(f"   AAPL price is 0  market may be closed. Raw: {data}")
        elif resp.status_code == 401:
            errors.append("Finnhub API key is invalid")
            print(f"   API key invalid (401)")
        elif resp.status_code == 429:
            warnings.append("Finnhub rate limited")
            print(f"   Rate limited (429)  wait a minute and retry")
        else:
            errors.append(f"Finnhub unexpected response: {resp.status_code}")
            print(f"   Unexpected: HTTP {resp.status_code}: {resp.text[:200]}")

    #  5. Check CoinGecko API 
    header("5. CoinGecko API Check")
    async with httpx.AsyncClient() as client:
        cg_key = settings.coingecko_api_key.get_secret_value()
        resp = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            headers={"x-cg-demo-api-key": cg_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            btc_price = data.get("bitcoin", {}).get("usd", "N/A")
            print(f"   BTC price: ${btc_price}")
        elif resp.status_code == 401:
            errors.append("CoinGecko API key invalid or wrong base URL")
            print(f"   Auth failed  check key and ensure using api.coingecko.com (not pro-api)")
        else:
            errors.append(f"CoinGecko unexpected: {resp.status_code}")
            print(f"   HTTP {resp.status_code}: {resp.text[:200]}")

    #  6. Check DeFi Llama API 
    header("6. DeFi Llama API Check")
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.llama.fi/v2/historicalChainTvl", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                latest_tvl = data[-1].get("tvl", 0)
                print(f"   Total crypto TVL: ${latest_tvl / 1e9:.1f}B")
            else:
                warnings.append("DeFi Llama returned empty data")
                print(f"   Empty response")
        else:
            errors.append(f"DeFi Llama: HTTP {resp.status_code}")
            print(f"   HTTP {resp.status_code}")

    #  7. Check Database 
    header("7. Database Check")
    try:
        import aiosqlite

        db_path = settings.db_path
        if not Path(db_path).exists():
            warnings.append(f"Database doesn't exist yet at {db_path}  will be created on first run")
            print(f"   {db_path} doesn't exist yet (normal for first run)")
        else:
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute("SELECT COUNT(*) FROM watchlist WHERE is_active = 1")
                row = await cursor.fetchone()
                count = row[0] if row else 0
                print(f"   Database exists, {count} active watchlist items")

                cursor = await conn.execute(
                    "SELECT asset_type, COUNT(*) FROM watchlist WHERE is_active = 1 GROUP BY asset_type"
                )
                rows = await cursor.fetchall()
                for asset_type, c in rows:
                    print(f"     {asset_type}: {c} items")
    except Exception as e:
        errors.append(f"Database error: {e}")
        print(f"   {e}")

    #  8. Timezone / Scheduler Check 
    header("8. Timezone & Scheduler Check")
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    print(f"  System time (UTC):   {datetime.utcnow().strftime('%H:%M:%S')}")
    print(f"  Time in {settings.timezone}: {now.strftime('%H:%M:%S')}")
    print(f"  Briefing scheduled:  {settings.briefing_hour:02d}:{settings.briefing_minute:02d}")

    target = now.replace(
        hour=settings.briefing_hour,
        minute=settings.briefing_minute,
        second=0,
        microsecond=0,
    )
    if target <= now:
        from datetime import timedelta
        target += timedelta(days=1)
        print(f"   Already passed today. Next: tomorrow at {target.strftime('%H:%M %Z')}")
    else:
        diff = target - now
        mins = int(diff.total_seconds() / 60)
        print(f"   Fires in {mins} minutes ({target.strftime('%H:%M %Z')})")

    #  9. Check file encoding 
    header("9. File Encoding Check")
    handlers_path = Path("src/bot/handlers.py")
    if handlers_path.exists():
        raw = handlers_path.read_bytes()
        if b"\xc3\xb0\xc2\x9f" in raw:
            errors.append("handlers.py has double-encoded UTF-8 emoji (mojibake). Needs fixing.")
            print(f"   MOJIBAKE DETECTED in handlers.py!")
            print(f"    The emoji were double-encoded. This causes silent failures.")
            print(f"    FIX: Re-save the file with proper UTF-8 encoding,")
            print(f"    or use the fix script below.")
        elif "📡" in raw.decode("utf-8", errors="replace"):
            print(f"   handlers.py has correct UTF-8 emoji")
        else:
            warnings.append("handlers.py may have encoding issues  emoji not found as expected")
            print(f"   Could not verify emoji encoding  check manually")
    else:
        errors.append("src/bot/handlers.py not found!")
        print(f"   File not found")

    # Also check scheduler.py
    scheduler_path = Path("src/bot/scheduler.py")
    if scheduler_path.exists():
        raw = scheduler_path.read_bytes()
        print(f"  scheduler.py size: {len(raw)} bytes, encoding looks OK")

    #  Summary 
    _print_summary(errors, warnings)


def _print_summary(errors: list[str], warnings: list[str]) -> None:
    header("DIAGNOSTIC SUMMARY")
    if not errors and not warnings:
        print("   All checks passed! The bot should work.")
    if warnings:
        print(f"   {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    - {w}")
    if errors:
        print(f"   {len(errors)} error(s) to fix:")
        for e in errors:
            print(f"    - {e}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
