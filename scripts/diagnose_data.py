from __future__ import annotations

import asyncio
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure emoji-safe output on Windows terminals.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def truncate(obj, max_len=200):
    s = str(obj)
    return s[:max_len] + "..." if len(s) > max_len else s


async def main() -> None:
    import httpx
    from src.config import settings

    # 1. Finnhub Raw Tests
    header("1. FINNHUB - Raw API Calls")
    fh_key = settings.finnhub_api_key.get_secret_value()
    base = "https://finnhub.io/api/v1"

    async with httpx.AsyncClient(timeout=15) as client:
        # 1a. Stock quote
        print("\n  [1a] Quote for NVDA:")
        try:
            r = await client.get(f"{base}/quote", params={"symbol": "NVDA", "token": fh_key})
            print(f"       Status: {r.status_code}")
            data = r.json()
            print(f"       Response: {json.dumps(data, indent=2)}")
            if data.get("c", 0) == 0:
                print("       NOTE: Price is 0 - market closed or symbol issue")
        except Exception as e:
            print(f"       ERROR: {e}")

        # 1b. Market status
        print("\n  [1b] Market status (US):")
        try:
            r = await client.get(f"{base}/stock/market-status", params={"exchange": "US", "token": fh_key})
            print(f"       Status: {r.status_code}")
            print(f"       Response: {json.dumps(r.json(), indent=2)}")
        except Exception as e:
            print(f"       ERROR: {e}")

        # 1c. Index proxies
        print("\n  [1c] Index quotes (SPY, QQQ, VIX):")
        for sym in ["SPY", "QQQ", "^VIX"]:
            try:
                r = await client.get(f"{base}/quote", params={"symbol": sym, "token": fh_key})
                d = r.json()
                print(f"       {sym}: price={d.get('c')}, change={d.get('dp')}%")
            except Exception as e:
                print(f"       {sym}: ERROR {e}")

        # 1d. Company news
        print("\n  [1d] News for NVDA (last 2 days):")
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        try:
            r = await client.get(
                f"{base}/company-news",
                params={"symbol": "NVDA", "from": yesterday, "to": today, "token": fh_key},
            )
            news = r.json()
            print(f"       Status: {r.status_code}, Count: {len(news)}")
            for item in news[:3]:
                print(f"       - {item.get('headline', 'N/A')[:70]}")
        except Exception as e:
            print(f"       ERROR: {e}")

        # 1e. Earnings calendar
        print("\n  [1e] Earnings calendar (today):")
        try:
            r = await client.get(
                f"{base}/calendar/earnings",
                params={"from": today, "to": today, "token": fh_key},
            )
            data = r.json()
            earnings = data.get("earningsCalendar", [])
            print(f"       Status: {r.status_code}, Count: {len(earnings)}")
            for e in earnings[:5]:
                print(f"       - {e.get('symbol')}: {e.get('date')} ({e.get('hour')})")
        except Exception as e:
            print(f"       ERROR: {e}")

    # 2. CoinGecko Raw Tests
    header("2. COINGECKO - Raw API Calls")
    cg_key = settings.coingecko_api_key.get_secret_value()
    cg_headers = {"x-cg-demo-api-key": cg_key}

    async with httpx.AsyncClient(timeout=15) as client:
        # 2a. Coins markets
        print("\n  [2a] /coins/markets for BTC, ETH, SOL:")
        try:
            r = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": "bitcoin,ethereum,solana",
                    "price_change_percentage": "24h,7d",
                    "per_page": 10,
                    "page": 1,
                    "sparkline": "false",
                },
                headers=cg_headers,
            )
            print(f"       Status: {r.status_code}")
            if r.status_code == 200:
                coins = r.json()
                for c in coins:
                    print(
                        f"       {c['symbol'].upper()}: "
                        f"${c.get('current_price')} "
                        f"24h={c.get('price_change_percentage_24h')}% "
                        f"7d={c.get('price_change_percentage_7d_in_currency')}%"
                    )
            else:
                print(f"       Response: {r.text[:300]}")
        except Exception as e:
            print(f"       ERROR: {e}")

        # 2b. Trending
        print("\n  [2b] /search/trending:")
        try:
            r = await client.get(
                "https://api.coingecko.com/api/v3/search/trending",
                headers=cg_headers,
            )
            print(f"       Status: {r.status_code}")
            if r.status_code == 200:
                coins = r.json().get("coins", [])[:5]
                for c in coins:
                    item = c.get("item", {})
                    print(
                        f"       #{item.get('score', '?')+1 if isinstance(item.get('score'), int) else '?'} "
                        f"{item.get('name')} ({item.get('symbol')})"
                    )
            else:
                print(f"       Response: {r.text[:300]}")
        except Exception as e:
            print(f"       ERROR: {e}")

    # 3. DeFi Llama Raw Tests
    header("3. DEFI LLAMA - Raw API Calls")

    async with httpx.AsyncClient(timeout=15) as client:
        # 3a. Historical TVL
        print("\n  [3a] /v2/historicalChainTvl (last 2 entries):")
        try:
            r = await client.get("https://api.llama.fi/v2/historicalChainTvl")
            print(f"       Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                if len(data) >= 2:
                    latest = data[-1]
                    prev = data[-2]
                    tvl = latest["tvl"]
                    change = ((latest["tvl"] - prev["tvl"]) / prev["tvl"]) * 100
                    print(f"       Latest TVL: ${tvl/1e9:.1f}B")
                    print(f"       24h change: {change:+.2f}%")
                else:
                    print(f"       Only {len(data)} entries returned")
        except Exception as e:
            print(f"       ERROR: {e}")

        # 3b. Top protocol movers
        print("\n  [3b] /protocols (top 5 by daily change):")
        try:
            r = await client.get("https://api.llama.fi/protocols")
            print(f"       Status: {r.status_code}")
            if r.status_code == 200:
                protocols = r.json()
                valid = [
                    p for p in protocols
                    if isinstance(p.get("change_1d"), (int, float))
                    and p.get("tvl", 0) > 100_000_000
                ]
                sorted_p = sorted(valid, key=lambda p: abs(p["change_1d"]), reverse=True)
                for p in sorted_p[:5]:
                    print(
                        f"       {p['name']}: TVL=${p['tvl']/1e9:.1f}B "
                        f"1d={p['change_1d']:+.1f}%"
                    )
        except Exception as e:
            print(f"       ERROR: {e}")

    # 4. SEC EDGAR Raw Test
    header("4. SEC EDGAR - Raw API Call")

    async with httpx.AsyncClient(
        timeout=15,
        headers={
            "User-Agent": "AlphaRadar contact@alpharadar.dev",
            "Accept-Encoding": "gzip, deflate",
        },
    ) as client:
        print("\n  [4a] NVDA submissions (CIK 0001045810):")
        try:
            r = await client.get("https://data.sec.gov/submissions/CIK0001045810.json")
            print(f"       Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                recent = data.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                eight_ks = [
                    (dates[i], forms[i])
                    for i in range(min(len(forms), len(dates)))
                    if forms[i] == "8-K"
                ][:5]
                print(f"       Total recent filings: {len(forms)}")
                print(f"       Recent 8-K filings: {len(eight_ks)}")
                for date, form in eight_ks:
                    print(f"       - {date}: {form}")
            else:
                print(f"       Response: {r.text[:300]}")
        except Exception as e:
            print(f"       ERROR: {e}")

    # 5. Test the Aggregator Directly
    header("5. AGGREGATOR - Full Pipeline Test")

    try:
        from src.storage.database import Database
        from src.data.aggregator import DataAggregator

        db = Database(settings.db_path)
        await db.initialize()

        aggregator = DataAggregator(db=db, settings=settings)
        await aggregator.initialize()

        print("\n  Running gather_briefing_data()...")
        data = await aggregator.gather_briefing_data()

        print(f"\n  Results:")
        print(f"    is_weekend:       {data.is_weekend}")
        print(f"    market_is_open:   {data.market_is_open}")
        print(f"    market_session:   {data.market_session}")
        print(f"    index_prices:     {len(data.index_prices)} items")
        for ip in data.index_prices:
            print(f"      - {ip.symbol}: ${ip.current_price} ({ip.change_24h_pct:+.2f}%)")
        print(f"    stock_prices:     {len(data.stock_prices)} items")
        for sp in data.stock_prices[:5]:
            print(f"      - {sp.symbol}: ${sp.current_price} ({sp.change_24h_pct:+.2f}%)")
        print(f"    crypto_prices:    {len(data.crypto_prices)} items")
        for cp in data.crypto_prices[:5]:
            print(f"      - {cp.symbol}: ${cp.current_price} ({cp.change_24h_pct:+.2f}%)")
        print(f"    news_items:       {len(data.news_items)} items")
        print(f"    earnings_today:   {len(data.earnings_today)} items")
        print(f"    sec_filings:      {len(data.sec_filings)} items")
        print(f"    total_tvl:        {data.total_tvl}")
        print(f"    tvl_change_24h:   {data.tvl_change_24h_pct}")
        print(f"    defi_movers:      {len(data.defi_movers)} items")
        print(f"    trending_crypto:  {len(data.trending_crypto)} items")
        print(f"    failed_sections:  {data.failed_sections}")
        print(f"    coingecko_usage:  {data.coingecko_usage}")

        print("\n  Testing formatter...")
        from src.briefing.formatter import BriefingFormatter

        formatter = BriefingFormatter()
        if data.is_weekend:
            msg = formatter.format_weekend(data)
        else:
            msg = formatter.format_daily(data)

        print(f"    Message length: {len(msg)} chars (limit: 4096)")
        print(f"\n   FULL BRIEFING OUTPUT ")
        print(msg)
        print(f"   END BRIEFING ")

        await aggregator.close()

    except Exception as e:
        import traceback
        print(f"\n   AGGREGATOR FAILED: {e}")
        traceback.print_exc()

    # Summary
    header("DONE")
    print("  Check the output above for any ERRORs or 0-count sections.")
    print("  If raw API calls work but aggregator returns empty,")
    print("  the bug is in how your providers parse the responses.")


if __name__ == "__main__":
    asyncio.run(main())
