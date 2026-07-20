# P2 — Historical Data Feasibility Audit

Access date: 2026-07-20. Audit scope is Binance Spot, long-only research only; no strategy, labels, features, backtester, full downloader, database or production runtime changes were made.

## Verdict

**FEASIBLE_WITH_LIMITATIONS.** The required lifecycle cases are now independently corroborated by official Binance announcements and checksum-verified archive observations. The limitations below remain explicit and non-critical to testing the three strategy families.

## STEP 1 — REPOSITORY FINDINGS

**VERIFIED:** Python 3.12 project with `httpx`, `aiosqlite`, NumPy, pytest and Ruff. Reusable patterns are HTTP/logging and test fixtures. There is no local historical Binance OHLCV store, no lifecycle/universe table, and no `/api/v3/klines` caller. Binance code is limited to a live miniTicker WebSocket and current aggregate-trades request. Current `RNDR -> RENDER` routing is a live alias only, not historic evidence. Production modules, prediction generation, Telegram behavior and database schemas remain untouched.

**RECOMMENDATION:** use `data.binance.vision` archives for historical P3 ingestion. REST klines are capped at 1,000 candles/call and lack archive checksum companions; archives are monthly/daily, checksum-backed and reproducible. REST may later fill recent incremental gaps only.

## Requirements matrix

| Field | Momentum | Rotation | Breakout |
|---|---|---|---|
| 1h OHLCV | REQUIRED | REQUIRED | REQUIRED |
| 4h / daily OHLCV | USEFUL_BUT_OPTIONAL | USEFUL_BUT_OPTIONAL | USEFUL_BUT_OPTIONAL / NOT_REQUIRED_FOR_FIRST_PASS |
| quote volume / trades | USEFUL_BUT_OPTIONAL / USEFUL_BUT_OPTIONAL | REQUIRED / REQUIRED | REQUIRED / USEFUL_BUT_OPTIONAL |
| taker-buy volume | NOT_REQUIRED_FOR_FIRST_PASS | NOT_REQUIRED_FOR_FIRST_PASS | USEFUL_BUT_OPTIONAL |
| listing age / historic membership | REQUIRED | REQUIRED | REQUIRED |
| delisting date / token mapping | USEFUL_BUT_OPTIONAL / REQUIRED | REQUIRED / REQUIRED | USEFUL_BUT_OPTIONAL / REQUIRED |
| market cap, spread, depth, individual trades | NOT_REQUIRED_FOR_FIRST_PASS | NOT_REQUIRED_FOR_FIRST_PASS | NOT_REQUIRED_FOR_FIRST_PASS |
| historic Sharia status | REQUIRED for execution eligibility; not required for broad non-execution research | same | same |

## VERIFIED — archive and timestamp evidence

Binance documents headerless 12-column Spot klines, checksum companions, all required intervals and microsecond timestamps from 2025-01-01: <https://github.com/binance/binance-public-data>. The bounded public probe used <https://data.binance.vision/> and downloaded **138,381 bytes** total (ZIPs plus checksums), far below 250 MB.

| Sample | Result |
|---|---|
| BTCUSDT 1h 2024-12 | 744 rows; milliseconds; complete |
| BTCUSDT 1h 2025-01 | 744 rows; microseconds; complete |
| ETHUSDT 4h 2025-01 | 186 rows; microseconds; complete |
| SOLUSDT 1d 2026-06 | 30 rows; microseconds; complete |
| KAITOUSDT 1h 2025-02 | first observed candle 2025-02-20 13:00Z; 203 rows |
| RNDRUSDT 1h 2024-07 | last observed candle 2024-07-22 02:00Z; 507 rows |

Every downloaded ZIP matched its official SHA-256 companion. All validated samples had valid ZIP members, 12 columns, numeric fields, OHLC/volume invariants, monotonic opens, zero duplicates and zero missing expected bars. DREPUSDT 2025-04 returned 404: **LIMITATION:** this is not proof of delisting. Full machine evidence is in `artifacts/p2`.

P2 tests prove millisecond/microsecond normalization, a 2025 boundary, ambiguous-value rejection, valid/invalid checksum handling, malformed columns, duplicates, gaps and deterministic artifacts. Existing application code does not parse Binance archive timestamps, so no existing production timestamp code misreads post-2025 archives; a future consumer must adopt P2 normalization.

## INFERENCE — universe and lifecycle

Option A historical market-cap top-N is not recommended for a low-cost first pass. Binance does not supply historical market caps. CoinGecko documents historical market-cap data but its current official documentation presents keyed historical routes and paid/inactive-asset limitations; no key was requested or used. Identifier collisions, dead assets and rebrands add reproducibility risk.

**RECOMMENDATION: Option B.** At each rebalance use only then-observable USDT Spot archive existence, observed listing age, trailing quote volume/trades, completeness and continuity. Never use present `exchangeInfo`, present universe or future volume. This is reproducible and survivorship-safer, though not market-cap top-30.

Future lifecycle fields: `canonical_asset_id, exchange_symbol, base_asset, quote_asset, valid_from, valid_to, event_type, predecessor_symbol, successor_symbol, conversion_ratio, source_url, confidence, observed_first_candle, observed_last_candle`. Distinguish pair removal, all-pair removal, token delisting, rebrand, swap, temporary gap and publication gap. First/last candles are proxies; missing archives alone are unsafe. RNDR/RENDER shows why this table is needed and must not be stitched in P2.

## LIMITATION — execution and Sharia data

Klines supply OHLCV, quote volume, trade count and taker-buy fields; they support realized-volatility and turnover proxies. They do not supply historic bid-ask spread, depth, price impact or realized slippage. Volume is neither market cap nor spread, and current order books cannot be projected backwards. A future first pass should use a conservative explicit fee plus slippage-sensitivity model, never claim observed historic depth.

System restrictions are stable: Spot only, long only, no leverage, lending, staking or yield. Token-level classifications can change and current allowlists cannot be back-applied. **RECOMMENDATION:** broad research results remain non-execution-eligible; execution eligibility uses only a small manually reviewed stable allowlist. Final criteria require qualified scholarly review; P2 makes no religious ruling.

## INFERENCE — storage/runtime

2019-01 through 2026-06 at roughly 50 point-in-time pairs/month means about 4,500 1h monthly files and 3.35M rows: ~150 MB compressed, ~400 MB CSV, ~150–300 MB Parquet. At 10 Mbps: about three minutes plus overhead; parsing about 2–10 minutes. Use canonical 1h and deterministic 4h/1d resampling. SQLite is adequate for metadata; a future analytical dataset may choose Parquet/DuckDB separately. Local development and VPS are suitable. AggTrades/trades are optional and potentially orders of magnitude larger.

## Sources

`artifacts/p2/source_manifest.json` records URLs and access date, including Binance archive/docs, REST docs, announcement index, and CoinGecko official endpoint, keyless and historical-data documentation.

## VERIFIED — lifecycle follow-up

| Case | Official authority | Archive evidence | Classification |
|---|---|---|---|
| AirDAO AMB/USDT | [2025-02-17 07:00 announcement](https://www.binance.com/en/support/announcement/detail/f3cc64a8758d4bb394e0830e5b31da97): all AMB Spot pairs, explicitly AMB/USDT, cease 2025-02-24 03:00 UTC | AMBUSDT 1h 2025-02: 555 rows, 2025-02-01 00:00 through 2025-02-24 02:00; checksum verified. 2025-03 archive 404. | VERIFIED_BY_ANNOUNCEMENT; VERIFIED_BY_ARCHIVE; post-event 404 is ARCHIVE_ABSENCE_ONLY. |
| Alpaca Finance ALPACA/USDT | [2025-04-24 04:00 announcement](https://www.binance.com/en/support/announcement/detail/0274f9d47da1437990bc13eb17b0ec99): all ALPACA Spot pairs cease 2025-05-02 03:00 UTC | Official/archive symbol is `ALPACAUSDT`. 1h 2025-05: 27 rows, 2025-05-01 00:00 through 2025-05-02 02:00; checksum verified. 2025-06 archive 404. | VERIFIED_BY_ANNOUNCEMENT; VERIFIED_BY_ARCHIVE; post-event 404 is ARCHIVE_ABSENCE_ONLY. |
| Render RNDR/USDT → RENDER/USDT | [2024-07-10 06:30 announcement](https://www.binance.com/en/support/announcement/detail/d1f2ae8d99b24439a7a900caa9bb6b3b): RNDR ends 2024-07-22 03:00; RENDER opens 2024-07-26 08:00; ratio 1:1. [Completion](https://www.binance.com/en/support/announcement/detail/7aa2bb48ab194bb9aebbc3f72b5819ed) published 2024-07-26 09:00. | RNDRUSDT last 2024-07-22 02:00; RENDERUSDT first 2024-07-26 08:00; both checksum verified. `open_time_delta_hours=102`; `missing_bar_count=101`; `announced_interruption_hours=101`. | VERIFIED_BY_ANNOUNCEMENT; VERIFIED_BY_ARCHIVE. Separate exchange symbols; no stitching. |
| Frontier FRONT/USDT → Self Chain SLF/USDT | [2024-08-19 11:00 announcement](https://www.binance.com/en/support/announcement/detail/3ab5488a00e04d4fb338c77ea28326a8): FRONT ends 2024-08-27 03:00; SLF opens 2024-08-30 08:00; ratio 1:1. [Completion](https://www.binance.com/en/support/announcement/detail/65f69bc68ca64182b6df3d958f14f38e) published 2024-08-30 10:00. | FRONTUSDT last 2024-08-27 02:00; SLFUSDT first 2024-08-30 08:00; both checksum verified. `open_time_delta_hours=78`; `missing_bar_count=77`; `announced_interruption_hours=77`. | VERIFIED_BY_ANNOUNCEMENT; VERIFIED_BY_ARCHIVE. Separate exchange symbols; no stitching. |

**LIMITATION:** the four cases validate the lifecycle method, not a complete global lifecycle catalogue. Future P3 must retain official source URLs, event types, confidence, predecessor/successor symbols and observed candle bounds. It must not infer a delisting from archive absence alone or stitch migration prices without a separately approved continuity policy.
