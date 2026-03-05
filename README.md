# AlphaRadar (Phase 1)

AlphaRadar is a personal daily financial intelligence Telegram bot.

Phase 1 scope:
- Data aggregation from Finnhub, CoinGecko, DeFi Llama, and SEC EDGAR
- Daily scheduled Telegram briefing
- On-demand briefing and watchlist management
- SQLite persistence for watchlist, cache, and API usage

No AI/ML/LLM logic is included in this phase.

## Quick Start

1. Create env file:
```bash
cp .env.example .env
```

2. Install dependencies:
```bash
pip install -e .[dev]
```

3. Run bot:
```bash
python -m src.main
```

4. Optional helper to get your Telegram chat id:
```bash
python scripts/get_chat_id.py
```

## Docker

```bash
docker compose up --build
```

## Commands

- `/start`
- `/briefing`
- `/watchlist`
- `/add <symbol>`
- `/remove <symbol>`
- `/status`
- `/help`

