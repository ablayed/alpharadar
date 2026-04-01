# AlphaRadar

AlphaRadar is a personal Telegram bot for daily market briefings, watchlist tracking, alerts, sentiment, and trade journaling.

## Required Environment

Copy `.env.example` to `.env` and set at least:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `FINNHUB_API_KEY`
- `COINGECKO_API_KEY`
- `GEMINI_API_KEY` if you want the agent pipeline enabled
- `TIMEZONE` if you do not want the default `Europe/London`

## Local Run

```bash
cp .env.example .env
pip install -e .[dev]
python -m src.main
```

Optional helper to get your Telegram chat id:

```bash
python scripts/get_chat_id.py
```

## Docker

```bash
docker compose up --build -d
docker logs alpharadar --tail 50
```

Persistent state lives in:

- `data/alpharadar.db`
- `logs/`

Optional FinBERT model files can be mounted from:

- `models/finbert-onnx/`

If the model is missing, the bot still starts and sentiment features degrade gracefully.

## DigitalOcean Deployment

Use a fresh Ubuntu droplet with Docker already installed, or install Docker before adding your user to the `docker` group.

### 1. Initial Server Setup

SSH in as root:

```bash
ssh root@<DROPLET_IP>
```

Create an unprivileged deploy user, copy your SSH access, add swap, and enable the firewall:

```bash
adduser ablaye
usermod -aG sudo ablaye
usermod -aG docker ablaye

rsync --archive --chown=ablaye:ablaye ~/.ssh /home/ablaye

fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

ufw allow OpenSSH
ufw enable

(crontab -l 2>/dev/null; echo "0 3 * * 0 docker system prune -f >> /var/log/docker-prune.log 2>&1") | crontab -

reboot
```

Then reconnect as your deploy user:

```bash
ssh ablaye@<DROPLET_IP>
```

### 2. GitHub Access

Generate a deploy key on the droplet and add the public key to your GitHub account or repo deploy keys:

```bash
ssh-keygen -t ed25519 -C "digitalocean-alpharadar"
cat ~/.ssh/id_ed25519.pub
```

### 3. Clone and Configure

```bash
git clone git@github.com:ablayed/alpharadar.git
cd alpharadar
cp .env.example .env
nano .env
```

Set the production secrets in `.env`.

Important migration note:

- Copy `data/alpharadar.db` from the Azure VM if you want to preserve watchlist state, briefing history, trade logs, and cached usage data.
- Copy `models/finbert-onnx/` from the Azure VM if you want FinBERT sentiment available immediately on the droplet.
- `GROQ_API_KEY` is not used by the current codebase. The optional LLM key is `GEMINI_API_KEY`.

Example state sync from your old VM:

```bash
scp azureuser@<AZURE_IP>:/path/to/alpharadar/data/alpharadar.db ./data/alpharadar.db
scp -r azureuser@<AZURE_IP>:/path/to/alpharadar/models/finbert-onnx ./models/
```

### 4. Build and Start

```bash
docker compose build
docker compose up -d
docker logs alpharadar --tail 50
docker stats --no-stream
```

### 5. Parallel Run and Cutover

- Keep Azure as primary for 3 to 5 days.
- Use a test bot token on DigitalOcean first.
- Verify Telegram commands such as `/status`, `/history`, and `/funding`.
- Once the droplet is stable, stop Azure, switch DigitalOcean to the real bot token, and restart the container.

## Core Telegram Commands

- `/briefing`
- `/watchlist`
- `/add <symbol>`
- `/remove <symbol>`
- `/status`
- `/history`
- `/funding`
- `/alerts`
- `/help`
