# ⚡ CryptoMind — Self-Learning Autonomous Crypto Trading System

A working implementation of the full architecture: research ingestion → NLP →
signal generation → adaptive risk → paper execution → backtesting → self-improvement
loop, with an orchestrator, audit log, and live dashboard.

**Safety-first defaults:** paper trading only (simulated $100k account), with
kill switch, daily loss halt, gross-exposure caps, and a full audit trail.
Directional shorts and market-neutral pair hedging are supported (runtime
toggles); real order routing is deliberately **not** enabled. No API keys are
required for core operation — everything runs on public, key-free data sources
(an optional Gemini LLM advisor is the only key-gated extra; it is inert
without a key).

**Contents:** [Quick start](#quick-start) · [Install](#install) · [Start, stop, restart](#start-stop-restart) · [First run](#first-run-what-to-expect) · [Configure](#configure) · [Operate](#operate) · [Troubleshooting](#troubleshooting) · [API](#api)

## Architecture → Module map

| Blueprint component | Implementation |
|---|---|
| Orchestrator / Control Plane | `app/orchestrator.py` — decision loop, health, human-in-the-loop pause/kill |
| Research & Data Ingestion | `app/data/research.py` — self-expanding source pool (news RSS + combined Reddit multireddit + auto-spawned Google News feeds per coin), adaptive 429 backoff, source promotion/demotion, Fear & Greed index, self-directed research queue |
| Dynamic Universe Discovery | `app/data/universe.py` — extracts coin mentions from research + CoinGecko trending, validates against Coinbase listing & liquidity floor, expands/prunes the tradeable universe (core 6 protected, cap 12, held positions never pruned) |
| Market Data & Order Book Feed | `app/data/market.py` — live Coinbase Exchange candles, tickers, L2 order-book depth/imbalance |
| Derivatives Feed | `app/data/derivatives.py` — OKX public API: perp funding rates, open-interest history, long/short account ratio, taker buy/sell aggression (per asset, key-free) |
| NLP / Signal Generation | `app/nlp/sentiment.py` — crypto sentiment lexicon, per-asset scores, narrative detection (drop-in interface for FinBERT/CryptoBERT) |
| Signal Engine | `app/signals/engine.py` — 5-strategy ensemble: trend, mean-reversion, breakout, sentiment, microstructure; outputs direction, confidence, edge, invalidation |
| Adaptive Risk Manager | `app/risk/manager.py` — vol-adjusted sizing, exposure/position caps, drawdown kill switch, daily loss halt, loss-streak risk scaling, regime scaling, cooldowns |
| Execution Engine (Paper) | `app/execution/paper.py` — OMS with fees, slippage, stops, take-profits, ATR trailing stops |
| Backtesting & Validation | `app/backtest/engine.py` — event-driven sim on ~950 real hourly bars, in-sample vs out-of-sample walk-forward split, overfitting flag |
| Self-Improvement Loop | `app/learn/loop.py` — bandit-style meta-learner: scores every signal against realized 1h forward returns, softmax-reweights the strategy ensemble (with exploration floor) |
| Storage / Audit | `app/db.py` — SQLite: events, trades, equity curve, scored signals |
| State Persistence | `app/persistence.py` — full system snapshot to `state.json` (~1/min, atomic): paper account & open positions, neural-net weights + replay buffer, Q-table, bandit posteriors, GA champion, discovered universe, research corpus + source-discovery memory (promotions/demotions/track records). Auto-restored on startup |
| Macro Context | Macro feeds (Fed/rates, inflation, crypto regulation via Google News) scored with a risk-on/risk-off lexicon; macro sentiment is 20% of market sentiment and feeds the ML model via sentiment features |
| Dashboard | `static/index.html` — live equity curve, signals, positions, weights, research feed, audit log, backtest runner |

## Quick start

```bash
git clone https://github.com/h3cker10632/CryptoMind.git && cd CryptoMind
python3 -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. Then, **once**, unlock the paper account — a fresh
install opens no trades (crypto or Polymarket) until you do:

```bash
curl -X POST http://127.0.0.1:8000/api/portfolio/migration/confirm
# -> {"ok": true, "ready": true, "cash": 100000.0, ...}
```

> A rebuilt version — research loop with a champion strategy, core and
> exploration sleeves, a versioned data store and research tools — lives on
> branch [`rebuild-engine-exploration-ui`](https://github.com/h3cker10632/CryptoMind/tree/rebuild-engine-exploration-ui).
> This README describes `main`.

## Install

**Python 3.10 or newer** (the code uses `X | None` annotations at runtime).
Docker and CI use 3.12; 3.13 also works. Run everything from the repository
root — launched from elsewhere, `uvicorn` fails with
`ModuleNotFoundError: No module named 'app'`. All data files are written next
to the code.

`pip install -r requirements.txt` installs: fastapi, uvicorn[standard], httpx,
numpy (required); websockets (real-time prices — without it: REST polling every
30 s, event `websockets package not installed — real-time stream disabled,
REST polling only`); Pillow (Telegram `/chart` image only); numba (faster
backtests — a bit-identical pure-Python kernel is used without it). For tests:
`pip install pytest`.

Optional, not installed: `transformers` + `torch` (set
`CRYPTOMIND_SENTIMENT_MODEL`, e.g. `ElKulako/cryptobert`), `joblib` / `pandas` /
`pyarrow` / the external `crypto_ml` lab (model advisor and ML autotrainer —
without them those features record why and stay idle), `crawl4ai` (the crawler
falls back to plain HTTP), `py-clob-client` (live Polymarket — not enabled).

**Network access.** Only public, key-free endpoints. It needs outbound HTTPS to
`api.exchange.coinbase.com` and `wss://ws-feed.exchange.coinbase.com` (prices —
required), plus `www.okx.com`, `api.coingecko.com`, `api.alternative.me`,
`nfs.faireconomy.media`, Google News / Reddit RSS, `gamma-api.polymarket.com`
/ `clob.polymarket.com`; and `api.telegram.org` /
`generativelanguage.googleapis.com` only if you use Telegram / the LLM advisor.
Without Coinbase the bot holds still (see [Troubleshooting](#troubleshooting));
other feeds only switch their own feature off.

## Start, stop, restart

Pick **one** way to run it, and run **one** instance per directory — a second
copy started in the same folder shares (and on exit overwrites) the first
one's `state.json` and database, even if it fails to bind the port.

| How | Command | Notes |
|---|---|---|
| Plain | `uvicorn app.main:app --host 127.0.0.1 --port 8000` | `--host 0.0.0.0` makes it reachable from your network (then read [Remote access](#remote-access-and-the-api-token)). Ctrl+C saves state and exits. |
| Supervisor (Linux/macOS) | `bash run.sh` | Loops `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000` and relaunches 2 s after any exit, so code changes are picked up on restart. **Ctrl+C restarts it** — press Ctrl+C twice within 2 s, or use the dashboard's *Kill server*. Port and host are fixed in the script. |
| Supervisor (Windows) | `.\run.ps1` (PowerShell) | Same loop. If scripts are blocked: `powershell -ExecutionPolicy Bypass -File .\run.ps1`. |
| systemd (Linux server) | `deploy/cryptomind.service` | Expects the code in `/opt/cryptomind` with a venv in `/opt/cryptomind/.venv`, user `cryptomind`, optional env file `/etc/cryptomind.env` (chmod 600). Install: `sudo cp deploy/cryptomind.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now cryptomind`. Logs: `journalctl -u cryptomind -f`. The unit can only write inside `/opt/cryptomind`. |
| Docker | `export CRYPTOMIND_API_TOKEN=$(openssl rand -base64 32)` then `docker compose up -d` | Port bound to `127.0.0.1:8000` only; the token is required for every control action (see [Remote access](#remote-access-and-the-api-token)). Read [Docker notes](#docker-notes) before upgrading. |

**Dashboard buttons.** *Restart server* saves state and relaunches with the
current code: under `run.sh` / `run.ps1` / systemd the supervisor restarts it;
started plainly it spawns `relaunch.py`, which waits for the old process and
the port and then starts a detached successor (log: `relaunch.log`; its console
output is discarded, and it binds `0.0.0.0` unless you had passed `--host`).
*Kill server* pauses trading, **sells every position it can price** (setting
`flatten_on_shutdown`, default on), saves, writes `.shutdown` and exits;
`run.sh` / `run.ps1` then stop for real, but systemd and Docker
(`Restart=always` / `restart: always`) start it again.

**State survives restarts.** A normal stop (Ctrl+C, SIGTERM, *Restart*,
`systemctl stop`) saves `state.json`; it is also saved about once a minute while
prices flow, or on demand with `POST /api/control/save`. Open positions are
kept across a plain stop.

## First run: what to expect

1. **Paper account: $100,000**; the Polymarket book starts at $10,000. No API
   keys are needed.
2. **Confirm the ledger migration — required once.** The event log shows
   `Shared paper portfolio migration PENDING — POST
   /api/portfolio/migration/confirm …`, `/api/status` shows
   `"shared_ledger_ready": false`. Until you confirm, **no new entries open**
   (crypto or Polymarket), and the decision log doesn't say why. There is no
   dashboard button: `curl -X POST http://127.0.0.1:8000/api/portfolio/migration/confirm`
   (from the same machine; add the token header from anywhere else).
3. **About 60 seconds of warm-up.** The first events are
   `⏳ Launch preflight incomplete … new entries held until healthy` (logged
   twice) and `🟡 SAFE MODE engaged …` — normal until the price feed has
   delivered enough candles. `🟢 SAFE MODE cleared` follows about 2 minutes
   after prices arrive. (`preflight_ok` in `/api/status` stays false after
   that; safe mode is what actually gates entries.)
4. **Polymarket waits for a start**: its engine loop does not start by itself —
   press ▶ start engine on the 🎲 Polymarket tab, or
   `POST /api/polymarket/start`, **after every restart**.
5. **What runs by default:** the hourly signal bot with shorts, pair hedging
   and the meme sleeve on; the LLM advisor is enabled but does nothing without
   a key; the strategy researcher (every 6 h), GA evolution (every 20 min), ML
   autotrainer and crawler are on but idle without their optional packages;
   an auto-export report is written to `reports/` every 5 minutes.
6. **Files it creates** (all in the repository folder): `cryptomind.db` (+
   `-wal`, `-shm`; trades, equity curve, event log), `state.json` (account and
   learned state), `api_token.txt`, `alerts.json`, `settings.json` /
   `tunables.json` / `.secrets.json` (after your first change), `reports/`
   (auto-exports, newest 288 kept), `.cache/history/` (backtest price cache),
   `relaunch.log`.

## Configure

**Settings** (`GET/POST /api/settings`, also the dashboard's ⚙️ Settings; saved
in `settings.json`, secrets in `.secrets.json`, shown masked). Booleans may be JSON
`true` / `false` or the strings `"true"` / `"false"` / `"on"` / `"off"`. Numbers
are clamped to their range; unknown keys and invalid choices are ignored
silently.

| Setting | Default | What it does |
|---|---|---|
| `trade_mode` | `auto` | Stance `passive` / `auto` / `aggressive` (confidence gate 0.55 / 0.45 / 0.35). |
| `allow_shorts` / `hedge_enabled` | true / true | Directional shorts; market-neutral pair hedge. |
| `meme_trading_enabled` | true | Meme sleeve (DOGE, SHIB, PEPE, …) under a tighter risk envelope. |
| `exit_advisor_enabled`, `pattern_exit_enabled`, `exit_throttle_enabled` | true | Early loss-cut and pattern exits. |
| `llm_advisor_enabled` | true | One LLM vote; inert without a key. |
| `polymarket_enabled`, `pm_auto_trade` | true, true | Polymarket may bet (engine still needs ▶ start). |
| `researcher_enabled`, `ml_autotrain_enabled`, `crawl4ai_producer_enabled`, `model_advisor_enabled` | true | Background learners (need their optional packages). |
| `flatten_on_shutdown` | true | *Kill server* sells positions first. |
| `carry_equity` | true | false = every restart starts a fresh $100k (learned state kept). |
| `auto_export_enabled` / `auto_export_interval_sec` | true / 300 | Full-state report into `reports/`. |

**Tunables** (`GET/POST /api/tunables`, sliders in Settings, saved in
`tunables.json`; `POST /api/tunables/reset` restores defaults). The two to know:
`max_drawdown_kill` 0.15 (kill switch) and `daily_loss_limit` 0.05 (daily
halt). Others: `risk_per_trade` 0.0075, `max_open_positions` 4,
`max_gross_exposure` 0.60, `stop_atr_mult` 2.0, `take_profit_atr_mult` 3.0,
`trail_atr_mult` 2.5, `cooldown_sec` 900, `max_entries_per_hour` 12,
`min_confidence` 0.45, `fee_rate` 0.005, `slippage_bps` 10, `explore_prob`
0.12, and the `pm_*` / `meme_*` / `hedge_*` groups.

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `CRYPTOMIND_API_TOKEN` | contents of `api_token.txt` | Control-API token. |
| `CRYPTOMIND_ALLOW_LOOPBACK` | `1` | `1` = requests from the same machine need no token; `0` = always require it (set by `docker-compose.yml`). |
| `CRYPTOMIND_SUPERVISED` | unset | Set by `run.sh`, `run.ps1` and the systemd unit: *Restart* just exits and lets the supervisor relaunch. |
| `CRYPTOMIND_LLM_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` | unset | LLM key (first one set wins; beats the Settings key and `llm_key.txt`). |
| `CRYPTOMIND_LLM_BASE`, `CRYPTOMIND_LLM_MODEL` | Gemini OpenAI-compatible endpoint, `gemini-2.5-flash` | LLM endpoint and model. |
| `CRYPTOMIND_MODEL_DIR` | `model_artifact/` | Model-advisor artifact folder (`model.joblib`, `metadata.json`). |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `ALERT_WEBHOOK_URL` | unset | Seed the alert channels on first start (afterwards `alerts.json` wins; change them with `POST /api/alerts/config` or the 🔔 panel). |
| `CRYPTOMIND_SENTIMENT_MODEL` | unset | Transformer sentiment model (needs `transformers` + `torch`). |
| `CRYPTOMIND_VALIDATION_SEED` | 1337 | Seed for validation runs (negative = random). |

### Remote access and the API token

Reading (every `GET`) is open. **Every** state-changing call (`POST` / `PUT` /
`DELETE` / `PATCH` under `/api/`) needs the token unless the request comes from
the same machine (and `CRYPTOMIND_ALLOW_LOOPBACK` is not `0`). That includes
`/api/ingest/push`: an external signal producer must send the token header.

- The token is `CRYPTOMIND_API_TOKEN`, or else the text in `api_token.txt`
  (created on first start). Send it as `Authorization: Bearer <token>`,
  `X-API-Token: <token>` or `?token=<token>`:
  `curl -X POST -H "Authorization: Bearer $(cat api_token.txt)" http://HOST:8000/api/control/pause`
- **Dashboard from another machine:** open `http://HOST:8000/?token=<token>`
  once — the page stores it in the browser (it also asks for it on the first
  401). Or tunnel: `ssh -L 8000:127.0.0.1:8000 you@host`, then use
  http://localhost:8000.
- **Behind a reverse proxy on the same machine,** set
  `CRYPTOMIND_ALLOW_LOOPBACK=0` unless the proxy sends `X-Forwarded-For`:
  otherwise proxied requests look local and skip the token.
- `GET /api/security` shows whether your request would be authorized.
- Don't expose the port to an untrusted network; front it with a TLS reverse
  proxy.

## Operate

**Dashboard tabs:** Dashboard (equity, weights, learning stack, universe,
derivatives, signals, positions, hedges, trades, tearsheet, research feed,
audit log, alerts, shadow execution, backtests), ML Live, LLM Advisor,
Polymarket (start / stop / one cycle / reset). The header has Pause, Resume,
Kill switch, Reset kill, Settings, Restart server and Kill server.

| Action | Dashboard | API / Telegram |
|---|---|---|
| Pause / resume new entries | Pause / Resume (survives restarts) | `POST /api/control/pause`, `/resume`; Telegram `/pause`, `/resume` (not kept across restarts) |
| Kill switch (stop + sell positions) | Kill switch | `POST /api/control/kill`; `/kill` |
| Clear kill switch and daily halt | Reset kill | `POST /api/control/reset-kill` **then** `POST /api/control/resume`; `/resetkill` |
| Fresh $100k (learned state kept; refuses while positions are open) | — | `POST /api/control/reset-account` |
| Reset the Polymarket book ($10k) | Polymarket tab | `POST /api/polymarket/reset` |
| Close one position / all | — | Telegram `/close BTC`, `/close all` |
| Run the GA / researcher now | Learning panel | `POST /api/control/evolve`, `POST /api/research/run` |

**Tools** on this branch: `python -m tools.measure_scaleout` (profit-ladder
A/B), `python -m tools.crawl4ai_signal.run [--loop N]` (crawler, also run by
the server every 15 min), `tools/invo_signal/` (`poller`, `run_study`,
`make_synthetic`; configured with the `INVO_*` variables).

**Backups.** `bash backup_data.sh` copies the database (safely, while
running), `state.json`, settings, tunables, alerts and the token into
`backups/<timestamp>/`. It does not copy `.secrets.json`, `llm_key.txt`,
`reports/`, `discovered_strategies.json` or `model_artifact/` — copy those
yourself if you need them. On Windows, copy the files by hand. Restoring = stop
the app and copy the files back.

**Upgrade.** `git pull && pip install -r requirements.txt`, then *Restart
server* (or `systemctl restart cryptomind`). For Docker see below.

**Tests.** `pip install pytest && pytest -q` (about 620 tests). The tests
redirect the database and settings files to a temp folder.

### Docker notes

- The volume holds the whole `/app` folder (state files live next to the code).
  The image keeps its code in `/src` and copies it over `/app` on every start,
  so `git pull && docker compose up -d --build` upgrades the code and keeps
  your state. Plain `docker run` gets an anonymous volume at `/app`; give it a
  name (`-v cryptomind-data:/app`) to keep state when the container is removed.
- Runtime data and secrets (`state.json`, the database, `.secrets.json`,
  `llm_key.txt`, `reports/`, `.cache/`, …) are excluded from the image by
  `.dockerignore`.
- No token in the environment? `docker compose exec cryptomind cat /app/api_token.txt`.
- The container health check only tests that the web server answers; it stays
  green while the price feed is down.

## Troubleshooting

Start with three places: the **Audit log** on the dashboard (or
`GET /api/events`, newest 100), `GET /api/status`, and
`GET /api/decisions?action=skip` (why each candidate trade was skipped).
Application messages go to the database, not the terminal — the terminal only
shows web requests and crashes. For more history:

```bash
sqlite3 cryptomind.db "SELECT datetime(ts,'unixepoch'),kind,message FROM events WHERE kind IN ('error','warn','risk') ORDER BY ts DESC LIMIT 200"
```

(The events table is never pruned on this branch; it grows with the database.)

### It doesn't start

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | Started outside the repository folder. | `cd` into the repo first (or `uvicorn --app-dir /path/to/CryptoMind app.main:app`). |
| `ModuleNotFoundError` for fastapi / numpy / … | Dependencies not installed in this Python. | Activate the venv; `pip install -r requirements.txt`. |
| `SyntaxError` / `TypeError: unsupported operand type(s) for \|` | Python older than 3.10. | Use Python 3.10+ (3.12 recommended). |
| `[Errno 98] … address already in use` (Windows: `[WinError 10048]`) | Port taken — often a CryptoMind already running. | Stop the other instance or use `--port 8001`. A failed second start still overwrites `state.json` on exit; never run two in one folder. |
| *Restart server* and it never comes back | The successor couldn't bind the port or crashed. | Read `relaunch.log`; start it yourself: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`. |
| `run.sh` won't stop | Ctrl+C makes uvicorn exit cleanly, which the supervisor restarts. | Ctrl+C twice within 2 s, or *Kill server*. |
| systemd: `status=203/EXEC` or permission errors | Code not in `/opt/cryptomind`, no `.venv` there, or the folder isn't owned by user `cryptomind`. | Match the paths in `deploy/cryptomind.service` or edit it; `sudo chown -R cryptomind: /opt/cryptomind`. |

### It runs but nothing trades

Work down this list — the first match is usually it:

1. **Ledger migration not confirmed** (fresh install): `/api/status` shows
   `"shared_ledger_ready": false`, and no decision row explains it. Fix:
   `POST /api/portfolio/migration/confirm` (see [First run](#first-run-what-to-expect)).
2. **No price data**: the header's data badge is red, `/api/status` has
   `"data_healthy": false` and the log repeats `Market data error: …` every
   30 s. Check that the machine can reach `api.exchange.coinbase.com`
   (`curl -sI https://api.exchange.coinbase.com/products/BTC-USD/ticker`), its
   clock, any proxy or firewall. A feed that never came up raises no
   "feed DOWN" alert — only those log lines. A single coin that fails (e.g. a
   `429 Too Many Requests` on a newly discovered coin) is only skipped —
   `Market data: skipping X until it recovers` — and doesn't take the feed down.
3. **Safe mode** (`🟡 SAFE MODE engaged — …`): the feed is unhealthy, prices
   are older than 3 minutes, or the decision loop failed 4 times in a row. It
   clears by itself 2 minutes after the cause goes away.
4. **Paused, killed or halted**: header badge / `/api/status`
   (`kill_switch`, `halted_today`). The dashboard Pause survives restarts —
   press Resume. See [Kill switch](#kill-switch-and-daily-halt).
5. **Gates on each trade** — read `GET /api/decisions?action=skip`: confidence
   below the stance gate, HTF veto (`long vetoed: HTF strongly bearish`),
   `cooldown` (15 min per coin), `max open positions`, `max gross exposure`,
   `macro blackout` (around high-impact US events), `funding extreme`,
   `protection` locks (`🛑 PROTECTION engaged`), notional below $50, a
   take-profit that can't clear 2.5× round-trip costs, the entry cap of 12 per
   hour, fewer than 60 candles for a coin.
6. **Polymarket idle**: engine not started (▶ start engine after each
   restart), `pm_auto_trade` off, or no market clears `pm_min_edge` / the cost
   floor — see `/api/polymarket/status` → decisions.

### Positions are losing and still held

Exits on this branch: ATR stop (2×) and trailing stop (2.5×, tightened to 1.2×
during macro blackouts), take-profit (3×), peak-giveback (35% of the peak gain,
armed after +1%, never into a loss), opposing-signal exit after 5 minutes,
pattern exit and the exit advisor's loss-cut. There is no maximum holding time,
so a position between its stop and its target is held. **Stops only run while
prices flow** — a coin with no price is never sold blind. The automatic kill
switch blocks new entries but does **not** sell; the *Kill switch* button does.
Polymarket bets exit at +15 / −20 probability points, else at resolution.

### Kill switch and daily halt

- **Kill switch** trips when total equity falls `max_drawdown_kill` (15%) from
  its peak since the last reset; alert `KILL SWITCH TRIPPED`. **Daily halt**:
  the day's loss reaches `daily_loss_limit` (5%); alert `DAILY LOSS HALT`; it
  clears at 00:00 UTC by this machine's clock.
- Both block new entries; exits keep running. Both survive restarts.
- To clear: *Reset kill* (dashboard) or Telegram `/resetkill`. Via the API it is
  two calls — `POST /api/control/reset-kill` then `POST /api/control/resume`
  (resume refuses while killed: `Trading is KILLED — reset the kill switch
  first`). Resetting re-arms the 15% limit from the current equity.

### State and files

- **`State load failed (corrupt file?)`** at startup: the app then starts fresh
  and **overwrites `state.json` within about a minute** — copy it aside
  immediately (or stop the app), then restore from `backups/`. There is no
  automatic `.bak`. Cash lives in the database, so it survives.
- **`database is locked`**: usually two instances in one folder. Writes are
  retried, then dropped silently — stop the extra instance.
- A corrupt `settings.json` / `tunables.json` is ignored silently (defaults
  used): check `GET /api/settings`.
- `carry_equity: false` makes every restart a fresh $100k.
- While the price feed is down the minute-by-minute save is skipped; state is
  still saved on a normal stop.

### Background load

`Decision loop STALLED` (critical alert) means no decision for 2 minutes — the
machine is overloaded or blocked; `Decision loop failing` means 5 errors in a
row (traceback in the `Orchestrator tick failed` event). To lighten the load:
raise `evolve_every_sec` (GA, default 1200 s), turn off `researcher_enabled`,
`ml_autotrain_enabled`, `crawl4ai_producer_enabled`, or raise
`auto_export_interval_sec` (default 300 s).

### Alerts don't arrive

- Configure Telegram / webhook in the 🔔 panel or `POST /api/alerts/config`,
  then `POST /api/alerts/test`. Failures log `Telegram send failed: …`.
- Only alerts at or above `push_level` (default `warning`) are pushed; the same
  title is sent at most once per 15 minutes (critical ones always).
- `Market data feed DOWN` fires only when a working feed fails, not when it
  never came up. The startup alert says "State restored" even on a fresh start.

### Dashboard problems

- **Buttons do nothing / `401 unauthorized`**: you are not on the same machine
  (or `CRYPTOMIND_ALLOW_LOOPBACK=0`, as under Docker). Open
  `http://HOST:8000/?token=<token>` once, or enter the token when asked. A
  wrong token is forgotten on the next 401.
- **Can't reach it from another device**: started with `--host 127.0.0.1`
  (default), a firewall, or Docker's `127.0.0.1:8000` binding. Prefer an SSH
  tunnel over opening the port.

## API

- `GET /api/status` — full system snapshot (equity, risk, regime, weights, exit-advisor, memes…)
- `GET /api/signals` · `/api/market` · `/api/derivatives` · `/api/research` · `/api/learning` · `/api/universe` · `/api/decisions` · `/api/shadow`
- `GET /api/trades` · `/api/equity` · `/api/events`
- `GET /api/backtest?product=BTC-USD&strategy=trend|meanrev|breakout`
- `POST /api/control/pause` · `/resume` · `/kill` · `/reset-kill` · `/save` · `/reset-account` · `/evolve`
- `GET /api/portfolio/migration` · `POST /api/portfolio/migration/confirm` — one-time ledger unlock on a fresh install
- `POST /api/polymarket/start` · `/stop` · `/tick` · `/reset` · `GET /api/polymarket/status`
- `GET /api/security` · `GET /api/export` · `GET/POST /api/tunables` · `POST /api/control/restart` · `/shutdown`
- `POST /api/settings` — toggle `allow_shorts`, `hedge_enabled`, `llm_advisor_enabled`, `exit_advisor_enabled`, `meme_trading_enabled`, trade stance…

## The learning intelligence stack

Multiple learning algorithms run concurrently (`app/learn/`), all learning from
realized, after-cost PnL:

1. **Signal scoring** (`loop.py`) — every strategy signal is labeled with its
   realized 1-hour forward return, aligned to direction.
2. **Regime-conditioned Thompson sampling** (`bandit.py`) — each
   (market regime, strategy) arm keeps a Gaussian posterior over aligned
   returns; ensemble weights are Thompson-sampled per regime, so exploration
   vs exploitation is handled by Bayesian uncertainty, not a fixed schedule.
   Weights are EMA-smoothed with a 4% exploration floor.
3. **Online neural-net committee** (`online_model.py`) — an ensemble of three
   independently-seeded 18→16→1 tanh MLPs trained continually (SGD + AdaGrad) on
   live feature snapshots vs 30-min forward returns, with **quantile heads
   (P10/P90)** for an aleatoric band and inter-member disagreement for epistemic
   uncertainty. Continual-learning safeguards: prioritized experience replay
   (4000 samples, error-weighted), online feature standardization (Welford), and
   honest held-out directional-accuracy tracking (predictions recorded *before*
   labels arrive). Once warmed up it joins the ensemble as the `ml` strategy,
   trust-weighted by accuracy, and its combined uncertainty scales position size
   (0.4×–1.0×) so the book bets small when unsure. Weights + replay persist across
   restart.
4. **Drift detection** (`drift.py`) — Population Stability Index over the
   feature stream; on drift (PSI > 0.25) the online model's learning rate is
   boosted ×3 for fast re-adaptation, then decays back.
5. **Q-learning risk controller** (`rl_risk.py`) — tabular RL agent with state
   (trend, vol, drawdown bucket, loss-streak bucket) and actions
   {0.25…1.25}× risk scale. Reward = equity log-return minus a drawdown
   penalty; epsilon-greedy with decay. Its chosen multiplier feeds directly
   into position sizing every tick.
6. **Genetic strategy evolution** (`evolution.py`) — a GA (population 24,
   8 generations, NSGA-II multi-objective selection over {return, −drawdown,
   per-trade Sharpe}) evolves trading-rule genomes against real history in a
   background thread. Promotion requires **purged walk-forward** validation
   (profit in ≥60% of OOS windows + a deflated-Sharpe gate), and up to 3
   validated genomes are averaged as a champion **portfolio** to dilute
   overfit outliers. The backtester is **NumPy-vectorized** (~5× faster,
   numerically identical, parity-tested; pure-Python fallback if NumPy is absent).
7. **Adaptive loss-cut exit advisor** (`exit_advisor.py`) — after the hard stops
   run, forms an expected next-horizon return in the position's own frame (ML
   committee view blended with a learned per-state value) and **cuts a losing
   position early** when it confidently expects the move to keep going against it.
   Learns **hold-vs-fold** per market state from the counterfactual: what price
   actually did next. Never overrides the hard stop/target/liquidation; exempts
   hedge legs. Toggle `exit_advisor_enabled` (default on).
8. **Direction learner** (`direction.py`) — tracks realized net PnL per
   (regime, direction) and nudges the composite toward the side that has actually
   paid (a bounded tie-breaker, never an override), plus a **multi-timeframe veto**
   that blocks entries fighting a strongly-aligned higher-timeframe trend (stops
   shorting into strong uptrends and vice-versa).

Inspect everything live: `GET /api/learning` returns bandit posteriors,
model accuracy, Q-values, PSI per feature, GA generation history, exit-advisor
and direction-learner stats; the dashboard's "Learning intelligence stack" panel
and the `/brains` Telegram command render it all.

## Meme-coin trading (on by default)

A dedicated meme sleeve (`app/data/memes.py`): a **curated seed** of Coinbase-listed
meme majors (DOGE/SHIB/PEPE/BONK/WIF/FLOKI) is eligible immediately, and CoinGecko's
meme-token category is polled so **hot new memes auto-surface** and get tagged. Memes
trade under a **tighter risk envelope** — reduced dollar-risk and position cap
(`meme_risk_factor`), wider ATR stops/targets (`meme_stop_widen`), a concurrent-meme
cap (`meme_max_positions`) and a total meme-exposure cap (`meme_max_exposure`) — so
one meme candle can't run away with the book. Non-meme trading is unaffected. Toggle
`meme_trading_enabled` (dashboard, `/api/settings`); high variance by nature.

## Optional LLM advisor (Gemini, inert without a key)

An advisor (enabled by default, but it does nothing without a key) (`app/learn/llm_advisor.py`) contributes **one** directional vote
that the Thompson bandit weights like any other strategy — never the driver. Defaults
to **Gemini** (`gemini-2.5-flash`); provide a key via `CRYPTOMIND_LLM_KEY` / `GEMINI_API_KEY` /
`OPENAI_API_KEY`, the Settings panel, or `llm_key.txt` (gitignored) — in that order. With no key it is completely inert (no cost,
no effect). Enable via the dashboard toggle, `/api/settings`, or the `/llm on` Telegram
command.

## Extending to live trading (shadow plumbing built; real orders deliberately NOT enabled)

The production execution machinery now exists and runs in **shadow mode** — it
mirrors the paper account through a real OMS against live prices but **never
sends a real order**:

- **OMS** (`app/execution/oms.py`): durable trade intents, client order IDs
  (idempotent retries), an explicit order state machine, partial-fill VWAP
  aggregation, idempotent fills, **fail-closed** on ambiguity, and a
  reconciliation loop that treats the venue as the source of truth.
- **`Venue` interface** is the single seam. `ShadowVenue`/`ShadowBroker`
  (`app/execution/shadow.py`) implement it as a simulation and measure
  **live-vs-paper divergence** (`GET /api/shadow`). A `LiveVenue` against
  CCXT / an exchange SDK would plug in here — keep read-only vs trading keys
  separated, no withdrawal permission, IP-whitelisted — and progress paper →
  shadow → canary → live with human approval gates. It is intentionally
  unimplemented.

Money is normalised with `Decimal` to exchange tick/lot/min-notional
(`app/money.py`), market data has a **WebSocket** stream
(`app/data/ws_market.py`), and the control API now **requires a token** for
any state change (`app/security.py`). Automated trading involves substantial
risk of loss and jurisdiction-specific regulation — treat this as research
infrastructure first. See `CRITIQUE.md` and `CHANGES.md`.

## Security

State-changing control endpoints (`/api/control/*`, settings, tunables, alerts,
portfolio migration) require `Authorization: Bearer <token>` unless called from
the same machine — details, the exceptions, and remote use:
[Remote access and the API token](#remote-access-and-the-api-token). The token
is auto-generated in `api_token.txt` (0600) on first run, or set
`CRYPTOMIND_API_TOKEN`. `docker-compose.yml` sets `CRYPTOMIND_ALLOW_LOOPBACK=0`
to require it everywhere. Front the app with a TLS reverse proxy; never expose
the port directly.

## Validation

`GET /api/backtest/composite?product=BTC-USD` validates the **real ensemble**
(not just the three legacy single-strategy rules) against a **buy-and-hold
benchmark**, with rolling **walk-forward**, **Deflated Sharpe**, **PBO**, and
win-rate / expectancy confidence intervals. Promotion gates: DSR>0.95,
PBO<0.30, walk-forward pass, and beats buy-and-hold. `pytest -q` runs the suite.

## Notes on realism

- **Real retail costs modeled**: 50 bps taker fee + 10 bps slippage per side
  (~1.2% round trip — matches Coinbase Advanced/Kraken base tiers). A
  cost-viability gate rejects any trade whose honest take-profit can't clear
  round-trip cost by >= 2.5x, so structurally unprofitable trades are skipped
  (the target is never quietly widened to the cost floor).
- **Swing-horizon sizing**: stops/targets/trailing are sized off a
  higher-timeframe ATR (default 1h, aggregated from 5m bars via the
  `swing_atr_bars` tunable: 12 = 1h, 3 = 15m, 1 = native 5m) so a trade can
  clear costs at its natural horizon. Widening the horizon — not loosening
  fees — is what makes a signal cost-viable; it's a different strategy, not a
  looser 5m scalp.
- **Market-neutral pair hedge**: longs the laggard / shorts the leader when a
  correlated pair's spread diverges beyond `hedge_z_entry`. Managed as a unit:
  both legs live or neither (a failed second leg unwinds the first), the
  directional signal-flip exit can never close a hedge leg, a pair cost gate
  requires the expected reversion move to clear round-trip cost on all four
  fills (`hedge_cost_multiple`), a re-entry cooldown (`hedge_cooldown_hours`)
  stops churn, the kill switch / daily-loss halt / gross-exposure cap all block
  new hedge risk, and realized PnL is attributed to the `hedge` learning sleeve.
- **Alerting**: Telegram + webhook (Discord/Slack/ntfy) push for kill-switch,
  daily-loss halt, feed outages, loop stalls; configure via dashboard 🔔 or
  `POST /api/alerts/config`; test with `POST /api/alerts/test`.
- **Rich trade notifications**: every position OPEN (side, qty, notional,
  stop/TP, R:R, strategy votes, regime) and CLOSE (entry→exit, PnL $ and %,
  hold time, reason) is pushed to Telegram. Toggle with `push_trades` (dashboard
  button or `/notify on|off` in the bot).
- **Two-way Telegram command bot**: once a bot token + chat id are set, control
  and query the system from your phone. Commands are private to your chat id.
  `/status /positions /trades /pnl /balance /stats /brains /signals /decisions /diag /chart /risk /why`
  `/pause /resume /kill /resetkill /close <product>|all /shorts on|off /mode <stance> /llm on|off`
  `/set <tunable> <value> /get <tunable> /tunables /notify on|off /help`.
  Notably `/resetkill` clears the kill switch + daily halt remotely, and
  `/close BTC` (or `/close all`) flattens a live position from your phone.
- Backtests use intra-bar low/high for stop/target fills (no close-only cheating)
  and report in-sample vs out-of-sample separately.
- Kill switches: 15% max drawdown, 5% daily loss, data-feed health gate.
