# ⚡ CryptoMind — Evidence-Gated Crypto Paper-Trading System

A paper-trading system that only lets a strategy, a model or an outside data
source drive trades once tested history shows it helps — and keeps measuring
after that. Paper trading only (simulated $100k account); real order routing is
deliberately **not** enabled. No API keys are needed for core operation.

**Safety-first defaults:** kill switch, daily-loss halt, per-sleeve budgets,
stale-data guards ("never trade blind"), a full audit trail, and evidence gates
on every learner.

**Contents:** [Quick start](#quick-start) · [Install](#install) · [Start, stop, restart](#start-stop-restart) · [First run](#first-run-what-to-expect) · [Configure](#configure) · [Operate](#operate) · [Troubleshooting](#troubleshooting) · [API](#api) · [What runs today](#what-runs-today)

## What runs today

| Sleeve | What it does | Code |
|---|---|---|
| **Core** (opt-in: set `core_allocation_pct` > 0) | Holds the research loop's **champion** (default: BTC/ETH trend, 125-day average with a 2% buffer) — the same function the backtest ran, on the versioned data store. Stale data → hold. Its own **tracking monitor** alerts when holdings drift from the champion's targets. | `app/strategies/core.py`, `app/engine/` |
| **Research loop** (daily) | Backtests every candidate (built-in + queued) on the point-in-time universe with real costs, forward-tracks each from the day its config was frozen, and promotes only when it beats the champion in both backtest halves, clears the **deflated Sharpe** over every variant ever tried, and wins a **paired always-valid forward test**. Clear losers are retired early. Prices the champion under other costs and **after tax**. | `app/engine/challengers.py`, `evidence.py`, `costs_tax.py` |
| **Exploration** (opt-in) | Fast strategies on their own NAV-tracked books; benched at −20%; money follows 30-day results **shrunk by how much evidence they carry**. | `app/strategies/exploration.py` |
| **Polymarket** | Prediction-market sleeve on its **own $500 paper bankroll** (`pm_start_cash`), separate from the main account. Its engine starts from the dashboard's 🎲 tab (▶ start engine) after every restart; it then forecasts every tracked market and **bets only once its resolved forecasts beat the market price** (Brier, per market) — raw or via a learned, walk-forward **calibration**. | `app/markets/polymarket/` |
| **Hourly bot** (legacy) | The original hourly signal ensemble and learner stack. Its replay is negative, so `hourly_bot_mode` auto keeps new entries off; learners stay gated by the learner ablation. | `app/signals/`, `app/learn/` |

The kill switch and daily-loss halt measure the main account **excluding the
core's P&L** (the core is built to ride through drawdowns) and stop new entries
in the bot and exploration sleeves. Polymarket runs its own bankroll.

## How it learns (and how fast)

The bottleneck is statistical power, not compute, so the pipeline is built to
get more evidence per day and to reject bad ideas cheaply. Details, settings and
verification for every step: **[docs/IMPROVEMENT_PIPELINE.md](docs/IMPROVEMENT_PIPELINE.md)**.

1. **Data** — versioned candle/funding store with `as_of` loads
   (`app/data/store.py`), plus a point-in-time store for external series with
   history: Fear & Greed, DVOL, stablecoin supply, on-chain activity, and every
   ingested signal (same-second rows in one push stored as their mean)
   (`app/data/series.py`, `tools/series_sync.py`).
2. **Screen** — does a feature predict anything at all? Cross-coin rank IC /
   time-series correlation, Hansen-Hodrick t for the overlapping labels (no
   verdict below 20 non-overlapping observations), both halves, Holm-adjusted
   (`app/engine/screen.py`, `tools/signal_screen.py`).
3. **Model** — pooled walk-forward ML over every liquid coin's history
   (market-relative, vol-scaled, overlap-weighted labels; ridge / boosted trees),
   judged against a momentum baseline before it may become a candidate
   (`app/ml/`, `tools/ml_lab.py`).
4. **Candidate** — frozen configs queued without code changes
   (`tools/candidates.py`, `/api/research/candidates`); each one is a counted trial.
5. **Promote** — research loop gates (above); the promotion rule itself is
   backtested over history (`tools/promotion_backtest.py`).
6. **Trade & watch** — core tracking monitor, per-sleeve limits, scorecard
   (`GET /api/scorecard`) with forward tests, retirements and cost/tax scenarios.

Run every stage once, start to finish: `python tools/run_pipeline.py`
(`--synthetic` for an offline dry run on a synthetic market). Heavy jobs (data
sync, research loop, screen, ML lab, ablation) also run on their own
schedules as separate lower-priority processes (`nice` 10 on Linux, below
normal on Windows); the replay runs in a worker thread. The replay keeps its per-bar signal data on
disk and only computes new bars; the ablation trains its seeds in parallel.

## Module map

| Component | Implementation |
|---|---|
| Orchestrator | `app/orchestrator.py` — decision loop, core / exploration / replay loops, background jobs, scorecard |
| Data | `app/data/store.py` (candles, funding, revisions, point-in-time universe), `app/data/series.py` + `series_sources.py` (external series), `app/data/market.py` (live Coinbase feed), `app/data/ingest.py` (external-signal seam) |
| Portfolio engine | `app/engine/` — panel, causal features, strategies as pure functions, drift simulation, registry, challengers, evidence, screen, costs/taxes, promotion backtest |
| ML | `app/ml/` — pooled dataset, walk-forward models, evaluation, ML candidate strategies; `app/learn/online_model.py` (hourly bot's online net, trusted only on clustered skill) |
| Risk | `app/risk/manager.py` — sizing, kill switch / daily halt (ex-core basis), protections |
| Execution | `app/execution/` — paper broker, OMS + shadow venue, shared costs |
| Polymarket | `app/markets/polymarket/` — engine, forecast ledger, skill gate, calibration |
| Storage | `app/db.py` (SQLite audit), `app/persistence.py` (`state.json` snapshots) |
| Dashboard | `static/index.html` — portfolio-first layout, scorecard |

## Quick start

```bash
git clone https://github.com/h3cker10632/CryptoMind.git && cd CryptoMind
git checkout rebuild-engine-exploration-ui
python3 -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. Then, **once**, unlock the paper account (a fresh
install opens no positions until you do — see [First run](#first-run-what-to-expect)):

```bash
curl -X POST http://127.0.0.1:8000/api/portfolio/migration/confirm
```

Nothing trades on its own until you also turn a sleeve on — see
[Turning the sleeves on](#turning-the-sleeves-on).

## Install

**Python 3.10 or newer** (the code uses `X | None` annotations at runtime).
Docker and CI use 3.12; 3.13 also works. Run everything from the repository
root — launched from elsewhere, `uvicorn` fails with
`ModuleNotFoundError: No module named 'app'`. All data files are written next
to the code, wherever you start it from.

`pip install -r requirements.txt` installs everything needed. What each
package does, and what happens without it:

| Package | Needed for | Without it |
|---|---|---|
| fastapi, uvicorn[standard], httpx, numpy | the app | won't start |
| websockets | real-time price stream | REST polling every 30 s only (event: `websockets package not installed — real-time stream disabled, REST polling only`) |
| scikit-learn | hourly bot's trade filter, ML lab's boosted trees, daily-lab rank model | numpy fallbacks (trade filter → binned logit, `gbt` → ridge/logistic) |
| numba | faster backtest features | bit-identical pure-Python kernel |
| Pillow | Telegram `/chart` image | no chart image |
| pytest (not in requirements) | `pytest -q` | `pip install pytest` |

Optional extras, not installed by default: `transformers` + `torch` (set
`CRYPTOMIND_SENTIMENT_MODEL`, e.g. `ElKulako/cryptobert`), `pandas` /
`joblib` / `crypto_ml` (model advisor, ML autotrainer), `crawl4ai` +
`playwright` (crawler in `tools/crawl4ai_signal/`), `py-clob-client` (live
Polymarket — not enabled), `pyarrow` (two tests skip without it).

**Network access.** The app only reads public, key-free endpoints. It needs
outbound HTTPS to `api.exchange.coinbase.com` and
`wss://ws-feed.exchange.coinbase.com` (prices — required), plus
`www.okx.com`, `api.coingecko.com`, `api.alternative.me`,
`nfs.faireconomy.media`, `news.google.com` and other RSS feeds,
`api.hyperliquid.xyz`, `www.deribit.com`, `stablecoins.llama.fi`,
`community-api.coinmetrics.io`, `gamma-api.polymarket.com` /
`clob.polymarket.com`; and `api.telegram.org` /
`generativelanguage.googleapis.com` only if you use Telegram / the LLM advisor.
Firewalled hosts degrade only their own feature, except Coinbase: without it
the bot holds still (see [Troubleshooting](#troubleshooting)).

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
| Docker | `export CRYPTOMIND_API_TOKEN=$(openssl rand -base64 32)` then `docker compose up -d` | Port bound to `127.0.0.1:8000` only; the token is required for every control action, even from inside the box. Read [Docker notes](#docker-notes) before upgrading. |

**Dashboard buttons.** *Restart server* saves state and relaunches with the
current code: under `run.sh` / `run.ps1` / systemd the supervisor restarts it;
started plainly it spawns `relaunch.py`, which waits for the old process and
the port and then starts a detached successor (log: `relaunch.log`; its console
output is discarded, and it binds `0.0.0.0` even if you had started on
`127.0.0.1`). *Kill server* pauses trading, **sells every bot position it can
price** (setting `flatten_on_shutdown`, default on), saves, writes `.shutdown`
and exits; `run.sh` / `run.ps1` then stop for real, but systemd and Docker
(`Restart=always` / `restart: always`) start it again.

**State survives restarts.** A normal stop (Ctrl+C, SIGTERM, *Restart*,
`systemctl stop`) saves `state.json`; it is also saved about once a minute while
prices flow, or on demand with `POST /api/control/save`. Open positions are
kept across a plain stop. Core and exploration holdings are never flattened.

## First run: what to expect

1. **Paper account: $100,000**, plus a separate **$500 Polymarket bankroll**
   (`pm_start_cash`). No API keys are needed.
2. **Confirm the ledger migration — required once.** On a fresh install the
   event log shows `Shared paper portfolio migration PENDING — POST
   /api/portfolio/migration/confirm …` and `GET /api/portfolio/migration`
   returns `{"ready": false}`. Until confirmed, the hourly bot opens nothing.
   There is no dashboard button:
   `curl -X POST http://127.0.0.1:8000/api/portfolio/migration/confirm`
   (from the same machine; add the token header from anywhere else).
3. **About 60 seconds of warm-up.** The first event lines are
   `⏳ Launch preflight incomplete … new entries held until healthy` and
   `🟡 SAFE MODE engaged …` — normal until the price feed has delivered enough
   candles. `🟢 SAFE MODE cleared` follows about 2 minutes after prices arrive.
4. **Background schedule.** Loops start staggered: decisions every 20 s from
   10 s; Polymarket split and LLM advisor at 30 s; core hourly from 3 min;
   exploration every 5 min from 3m20s; replay and research jobs checked every
   10 min from 4 min (the first data sync downloads up to 10 years of daily
   candles for every Coinbase USD coin, so the first run of the research tools
   takes a while).
5. **Files it creates** (all in the repository folder): `cryptomind.db` (+ `-wal`,
   `-shm`; trades, equity curve, event log), `state.json` (account and learned
   state), `api_token.txt`, `alerts.json`, `settings.json` / `tunables.json` /
   `.secrets.json` (after your first change), `reports/` (research state,
   exploration books, champion, auto-exports), `.cache/` (price history and the
   data store `.cache/store/`), `relaunch.log`.

### Turning the sleeves on

By default **nothing trades**: the hourly bot is gated off by its own
negative replay, the core and exploration sleeves are off, and the Polymarket
engine waits for a manual start. These settings are not in the dashboard's
Settings panel — use the API (from the same machine no token is needed;
otherwise see [Remote access](#remote-access-and-the-api-token)):

```bash
S=http://127.0.0.1:8000/api/settings
# Core: hold the research loop's champion with 50% of the account
curl -X POST $S -H 'Content-Type: application/json' -d '{"core_allocation_pct": 50}'
# Exploration sleeve (its share: exploration_allocation_pct, default 50)
curl -X POST $S -H 'Content-Type: application/json' -d '{"exploration_enabled": true}'
# Force the hourly bot on despite its replay ("auto" = only if replay is positive in both halves)
curl -X POST $S -H 'Content-Type: application/json' -d '{"hourly_bot_mode": "on"}'
# Polymarket engine: start it (also the "▶ start engine" button) — after EVERY restart
curl -X POST http://127.0.0.1:8000/api/polymarket/start
```

`GET /api/settings` shows every current value. Booleans may be JSON
`true` / `false` or the strings `"true"` / `"false"` / `"on"` / `"off"`.
Numbers are clamped to their allowed range; unknown keys and invalid choices are
silently ignored.

## Configure

**Settings** (`GET/POST /api/settings`, saved in `settings.json`; secrets in
`.secrets.json`, shown masked) — the ones you are most likely to change:

| Setting | Default | What it does |
|---|---|---|
| `core_allocation_pct` | 0 | Share of the account the core holds (0 = off). |
| `core_strategy` | `champion` | `champion` = hold the research loop's champion (default BTC/ETH above their 125-day average with a 2% buffer); `settings` = use the `core_*` settings. |
| `exploration_enabled` / `exploration_allocation_pct` | false / 50 | Exploration sleeve and its share. |
| `exploration_bench_drawdown` | 0.20 | A member that loses this much of the money it was given is sold out and benched (checked daily). |
| `exploration_reinstate_days` | 30 | Days on the bench before it may come back (if its tracked return is positive). |
| `hourly_bot_mode` | `auto` | `auto` / `on` / `off` for the legacy hourly bot. |
| `polymarket_enabled`, `pm_auto_trade` | true, true | Polymarket may bet once its forecasts beat the market (skill gate: ≥ `pm_gate_min_markets` 100 resolved markets). |
| `pm_start_cash` | 500 | Polymarket bankroll, applied on its next reset. |
| `meme_trading_enabled` | false | Meme-coin sleeve. DOGE-USD counts as a meme, so with this off the bot never trades DOGE. |
| `allow_shorts` / `hedge_enabled` | true / false | Directional shorts; market-neutral pair hedge. |
| `trade_mode` | `auto` | Stance: `passive` / `auto` / `aggressive` (confidence gate 0.55 / 0.45 / 0.35). |
| `flatten_on_shutdown` | true | *Kill server* sells bot positions first. |
| `carry_equity` | true | false = every restart starts a fresh $100k (learned state kept). |
| `replay_enabled` | true | Daily strategy replay **and** every background research job (data sync, research loop, screen, ML lab). Turning it off also stops the data sync. |
| `data_sync_interval_sec`, `research_loop_interval_sec` | 86400 | How often the data sync and research loop run. |
| `research_extras_interval_sec` | 604800 | Signal screen, ML lab, promotion backtest. |
| `auto_export_enabled` / `auto_export_interval_sec` | true / 14400 | Periodic full-state report into `reports/` (newest 84 kept). |

**Tunables** (`GET/POST /api/tunables`, saved in `tunables.json`;
`POST /api/tunables/reset` restores defaults; most are sliders in the
dashboard's Settings): risk and costs. The two you should know:
`max_drawdown_kill` 0.15 (kill switch) and `daily_loss_limit` 0.05 (daily
halt). Others: `risk_per_trade`, `max_open_positions` 4, `max_gross_exposure`
0.60, `stop_atr_mult` 3.0, `take_profit_atr_mult` 6.0, `cooldown_sec` 43200,
`max_entries_per_hour` 2, `min_confidence` 0.45, `fee_rate`, `slippage_bps`.

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `CRYPTOMIND_API_TOKEN` | contents of `api_token.txt` | Control-API token. |
| `CRYPTOMIND_ALLOW_LOOPBACK` | `1` | `1` = requests from the same machine need no token; `0` = always require it (set by `docker-compose.yml`). |
| `CRYPTOMIND_SUPERVISED` | unset | Set by `run.sh`, `run.ps1` and the systemd unit: *Restart* just exits and lets the supervisor relaunch. |
| `CRYPTOMIND_LLM_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` | unset | LLM advisor key (first one set wins; beats the Settings key and `llm_key.txt`). |
| `CRYPTOMIND_LLM_BASE`, `CRYPTOMIND_LLM_MODEL` | Gemini OpenAI-compatible endpoint, `gemini-2.5-flash` | LLM endpoint and model. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `ALERT_WEBHOOK_URL` | unset | Seed the alert channels on first start (afterwards `alerts.json` wins; change them with `POST /api/alerts/config` or the 🔔 panel). |
| `CRYPTOMIND_SENTIMENT_MODEL` | unset | Transformer sentiment model (needs `transformers` + `torch`). |
| `CRYPTOMIND_ABLATION_WORKERS` | CPUs − 1 | Processes for the weekly learner ablation. |
| `CRYPTOMIND_EXPERIMENTS` | `reports/experiments` | Experiment registry folder. |
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
  `CRYPTOMIND_ALLOW_LOOPBACK=0`: proxied requests arrive from 127.0.0.1 and
  would otherwise skip the token unless the proxy sends `X-Forwarded-For`.
- `GET /api/security` shows whether your request would be authorized.
- Don't expose the port to an untrusted network; front it with a TLS reverse
  proxy.
- Saved secrets (LLM key, wallet key, tokens) are masked in `GET /api/settings`,
  `GET /api/export`, the auto-export reports and the event log.

## Operate

**Dashboard tabs:** 📊 Dashboard (equity, scorecard with champion vs
candidates, positions, trades, audit log, alerts; the hourly bot and legacy
diagnostics are lower down), 🔬 ML Live, 🧠 LLM Advisor, 🎲 Polymarket (start /
stop / one cycle / reset its book). The header has Pause, Resume, Kill switch,
Reset kill, Settings, Restart server and Kill server.

**Controls:**

| Action | Dashboard | API / Telegram |
|---|---|---|
| Pause / resume new entries (survives restarts) | Pause / Resume | `POST /api/control/pause`, `/resume`; `/pause`, `/resume` |
| Kill switch (stop + sell bot positions) | Kill switch | `POST /api/control/kill`; `/kill` |
| Clear kill switch and daily halt | Reset kill | `POST /api/control/reset-kill` **then** `POST /api/control/resume`; `/resetkill` |
| Fresh $100k (learned state kept; refuses while bot positions are open) | — | `POST /api/control/reset-account` |
| Reset Polymarket bankroll | 🎲 tab | `POST /api/polymarket/reset` |
| Close one position / all | — | Telegram `/close BTC`, `/close all` |

**Research tools** (each also runs on its own schedule as a low-priority
background process; run them by hand to see their output):

| Command | What it does |
|---|---|
| `python tools/run_pipeline.py` | Every stage once: sync → screen → ML → research → promotion backtest. `--synthetic` runs on a throwaway synthetic market (offline); `--skip-sync`, `--stages`, `--json`. |
| `python tools/data_sync.py` | Fill / refresh the data store (`--only daily\|hourly\|funding`, `--quality`). |
| `python tools/series_sync.py` | Fear & Greed, DVOL, stablecoin supply, on-chain series (`--dry-run`). |
| `python tools/research_loop.py` | Backtest, forward-track and maybe promote candidates (`--status` prints the last report). |
| `python tools/candidates.py list\|add NAME '{json}'\|remove NAME` | The candidate queue. |
| `python tools/signal_screen.py`, `tools/ml_lab.py`, `tools/promotion_backtest.py` | Screen features, train/evaluate the pooled ML model, backtest the promotion rule. |
| `python tools/backtest.py`, `tools/replay_backtest.py`, `tools/learner_ablation.py` | Portfolio backtests, hourly-bot replay, learner ablation (~20–30 min). |

**Backups.** `bash backup_data.sh` copies the database (safely, while
running), `state.json`, settings, tunables, alerts and the token into
`backups/<timestamp>/`. It does **not** copy `reports/` (exploration books,
champion, candidate queue, research state), `.cache/store/` (the data store),
`.secrets.json` or `llm_key.txt` — add them yourself:
`tar czf backups/extra-$(date +%F).tgz reports .cache/store .secrets.json llm_key.txt 2>/dev/null`.
Exploration's cash is in the database and its positions in
`reports/exploration_state.json`: back them up together. On Windows, copy the
same files by hand.

**Upgrade.** `git pull && pip install -r requirements.txt`, then *Restart
server* (or `systemctl restart cryptomind`). For Docker see below.

**Tests.** `pip install pytest && pytest -q` (about 840 tests, a few minutes;
set `OMP_NUM_THREADS=1` on small machines). The tests redirect the database
and settings to a temp folder but not `reports/` or `.cache/store/`, so run
them in a separate checkout if this one holds live data.

### Docker notes

- The volume holds the whole `/app` folder (state files live next to the code).
  The image keeps its code in `/src` and copies it over `/app` on every start,
  so `git pull && docker compose up -d --build` upgrades the code and keeps
  your state. Plain `docker run` gets an anonymous volume at `/app`; give it a
  name (`-v cryptomind-data:/app`) to keep state when the container is removed.
- Runtime data and secrets (`state.json`, the database, `.secrets.json`,
  `llm_key.txt`, `reports/`, `.cache/`, …) are excluded from the image by
  `.dockerignore`.
- No token in the environment? Read the generated one:
  `docker compose exec cryptomind cat /app/api_token.txt`.
- The container health check only tests that the web server answers; it stays
  green while the price feed is down.

## Troubleshooting

Start with three places: the **audit log** on the dashboard (or
`GET /api/events`, newest 100), `GET /api/status`, and
`GET /api/decisions?action=skip` (why each candidate trade was skipped).
Application messages go to the database, not the terminal — the terminal only
shows web requests and crashes. For more history:

```bash
sqlite3 cryptomind.db "SELECT datetime(ts,'unixepoch'),kind,message FROM events WHERE kind IN ('error','warn','risk') ORDER BY ts DESC LIMIT 200"
```

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

1. **Ledger migration not confirmed** (fresh install): `GET /api/portfolio/migration`
   shows `"ready": false`. Fix: `POST /api/portfolio/migration/confirm` (see
   [First run](#first-run-what-to-expect)).
2. **Every sleeve is off** — the default. The hourly bot only trades when its
   replay is positive in both halves (`hourly_bot_mode` auto); core and
   exploration are off; Polymarket needs ▶ start after each restart. See
   [Turning the sleeves on](#turning-the-sleeves-on); `GET /api/scorecard` →
   `gates` shows what is active.
3. **No price data**: the header's data badge is red, `/api/status` has
   `"data_healthy": false` and the log repeats `Market data error: …` every
   30 s. Check that the machine can reach `api.exchange.coinbase.com`
   (`curl -sI https://api.exchange.coinbase.com/products/BTC-USD/ticker`), its
   clock, any proxy or firewall. A feed that never came up raises no
   "feed DOWN" alert — only those log lines.
4. **Safe mode** (`🟡 SAFE MODE engaged — …`): the feed is unhealthy, prices
   are older than 3 minutes, or the decision loop failed 4 times in a row. It
   clears by itself 2 minutes after the cause goes away.
5. **Paused, killed or halted**: header badge / `/api/status`
   (`kill_switch`, `halted_today`). Pause survives restarts — press Resume. See
   [Kill switch](#kill-switch-and-daily-halt).
6. **Gates on each trade** — read `GET /api/decisions?action=skip`:
   `chop filter` (market trendiness below 0.12; setting `chop_filter_mode`),
   confidence below the stance gate, `cooldown` (12 h per coin after any
   trade), `max open positions`, `max gross exposure`, `macro blackout`
   (around high-impact US events), `meme trading disabled` (DOGE),
   `funding extreme`, `trade filter`, notional below $50, protections
   (`🛑 PROTECTION engaged`), the entry cap of 2 per hour. Maker entries
   (`entry_order_type` maker) only fill if price trades through them within an
   hour.
7. **Core holds but doesn't trade**: `CORE: no daily data newer than 3 days …
   holding positions, no trades` (or `no saved weights …` for an ML champion)
   — its data is stale; see [Stale data](#stale-data-and-background-jobs).
   `CORE: no live price for X … not traded` — no price for that coin yet.
8. **Polymarket only "would-open"**: its skill gate isn't passed yet (needs 100
   resolved markets where its forecasts beat the market price) — by design.

### Positions are losing and still held

That is usually the design, not a fault — each sleeve has its own exit rule:

- **Core**: no stop-loss. It sells a coin only when the champion drops it
  (default: a daily close more than 2% below its 125-day average), so a
  15–25% pullback inside an uptrend is held. The account kill switch ignores
  the core's P&L on purpose. Want a tighter exit? Queue a faster candidate
  (e.g. `btc_eth_trend_fast`, 75-day) and let the research loop compare it, or
  lower `core_allocation_pct`.
- **Exploration**: each member sells on its own signal; a member is benched
  (sold out) only when it is down `exploration_bench_drawdown` (20%) of the
  money it was given — checked once a day. Lower it (e.g. `0.10`) to cut losers
  sooner.
- **Hourly bot**: ATR stop (3×), trailing stop (3×), take-profit (6×),
  signal-flip exit after 12 h, pattern exit. The loss-cutting exit advisor only
  acts once its learner gate passes.
- **No prices = no exits.** Stops and exits only run while the price feed works;
  a coin with no price is never sold blind.
- **Kill switch** stops new buys; only the *Kill switch button* (manual) sells
  bot positions. Core holdings are never sold by it.

### Kill switch and daily halt

- **Kill switch** trips when the account (excluding the core, and Polymarket)
  falls `max_drawdown_kill` (15%) from its peak since the last reset; alert
  `KILL SWITCH TRIPPED`. **Daily halt**: the day's loss reaches
  `daily_loss_limit` (5%); alert `DAILY LOSS HALT`; it clears at 00:00 UTC by
  this machine's clock.
- Both block new hourly-bot and exploration buys; exits keep running; the core
  and Polymarket are not affected. Both survive restarts.
- To clear: *Reset kill* (dashboard) or Telegram `/resetkill`. Via the API it is
  two calls — `POST /api/control/reset-kill` then `POST /api/control/resume`
  (resume refuses while killed: `Trading is KILLED — reset the kill switch
  first`). Resetting re-arms the 15% limit from the current equity.

### Stale data and background jobs

- The data store is refreshed by `tools/data_sync.py`, run daily by the server.
  Check `GET /api/scorecard` → `data.store_last_sync` / `store_age_hours` and
  the events `Market data sync started` / `finished (exit N)`. A non-zero exit
  is retried after 6 h; run `python tools/data_sync.py` by hand to see the
  error (background tools' output is discarded).
- `replay_enabled: false` also stops the data sync and the research loop.
- The core trades only on data at most 3 days old; exploration members on data
  at most 3 bars old (3 hours for hourly members). Older → they hold, silently
  for exploration.
- A slow machine: background jobs run as low-priority processes (`nice` 10 /
  below-normal on Windows). Throttle them with `research_loop_interval_sec`,
  `research_extras_interval_sec`, `replay_interval_sec`,
  `replay_history_days`, or `CRYPTOMIND_ABLATION_WORKERS`. On a fresh or
  offline install the replay retries every 10 minutes until it has data.
- `Decision loop STALLED` (critical alert): no decision for 2 minutes — the
  machine is overloaded or something blocked the event loop; check CPU.
- `Decision loop failing` (critical): 5 decision errors in a row — the full
  traceback is in the `Orchestrator tick failed` event.

### State and files

- **`State load failed (corrupt file?)`** at startup: the app then starts fresh
  and **overwrites `state.json` within about a minute** — copy it aside
  immediately (or stop the app), then restore from `backups/`. There is no
  automatic `.bak`.
- **`database is locked`**: usually two instances in one folder. Writes are
  retried, then dropped silently — stop the extra instance.
- A corrupt `settings.json` / `tunables.json` is ignored silently (defaults
  used): check `GET /api/settings`.
- `carry_equity: false` makes every restart a fresh $100k — set it back to
  `true` if your account keeps resetting.

### Alerts don't arrive

- Configure Telegram / webhook in the 🔔 panel or `POST /api/alerts/config`,
  then `POST /api/alerts/test`. Failures log `Telegram send failed: …`.
- Only alerts at or above `push_level` (default `warning`) are pushed; the same
  title is sent at most once per 15 minutes. `Core off its champion` is a warning.
- `Market data feed DOWN` fires only when a working feed fails, not when it
  never came up.

### Dashboard problems

- **Buttons do nothing / `401 unauthorized`**: you are not on the same machine
  (or `CRYPTOMIND_ALLOW_LOOPBACK=0`). Open `http://HOST:8000/?token=<token>`
  once, or enter the token when asked. A wrong token is forgotten on the next
  401.
- **Page loads but stays empty**: the server is starting or busy — check
  `GET /api/status`; the browser console shows failed requests.
- **Can't reach it from another device**: started with `--host 127.0.0.1`
  (default), a firewall, or Docker's `127.0.0.1:8000` binding. Prefer an SSH
  tunnel over opening the port.

## API

- `GET /api/status` — full system snapshot (equity, risk, regime, weights, exit-advisor, memes…)
- `GET /api/signals` · `/api/market` · `/api/derivatives` · `/api/research` · `/api/learning` · `/api/universe` · `/api/decisions` · `/api/shadow`
- `GET /api/trades` · `/api/equity` · `/api/events`
- `GET /api/backtest?product=BTC-USD&strategy=trend|meanrev|breakout`
- `POST /api/control/pause` · `/resume` · `/kill` · `/reset-kill` · `/save` · `/reset-account` · `/evolve`
- `GET /api/scorecard` — champion vs candidates (backtest, deflated Sharpe, forward test, retirements), costs / after-tax, live vs holding BTC
- `GET /api/research/candidates` · `POST /api/control/research/candidates` (`{name, config}`) · `POST /api/control/research/candidates/remove`
- `GET /api/portfolio/migration` · `POST /api/portfolio/migration/confirm` — one-time ledger unlock on a fresh install
- `POST /api/polymarket/start` · `/stop` · `/tick` · `/reset` · `GET /api/polymarket/status`
- `GET /api/security` · `GET /api/export` · `GET/POST /api/tunables` · `POST /api/control/restart` · `/shutdown`
- `POST /api/settings` — toggle `allow_shorts`, `hedge_enabled`, `llm_advisor_enabled`, `exit_advisor_enabled`, `meme_trading_enabled`, trade stance…

## Legacy: the hourly bot's learning stack

> These learners serve the hourly bot, which is gated off by its replay; each
> learner is also gated by the learner ablation (`app/learn/gate.py`) and on
> 2026-10-02 none passed. The online net's vote is trusted only on clustered,
> baseline-adjusted skill (`online_model.trust`). Kept for reference.

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
   independently-seeded 30→16→1 tanh MLPs trained continually (SGD + AdaGrad) on
   live feature snapshots vs 24-hour forward returns, with **quantile heads
   (P10/P90)** for an aleatoric band and inter-member disagreement for epistemic
   uncertainty. Continual-learning safeguards: prioritized experience replay
   (4000 samples, error-weighted), online feature standardization (Welford), and
   honest held-out directional-accuracy tracking (predictions recorded *before*
   labels arrive). It votes as the `ml` strategy only once its skill over the
   always-majority baseline is > 2 clustered SE over ≥ 20 daily label windows
   (weight 0 until then), and its combined uncertainty scales position size
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

## Meme-coin trading (opt-in sleeve)

A dedicated meme sleeve (`app/data/memes.py`): a **curated seed** of Coinbase-listed
meme majors (DOGE/SHIB/PEPE/BONK/WIF/FLOKI) is eligible immediately, and CoinGecko's
meme-token category is polled so **hot new memes auto-surface** and get tagged. Memes
trade under a **tighter risk envelope** — reduced dollar-risk and position cap
(`meme_risk_factor`), wider ATR stops/targets (`meme_stop_widen`), a concurrent-meme
cap (`meme_max_positions`) and a total meme-exposure cap (`meme_max_exposure`) — so
one meme candle can't run away with the book. Non-meme trading is unaffected. Toggle
`meme_trading_enabled` (dashboard, `/api/settings`); high variance by nature.

## Optional LLM advisor (Gemini, off by default)

An opt-in advisor (`app/learn/llm_advisor.py`) contributes **one** directional vote
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
the same machine — details, the exceptions, and how to use the dashboard
remotely: [Remote access and the API token](#remote-access-and-the-api-token).
The token is auto-generated in `api_token.txt` (0600) on first run, or set
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
