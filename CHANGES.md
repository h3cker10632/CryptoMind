# CryptoMind — Hardening Changelog

Everything from `CRITIQUE.md` was addressed, **except** the recommendation to
turn shorts off (§2.5): per operator request, **shorts remain ON by default**
(`allow_shorts=True`), and the new live-execution machinery is built in
**shadow mode only — it never trades real money.**

Status legend: ✅ done · 🟡 done, opt-in/partial by design.

---

## Critical blockers (P0)

| Item | What was done |
|---|---|
| §2.1 No live execution layer | ✅ New OMS (`app/execution/oms.py`): durable trade **intents**, **client order IDs** (idempotent retries), explicit **state machine** (created→submitting→open→partial→filled/canceled/rejected/**unknown**), **partial-fill VWAP** aggregation, idempotent fill dedupe, **fail-closed** on timeout/unconfirmed-cancel, and a **reconciliation loop** that treats the venue as truth and **reports drift**. A `Venue` interface is the single live seam; `LiveVenue` is intentionally NOT implemented. |
| §2.1 Shadow trading | ✅ `ShadowVenue` + `ShadowBroker` (`app/execution/shadow.py`) mirror the paper account through the OMS against **live prices** with realistic fees/slippage/partials — **no real money** — and measure **live-vs-paper divergence** (roadmap's key Stage-3 metric), surfaced at `/api/shadow`. |
| §2.2 Unauthenticated control API | ✅ `app/security.py` middleware: every state-changing route (`/api/control/*`, settings, tunables, alert-config/test) needs `Authorization: Bearer <token>` (token in `api_token.txt`, 0600, or `CRYPTOMIND_API_TOKEN`). Loopback allowed for local ops; compose sets `CRYPTOMIND_ALLOW_LOOPBACK=0`. `/api/security` shows status. |
| §2.3 Float money math | ✅ `app/money.py` — `Decimal` price/qty rounding to **tick/lot** size + **min-notional** enforcement, wired into the paper broker and shadow venue at the fill boundary. Per-instrument rules registry. |
| §2.4 30s polling can't manage stops | ✅ `app/data/ws_market.py` — Coinbase **WebSocket** ticker stream with heartbeat, exponential-backoff reconnect, re-subscribe on universe change; REST stays authoritative for candles/book. Status at `/api/market.ws`. |
| §2.5 "spot long-only" vs shorts on | 🟡 Per request, **shorts kept ON**. README wording corrected; shorts now model **funding accrual + liquidation** (see §4.6) so they're no longer unrealistically riskless. |

## High priority (P1)

| Item | What was done |
|---|---|
| §3.1 Backtest only 3 toy rules | ✅ `app/backtest/composite.py` runs the **real ensemble** strategy functions + confidence/weighting + risk sizing + cost gate + ATR stops/targets/trailing (long & short), with a **buy-and-hold benchmark**. `/api/backtest/composite`. |
| §3.2 Weak overfitting defense | ✅ `app/backtest/stats.py` — **Deflated Sharpe (DSR)**, **Probabilistic Sharpe**, **PBO (CSCV)**, Wilson & bootstrap CIs. Composite report emits hard **gates** (DSR>0.95, PBO<0.30, walk-forward pass, beats buy&hold). GA promotion now uses a **DSR + OOS-trade-count** gate. |
| §3.3 ~40 days history | 🟡 Backtester pulls the max Coinbase allows (~950 hourly bars ≈ 40d) and now runs **rolling walk-forward** (positive-OOS fraction + Sharpe retention). Deeper multi-year history needs an external OHLCV source — hook is `fetch_history`. |
| §3.4 Blocking I/O on event loop | ✅ DB writes now go through a **background writer thread**; `persistence.save()` runs in an **executor**; reconciliation + backtests run off the hot path (`asyncio.to_thread`). |
| §3.5 SQLite + global lock | ✅ **WAL** mode + `busy_timeout`, async writer queue, **retention/prune** on the equity table, and an append-only **`order_events`** audit table. |
| §3.6 No tests | ✅ `tests/` with 29 tests (OMS state machine incl. duplicate/out-of-order/unknown/unconfirmed-cancel, money math, DSR/PSR/PBO, long/short PnL, liquidation, auth) + **GitHub Actions CI**. |

## Concrete bugs (all fixed)

1. ✅ Kill/shutdown no longer **fabricate an exit at entry price** when the feed is down — they strand + alert instead of faking a flat PnL.
2. ✅ Daily-loss rollover is now a **UTC calendar day**, not a rolling 24h.
3. ✅ `snapshot()` **unrealized PnL** rewritten correctly for longs **and** shorts.
4. ✅ Doc rot fixed: MLP is **17→16→1** (code + README).
5. ✅ Signal-engine per-product context moved to **thread-local** (reentrant; safe to parallelize).
6. ✅ Shorts now model **funding accrual** (live OKX rate) + **liquidation** on margin exhaustion.
7. ✅ Persistence **snapshot-copies** shared collections + a save lock (no "changed size during iteration").
8. ✅ Backtester reads **live `tv()` tunables** (matches the running system / GA sim).
9. ✅ Secrets: `api_token.txt` 0600 + gitignored; control API protects alert-config.
10. 🟡 Data polling backoff already present for research; market/deriv loops unchanged (documented risk).
11. ✅ `persistence.save()` off the event loop (#3.4).
12. ✅ Equity table **retention/prune** job.
13. 🟡 Regime still BTC-derived (documented; per-asset regime is a larger design change).
14. ✅ **Per-coin liquidity cap** (`liq_cap_pct` tunable) — position ≤ x% of ~24h dollar volume.

## Also added

- ✅ **Sentiment upgrade path**: optional CryptoBERT/FinBERT via `CRYPTOMIND_SENTIMENT_MODEL` (graceful fallback), plus **negation handling** in the lexicon.
- ✅ **Ops**: `Dockerfile` (non-root, healthcheck), `docker-compose.yml` (`restart: always`, localhost bind, secrets via env), `deploy/cryptomind.service` (systemd, hardened).
- ✅ **Dashboard**: shadow-execution & divergence panel, security panel, "Validate REAL ensemble" button with benchmark overlay + stat gates.
- ✅ **Statistical honesty**: win-rate Wilson intervals + bootstrap expectancy CIs on the composite report.

---

### How to run the new bits

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000     # api_token.txt is auto-created
# Dashboard → "Validate REAL ensemble" for DSR/PBO/walk-forward vs buy-and-hold
# /api/shadow shows the shadow OMS + live-vs-paper divergence (no real money)
docker compose up -d                                # always-on, restart:always
pytest -q                                           # 29 tests
```

### Deliberately NOT done
- **No `LiveVenue`.** Real order placement stays unimplemented by design — the OMS/`Venue` seam is ready, but going live requires exchange keys, human approval gates, and the roadmap's staged canary. Shadow mode gives you all the plumbing safely.
