# NOFX-inspired upgrades — Tier 1 + Tier 2 (delivered & pushed)

> **Status (current):** merged and pushed to GitHub `main` — HEAD **`4afb09a`**.
> Full suite: **197 passing**. The standalone `.diff`/`.zip` review artifacts
> this doc used to list have been removed from the repo (they don't belong in a
> public source tree); every change lives in git history and can be diffed
> between the commits below.

Two clean commits landed the tiers, then several features shipped on top (see
"After Tier 2").

| Commit | Tier | Summary |
| --- | --- | --- |
| `2ec7559` | Tier 1 | Safety layer independent of the learner |
| `f099ee9` | Tier 2 | Alpha upgrades (multi-TF, capital-flow, LLM arm) |

---

## Tier 1 — safety layer (commit 2ec7559, +11 tests)

1. **Decision audit trail** — new `decisions` SQLite table + `db.log_decision`.
   Every candidate a cycle considers is recorded: composite, confidence,
   ml-confidence, chosen action (`enter`/`explore`/`skip`/`reject`), the reason
   (e.g. the exact gate that blocked it), and notional **before and after** the
   risk cage clamped it. Surfaced at **`GET /api/decisions`** and a new
   **`/decisions`** Telegram command. This is the "no position without a paper
   trail" record — it answers *why nothing traded / why it was sized so small*
   from history alone.
2. **Per-position peak-giveback exit** (`paper.py`) — closes a *winner* that
   hands back too much of its best unrealized gain. Computed on a **price
   basis** (NOFX's live bug was using leverage-multiplied PnL, which silently
   halved the trigger); arms only after a real move (`trail_giveback_arm_pct`);
   works identically for longs and shorts; exempts hedge legs.
3. **Trade throttles** — `min_hold_sec` (a fresh position is exempt from
   signal-flip / giveback churn; its hard stop still applies) and
   `max_entries_per_hour` (anti-overtrading cap).
4. **Safe-mode circuit breaker** (`guardian.py`) — repeated loop failures or
   stale/unhealthy market data **suspend new entries while open risk keeps
   being managed**; auto-clears after a health dwell (anti-flap).
5. **Launch preflight** — one gate verifying feed health, sufficient history,
   model input-dim vs the feature builder, and kill state before the loop is
   trusted to open anything. Runs at loop start; reflected in the guardian
   snapshot (`/api/status`, `/api/decisions`).

New tunables: `min_hold_sec`, `max_entries_per_hour`, `trail_giveback_pct`,
`trail_giveback_arm_pct`, `safe_mode_fail_threshold`, `safe_mode_recover_sec`,
`data_stale_sec`.

## Tier 2 — alpha upgrades (commit f099ee9, +16 tests)

6. **True multi-timeframe context** — 5m bars are aggregated into 15m/1h/4h
   bars; each timeframe's trend + RSI is read and combined into a new
   `mtf_align` feature in [-1,1]. It confirms/damps `strat_trend` (a 5m cross
   fighting the higher-timeframe trend is halved) and is fed to the ML model
   (`N_IN` 17→18; persistence already restarts the model fresh on a dim change).
7. **OI-growth / capital-flow universe source** — coins with fast-expanding
   open interest (OKX perps) become discovery candidates *before* they trend in
   the news, with **multi-source tagging**: a coin surfaced by two independent
   sources (narrative **and** capital flow) gets a stronger standing prior.
   New `oi_growth_threshold` tunable; `sources`/`oi_growth` persisted and shown
   in `/api/universe`.
8. **Optional LLM advisor — ONE bandit-weighted arm, never the driver**
   (`app/learn/llm_advisor.py`). This is the deliberate inverse of NOFX (whose
   *entire* strategy is the LLM): the advisor contributes a single directional
   lean per coin as the `llm` strategy vote, which the Thompson bandit weights
   from realized PnL exactly like every other sleeve. If it adds edge it earns
   weight; if not, the forgetting bandit decays it to zero. **OFF by default**,
   inert with no API key, TTL-cached, defensive JSON parsing, refreshed off the
   hot path, exposed in `/api/learning`. Fully reversible.

Also: **determinism fix** — `TinyMLP` replay sampling now uses a per-net seeded
RNG (removed global-`random` flakiness that the dimension change exposed).

New tunables: `oi_growth_threshold`, `llm_refresh_sec`, `llm_lean_ttl_sec`.
New setting: `llm_advisor_enabled` (default False).

---

### How to try the LLM arm (opt-in)

> **Note:** the LLM advisor now defaults to **Gemini** (`gemini-2.5-flash`),
> migrated after the original Tier-2 delivery.
```
# put a Gemini key in llm_key.txt (gitignored), OR set one of:
export CRYPTOMIND_LLM_KEY=...     # or GEMINI_API_KEY / OPENAI_API_KEY
# optional overrides: CRYPTOMIND_LLM_BASE, CRYPTOMIND_LLM_MODEL
# then enable it:  POST /api/settings {"llm_advisor_enabled": true}
#              or the /llm on Telegram command, or the dashboard toggle
```
With no key it stays completely inert (no cost, no effect).

### Test hygiene
`tests/conftest.py` isolates **both** `tunables.json` and `settings.json`
into a temp dir, so tests never touch operator config. ML-dimension tests were
made `N_IN`-agnostic so future feature additions don't break them.

---

## After Tier 2 (current HEAD `4afb09a`)

Landed on top of the two tiers; all learn from realized PnL and are reported in
`/brains` + the status snapshot.

- **Adaptive loss-cut exit + hold-vs-fold learning** (`app/learn/exit_advisor.py`)
  — predictive early exit for losers the system expects to keep bleeding, learned
  from the counterfactual next-horizon move. Setting `exit_advisor_enabled`
  (default on) + dashboard toggle.
- **Direction learner** (`app/learn/direction.py`) — learned per-regime long/short
  edge nudges the composite toward the side that has paid, plus a multi-timeframe
  veto against fighting a strong higher-TF trend.
- **Meme-coin trading sleeve** (`app/data/memes.py`) — curated seed of Coinbase
  meme majors (DOGE/SHIB/PEPE/BONK/WIF/FLOKI) + CoinGecko meme-category
  auto-discovery, under a tighter risk envelope (reduced size, wider stops,
  concurrent + total meme-exposure caps). Setting `meme_trading_enabled` + toggle.
- **NumPy-vectorized GA backtester** (`app/learn/evolution.py`) — ~5× faster,
  numerically identical (parity-tested), NumPy optional with a pure-Python fallback.
- **ML-training persistence & bandit fixes** — merged in from parallel work on
  `main`; committee weights + replay survive restart.

New settings: `exit_advisor_enabled`, `meme_trading_enabled`.
New tunables: exit (`exit_cut_threshold`, `exit_min_loss_pct`, `exit_ml_weight`,
`exit_horizon_sec`), direction (`direction_bias_gain`, `direction_bias_cap`,
`mtf_veto_align`), meme (`meme_risk_factor`, `meme_stop_widen`,
`meme_max_positions`, `meme_max_exposure`).
New tests: `test_exit_and_direction.py`, `test_memes.py`, `test_evolution_numpy.py`.
