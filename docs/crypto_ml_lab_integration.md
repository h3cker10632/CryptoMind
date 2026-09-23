# Integrating `crypto_ml_lab` with CryptoMind — assessment & plan

**Companion repo:** `h3cker10632/MachineLearning` → `crypto_ml_lab/`
**Date:** 2026-09-23
**Status:** assessment only (no code changes yet)

> ⚠️ **Security:** the PAT used to read the companion repo is live and was exposed in
> chat. Revoke it (GitHub → Settings → Developer settings → Fine-grained tokens) and
> use a local git credential helper instead of pasting tokens.

---

## 1. What `crypto_ml_lab` is

A **leakage-aware, cost-aware ML research stack** purpose-built to learn from a trader
bot's JSON exports. Unlike physicsnemo (rejected earlier for domain mismatch), this is a
genuine domain fit — it already speaks our `cryptomind.export.v2` schema.

| Module | Role | Quality notes |
|---|---|---|
| `io.py` | JSON ingest + schema validation | Sorts, de-dupes on `(symbol, ts)`, reports missing-% |
| `cryptomind_import.py` | Convert **our** v2 export → parquet tables | Already targets `cryptomind.export.v2` exactly |
| `analyze_export.py` | Diagnostic report on an export | Ran on our data; produced `cryptomind_export_analysis.md` |
| `features.py` | Causal features + forward labels | Strict shift discipline; separates X from `y_*`; cross-sectional ranks within a timestamp |
| `validation.py` | Purged + embargoed walk-forward | Never random-splits market data |
| `models.py` | HistGradientBoosting multitask baseline | Handles one-class early data; proper metrics (AUC/AP/Brier/MAE) |
| `deep_model.py` | Causal Transformer (return dist + rug head) | Optional torch; heteroscedastic NLL loss |
| `risk.py` | Fail-closed pre-trade risk engine | Vol-scaled sizing, daily-loss/drawdown/liquidity/spread/staleness/rate limits |
| `execution.py` | Adapter interface + paper adapter + audit log | No creds, append-only audit |
| `live_guard.py` | One governed order path | Explicit `I_UNDERSTAND_LIVE_RISK` token gate |
| `monitoring.py` | PSI drift + prediction health | (referenced; small) |

**Verdict:** well-architected and complementary. Keep it as a **separate offline research
lab**, not merged into the live trading loop. The two systems form a loop:

```
CryptoMind (trades + logs)  ──export──▶  crypto_ml_lab (learn + validate offline)
        ▲                                             │
        └──────────── validated model artifact ◀──────┘
```

---

## 2. The blocking problem (confirmed on both sides)

`crypto_ml_lab`'s own `cryptomind_export_analysis.md`, generated from a real CryptoMind
auto-export, concludes:

> *"The export is a state/report snapshot, not a complete historical feature-and-label
> dataset. The signal-score history in this export has no realized `fwd_return` labels,
> so it must not be used as supervised truth."*

**This is correct, and the root cause is on our side.** In `app/export.py` /
`/api/export` the `signal_scores` section is:

```python
out["signal_scores"] = _safe(lambda: db.recent("signal_scores", history_limit))
```

`db.recent()` returns the **most recent** rows — which are almost all still *unscored*.
CryptoMind's scoring loop fills `fwd_return` only after the forward horizon elapses
(`SIGNAL_EVAL_HORIZON_SEC = 3600`, via `db.unscored_signals` → `db.score_signal`). So the
newest N rows are exactly the ones **without** labels. The lab receives features with null
targets and cannot train honestly.

Their analysis also lists the **"Required next learning dataset"**: per-decision causal
features + strategy votes + regime/spread/liquidity/funding/sentiment + action taken +
action gated + realized forward returns at multiple horizons + MAE/MFE + costs + a stable
run/model/config id. We already log **most** of these across `decisions`, `signal_scores`,
and `closed_trades` — they're just not joined and not filtered to labeled rows.

---

## 3. Data-contract gap analysis (what we have vs what the lab wants)

The lab's per-record contract (`io.REQUIRED = {ts, symbol, price}`,
`io.NUMERIC = price, volume, liquidity, buy_volume, sell_volume, unique_buyers,
unique_sellers, holder_top10_pct, bot_signal`) is **DEX/memecoin-shaped**. CryptoMind is a
CEX (Coinbase) spot/margin bot. Mapping:

| Lab field | CryptoMind source | Status |
|---|---|---|
| `ts` | decision/candle timestamp | ✅ have |
| `symbol` | product (BTC-USD…) | ✅ have |
| `price` | `market.price(p)` / candle close | ✅ have |
| `volume` | ticker/candle volume | ✅ have |
| `liquidity` | book depth (`bid_depth+ask_depth`) | ⚠️ approximate |
| `buy_volume`/`sell_volume` | book imbalance (not true taker flow) | ⚠️ proxy only |
| `unique_buyers`/`unique_sellers` | — | ❌ not available (CEX) |
| `holder_top10_pct` | — | ❌ on-chain only, N/A for CEX majors |
| `bot_signal` | composite signal / ml_confidence | ✅ have |

**Takeaway:** on-chain memecoin fields don't apply to our CEX universe — that's fine, they're
optional/nullable in the lab. The valuable, learnable signal we *can* provide is the
**scored-signal + decision-context** join, which is CEX-appropriate and label-complete.

---

## 4. Recommended sequence

### Phase 1 — ML dataset emitter (highest value, low risk) ⭐ ✅ DONE

**Shipped** in `app/ml_export.py` + `GET /api/ml_dataset` + `python -m app.ml_export`.
Emits `cryptomind.ml_dataset.v1`:
- **`events`** — a leakage-free per-bar time series (candle OHLCV + venue; live
  book depth/imbalance/spread + sentiment stamped only on the latest bar, older
  bars left null so nothing is back-filled). Satisfies the lab's
  `io.REQUIRED={ts,symbol,price}` contract and is consumable directly by
  `crypto_ml prepare` (which derives its own causal features + forward labels).
- **`labels`** — only signals with a realized `fwd_return` (`db.labeled_signals()`,
  `scored=1`), joined to nearest-at-or-before decision context (votes, regime,
  composite, confidence, ml_confidence, size pre/post, action, gated reason) and
  the matched trade outcome (pnl, MAE, exit reason, hold time). Carries
  `y_fwd_return` + `y_direction_correct`.

New DB helpers: `db.labeled_signals()`, `db.all_decisions()`. CLI supports
`--split` (writes `events.json` + `labels.json`, the former drop-in for the lab)
and `--compact`. 10 tests in `tests/test_ml_export.py`. No trading-core changes.

Runbook:
```bash
# from the running bot
curl -s "http://localhost:8000/api/ml_dataset?download=false" -o ds.json
# or offline from saved state
python -m app.ml_export outdir/ --split
# then, in crypto_ml_lab:
python -m crypto_ml.cli validate --input outdir/events.json
python -m crypto_ml.cli prepare  --input outdir/events.json --output data/features.parquet
python -m crypto_ml.cli train    --input data/features.parquet --output artifacts/run_001
python -m crypto_ml.cli backtest --input data/features.parquet --model artifacts/run_001/model.joblib
```

### (original Phase 1 rationale, retained)
Add a **training-grade** export to CryptoMind (new function + endpoint, e.g.
`/api/ml_dataset` / `python -m app.ml_export`) that emits **only labeled rows**:

- Pull **scored** signals (`db.strategy_scores(lookback)` already filters `scored=1 AND
  fwd_return IS NOT NULL`) — not `db.recent(...)`.
- Join each to its decision-time context from `decisions` (votes, regime, composite,
  confidence, ml_confidence, size pre/post, action, gated reason) on `(ts, product)`.
- Attach realized forward return(s) + trade outcome (from `closed_trades`: pnl, MAE via
  `mae_price`, exit_reason, hold time).
- Emit newline-delimited JSON in the lab's `{events:[...]}` shape (or a parquet the
  `cryptomind_import` path already understands), with a **stable run/config id**.

This alone unblocks honest training and is self-contained — **no trading-core changes,
stable APIs preserved.** Fits the "own code / incremental patch" rule.

### Phase 2 — model-advisor round-trip (medium, touches the ensemble) ✅ DONE

**Shipped** in `app/learn/model_advisor.py` (`ModelAdvisor`, singleton `advisor`) +
`strat_model` in `app/signals/engine.py`. A validated `crypto_ml_lab` artifact contributes
**one bandit-weighted advisory vote** (the `model` arm) — never the driver. Same governance
as the LLM advisor:

- **Off by default** (`model_advisor_enabled` setting; dashboard toggle "ML-model advisor").
- **Inert** unless a model artifact exists AND its deps import — never blocks/raises into the
  decision loop. `lean()` reads a cache; a background `orchestrator.model_advisor_loop`
  (registered in `main.py`) does inference out of band.
- Leans clamped to [-1, 1], expire after `model_lean_ttl_sec`; decay to 0 on any error.
- **No train/serve skew:** features are built with crypto_ml_lab's OWN
  `features.make_features` + `model_matrix` (columns aligned to the artifact's saved feature
  list from `metadata.json`). If `crypto_ml` isn't importable the advisor stays inert and
  says so in stats — it never guesses with a divergent feature set.

Artifact: `<model_dir>/model.joblib` + `<model_dir>/metadata.json` (default `model_artifact/`,
override `CRYPTOMIND_MODEL_DIR`). Signal shaping mirrors the lab's backtest:
`lean = tanh(E[r_shortest] * model_return_scale)`, forced to 0 when rug/risk ≥
`model_rug_veto`. Four tunables added: `model_refresh_sec`, `model_lean_ttl_sec`,
`model_rug_veto`, `model_return_scale`. Stats surface at
`learner.full_stats()["model_advisor"]` (→ `/api/learning`).

Round-trip verified end-to-end: train a real `MultiTaskBaseline` via `crypto_ml`
prepare/train → advisor loads it via joblib → aligns to the saved 23-feature list →
produces a valid bounded lean; rug-veto and TTL/clamp paths confirmed. 13 tests in
`tests/test_model_advisor.py`. Suite: 350 passed (was 337).

**Promotion bar (unchanged):** only enable after the lab's own gates pass — purged OOS,
calibrated probabilities, fee/slippage stress. Enabling the toggle without a validated
artifact simply does nothing.

### Phase 3 — formalize the contract / optionally vendor
Either keep the lab a separate repo and version the export schema as the API between them,
or vendor it as `research/crypto_ml_lab/` (offline only, excluded from the live import
graph). Document the `export → prepare → train → backtest` runbook.

---

## 5. Governance guardrails (unchanged standing rules)

- **Paper-only** stays the default on both sides; the lab's `live_guard` already enforces an
  explicit approval token — do not wire live CEX credentials.
- The lab's own README promotion checklist (untouched final period, multiple regimes, purged
  validation, calibration, cost stress, drift monitoring, human approval) is the bar before
  any model influences sizing.
- Preserve CryptoMind's stable API surface and quant/learning core; all additions are new,
  optional modules.

---

## 6. One honest caveat about current results

The lab's analysis of our latest export shows the bot is currently **losing** (200 closed
trades, 29 wins, profit factor 0.166, worst loss streak 33). That's paper mode on a
cold-started learner with tiny model history (40 updates, 36.8% dir. accuracy) — expected
for an unseasoned system, and precisely why offline, leakage-safe learning on a proper
dataset (Phase 1) is the right next move rather than trusting the online learner to control
capital. Do **not** interpret the lab as a promise of improvement; it's a measurement
apparatus that will *reject* models that don't beat baselines out-of-sample.
