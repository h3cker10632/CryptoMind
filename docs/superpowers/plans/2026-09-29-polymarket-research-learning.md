# Polymarket Research and Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-backed research and durable, position-independent Polymarket forecasts so the paper learner can measure every tracked market at authoritative resolution.

**Architecture:** Keep the Gamma/CLOB client read-only and add a separate bounded RSS evidence cache and research-conditioned advisor. Persist one point-in-time forecast per condition/time-to-resolution bucket in SQLite, score it once at resolution, and report paired forecast-vs-market metrics through existing PM learning/status surfaces. Add PM learner state to the existing atomic JSON snapshot; preserve separate crypto learning state and the existing paper-only risk gates.

**Tech Stack:** Python, FastAPI, SQLite, httpx, Python standard-library XML parsing, pytest, existing static dashboard.

**Spec:** `docs/superpowers/specs/2026-09-29-polymarket-research-learning-design.md`

## Global Constraints

- Track up to 200 active, liquid, binary markets using existing filters.
- Run quote/decision scans every 20 seconds; refresh news on an independent slower cadence.
- Do not treat repeated quote snapshots as independent forecast samples.
- Keep the existing minimum-edge, confidence, liquidity, fractional-Kelly, per-position, and maximum-position gates unchanged.
- Keep on-chain execution disabled; all tests are offline and no real order is submitted.
- Unresolved or ambiguous markets stay unlabeled; outcome learning is exactly once.
- Compare bot and market Brier/log-loss on the exact same resolved forecast set.

---

### Task 1: SQLite Forecast Ledger

**Files:**
- Modify: `app/db.py`
- Test: `tests/test_polymarket.py`

**Interfaces:**
- `db.init()` creates `pm_forecasts` idempotently.
- `db.record_pm_forecast(forecast: dict) -> bool` inserts by unique `(condition_id, ttl_bucket)` and returns false on duplicates.
- `db.pending_pm_forecasts(limit: int = 200) -> list[dict]` returns unresolved rows oldest-first and decodes JSON snapshots.
- `db.label_pm_forecast(forecast_id: int, outcome0_won: int) -> bool` changes an unresolved row once and returns whether it labeled a row.
- `db.pm_forecast_rows(limit: int = 10000) -> list[dict]` returns resolved and unresolved audit rows.

- [ ] **Step 1: Write the failing ledger tests.** Use a temporary SQLite path by monkeypatching `app.db.DB_PATH`; call `db.init()`, insert the same forecast twice, read it back, label twice, and assert one row, decoded snapshots, and only the first label succeeds.
- [ ] **Step 2: Run the focused test and confirm it fails because the PM ledger API/table is missing.**

Run: `pytest tests/test_polymarket.py -k forecast_ledger -q`
Expected: FAIL with the missing table/API, not a setup or import error.

- [ ] **Step 3: Add the idempotent schema, unique index, and synchronous ledger read/write helpers.** Keep JSON serialization at the DB boundary and use an atomic conditional update for exactly-once labels.
- [ ] **Step 4: Run the focused test and confirm duplicate insert and duplicate label behavior pass.**

Run: `pytest tests/test_polymarket.py -k forecast_ledger -q`
Expected: PASS.

---

### Task 2: PM Research Cache and Evidence-Aware Advisor

**Files:**
- Create: `app/markets/polymarket/research.py`
- Modify: `app/markets/polymarket/llm.py`
- Modify: `app/markets/polymarket/signals.py`
- Modify: `tests/test_polymarket.py`

**Interfaces:**
- `PMResearchCache.refresh(markets) -> int` refreshes a bounded number of normalized Google News RSS queries on a slower cadence; it does not run on every engine tick.
- `PMResearchCache.evidence(condition_id) -> list[dict]` returns only fresh, relevant evidence with stable evidence ID, publisher, canonical URL, publication/retrieval timestamps, matched terms, and bounded excerpt.
- `PMResearchCache.stats() -> dict` reports coverage, freshness, and last bounded error without secrets.
- `PMLLMAdvisor.research_lean(market: dict, evidence: list[dict]) -> dict` returns a bounded lean and valid cited evidence IDs; missing/weak evidence or malformed output returns zero lean and no citations.
- Add a `research` strategy arm independent of `llm`; `signals.evaluate(..., research_lean=0.0, research_influence=1.0)` includes it in votes while preserving all other calculations and gates.

- [ ] **Step 1: Write offline tests for normalized-query cache reuse, URL deduplication, relevance/recency rejection, provenance fields, valid-citation filtering, malformed-output abstention, and the independent `research` signal arm.** Inject RSS/XML and advisor responses at the boundary; do not issue network requests.
- [ ] **Step 2: Run the focused tests and confirm failures describe missing research behavior.**

Run: `pytest tests/test_polymarket.py -k 'research_cache or research_advisor or research_signal' -q`
Expected: FAIL because the PM-specific cache, evidence advisor, or strategy arm does not exist.

- [ ] **Step 3: Implement bounded RSS parsing with stdlib XML parsing, canonical-URL deduplication, per-query cache reuse, recency/relevance checks, and strict request/result bounds.** Extend the advisor using the existing shared credential/request plumbing, and accept only citation IDs supplied with the request.
- [ ] **Step 4: Add the research lean as its own signal arm; when research is missing or abstains it contributes exactly zero.** Keep the existing `llm` behavior unchanged.
- [ ] **Step 5: Re-run the focused tests.**

Run: `pytest tests/test_polymarket.py -k 'research_cache or research_advisor or research_signal' -q`
Expected: PASS with no network access.

---

### Task 3: Learner Snapshots and Resolution Metrics

**Files:**
- Modify: `app/markets/polymarket/learner.py`
- Modify: `app/persistence.py`
- Modify: `tests/test_polymarket.py`

**Interfaces:**
- `PMLearner.capture() -> dict` returns a versioned JSON-safe state including bandit arms, online weights/statistics, signal/trade counters, and attributions.
- `PMLearner.restore(state: dict | None) -> bool` restores only compatible versions and feature dimensions.
- `PMLearner.stats(forecasts: list[dict] | None = None) -> dict` retains current fields and may include resolution-derived strategy metrics when rows are provided.
- Shared persistence snapshot includes a `polymarket_learner` member and restores it without affecting the crypto learner.

- [ ] **Step 1: Write tests for capture/restore round-trip, incompatible-version rejection, and preserved fresh state after invalid restore.**
- [ ] **Step 2: Run the focused learner tests and confirm the new methods are absent.**

Run: `pytest tests/test_polymarket.py -k 'learner_capture_restore' -q`
Expected: FAIL because PM learner state cannot yet be captured/restored.

- [ ] **Step 3: Implement versioned learner serialization and connect it to `persistence.capture()` / `persistence.load()` using the existing state schema.**
- [ ] **Step 4: Run the focused learner tests and then existing persistence tests.**

Run: `pytest tests/test_polymarket.py -k 'learner_capture_restore' -q`
Expected: PASS.

---

### Task 4: Engine Forecasting and Exactly-Once Outcome Learning

**Files:**
- Modify: `app/markets/polymarket/engine.py`
- Modify: `app/markets/polymarket/client.py`
- Modify: `app/markets/polymarket/learner.py`
- Modify: `tests/test_polymarket.py`

**Interfaces:**
- Engine computes a deterministic time-to-resolution bucket for each eligible market and records one forecast per `(condition_id, ttl_bucket)` with bot outcome-0 probability, market-implied probability, votes, feature snapshot, and research provenance.
- Resolution checking covers open positions and bounded outstanding forecasts that are plausibly due; a resolver result is accepted only when closed and prices unambiguously identify one binary winner.
- Resolution labels are stored before learner updates; already-labeled rows cannot update bandit or online model again.
- Existing closed-position trade-PnL learning remains distinct from forecast outcome learning.

- [ ] **Step 1: Write an engine test with a skipped market that records one forecast, resolves without an open broker position, updates signal and online learners from the saved feature/vote snapshot, and does not update again on a second tick.** Add an unresolved case that stays unlabeled.
- [ ] **Step 2: Run the targeted engine test and confirm it fails because skipped markets have no persisted forecast.**

Run: `pytest tests/test_polymarket.py -k 'skipped_market_forecast_resolution' -q`
Expected: FAIL on missing forecast/resolution behavior.

- [ ] **Step 3: Integrate research refresh and advisor results with existing market-only LLM and heuristic signals; preserve existing sizing inputs and gates.**
- [ ] **Step 4: Persist eligible point-in-time forecasts and process a bounded due batch of unresolved rows. Use the stored votes/features, label once, and handle no/ambiguous resolution as retryable.
- [ ] **Step 5: Set the shipped PM cadence to 20 seconds and universe cap/default to 200 without changing edge, confidence, size, liquidity, position, or execution gates.**
- [ ] **Step 6: Re-run engine-focused tests plus all existing PM tests.**

Run: `pytest tests/test_polymarket.py -q`
Expected: PASS; existing risk-sizing behavior and paper broker tests remain unchanged.

---

### Task 5: Paired Forecast Reporting and Dashboard

**Files:**
- Modify: `app/markets/polymarket/engine.py`
- Modify: `app/markets/polymarket/learner.py`
- Modify: `app/main.py`
- Modify: `static/index.html`
- Modify: `tests/test_polymarket.py`

**Interfaces:**
- Existing `/api/polymarket/learning` and `/api/polymarket/status` expose forecast counts, unresolved count, research-covered count, abstentions, source freshness/coverage, bot and market Brier/log-loss over identical labeled rows, their difference, and per-strategy resolution measurements.
- Existing trade stats remain separate; endpoint output contains no API keys or credential material.
- Dashboard learning section displays forecast and research coverage plus paired scores without replacing paper-trade performance.

- [ ] **Step 1: Write report tests with a fixed resolved/unresolved ledger set; assert bot and market scores use the same rows and metrics, unresolved count is excluded, research coverage is correct, and no credentials are returned.**
- [ ] **Step 2: Run the focused report test and confirm the fields are absent.**

Run: `pytest tests/test_polymarket.py -k 'forecast_report' -q`
Expected: FAIL because status currently reports only learner counters.

- [ ] **Step 3: Implement Brier/log-loss and per-strategy measurements over the resolved ledger rows, and expose them through the existing endpoints.**
- [ ] **Step 4: Add concise coverage/score fields to the existing Polymarket dashboard learning section.**
- [ ] **Step 5: Run focused API/report tests and static syntax validation.**

Run: `pytest tests/test_polymarket.py -k 'forecast_report' -q`
Expected: PASS.

---

### Task 6: Verification and Safety Regression

**Files:**
- Test: `tests/test_polymarket.py`
- Test: related API and persistence tests

- [ ] **Step 1: Run the complete Polymarket test module.**

Run: `pytest tests/test_polymarket.py -q`
Expected: PASS, offline.

- [ ] **Step 2: Run the full test suite.**

Run: `pytest -q`
Expected: PASS, or report pre-existing failures without broadening scope.

- [ ] **Step 3: Verify the final diff keeps `execution.py` untouched, keeps existing risk gates intact, and adds no real-order path.**

Run: `git diff --check`
Expected: no whitespace errors; inspect the final diff for safety invariants.