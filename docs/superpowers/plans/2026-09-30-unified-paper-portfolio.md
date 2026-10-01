# Unified Paper Portfolio Implementation Plan

> **Status: COMPLETE** — All 7 tasks implemented and verified (2026-09-30). `tests/test_portfolio.py` (36/36), `tests/test_polymarket.py` (37/37), meme suites all passing. Full repo suite shows only the 10 pre-existing baseline failures (see `/memories/repo/cryptomind-conventions.md`), none introduced by this work. `execution.py` untouched; meme risk caps unchanged.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use one persisted CryptoMind paper-cash balance across crypto, memes, and Polymarket without counting the shadow account or Webull test buying power as portfolio capital.

**Architecture:** A SQLite-backed `PaperPortfolio` owns shared free cash and idempotent cash events. Crypto and Polymarket keep their product-specific positions and fill math but debit/release that one account; aggregate equity marks both books against shared cash. Migration carries forward the restored crypto balance and blocks until the operator confirms legacy Polymarket positions are settled or empty.

**Tech Stack:** Python, SQLite (`app.db`), existing crypto and Polymarket brokers, FastAPI, pytest, atomic `state.json` persistence.

**Spec:** `docs/superpowers/specs/2026-09-30-unified-paper-portfolio-webull-design.md`

## Global Constraints

- Production/live trading remains disabled and is not implemented by this plan.
- Webull sandbox buying power is not CryptoMind portfolio capital.
- Coinbase remains the broad discovery/market-data source; the active universe stays bounded.
- Meme research priority may increase, but existing meme size, position-count, and exposure caps do not increase.
- Preserve crypto positions, risk kill/halt state, and learner state during migration.
- Do not initialize the shared account by adding the retired Polymarket $10,000 seed.
- Require explicit operator confirmation of legacy Polymarket positions before migration; missing state is not proof the old book was empty.
- The shadow execution mirror remains outside shared cash and aggregate equity.

---

### Task 1: Durable Shared Cash Ledger

**Files:**
- Modify: `app/db.py`
- Create: `app/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- `db.initialize_paper_portfolio(opening_cash: float, migration_id: str) -> bool` creates the account once; a repeated migration ID is a no-op.
- `db.paper_portfolio_account() -> dict | None` returns `cash`, `opening_cash`, and the current migration ID.
- `db.reserve_paper_cash(event_id: str, sleeve: str, amount: float, reference: str) -> bool` atomically subtracts positive cash if sufficient and records the event; duplicate IDs return false without a second debit.
- `db.apply_paper_cash_event(event_id: str, sleeve: str, delta: float, reference: str) -> bool` applies one signed settlement/funding/fee delta idempotently.
- `PaperPortfolio.cash`, `.reserve(sleeve, event_id, amount, reference)`, and `.apply(sleeve, event_id, delta, reference)` are the broker-facing API.

- [ ] **Step 1: Add failing SQLite tests** for account creation idempotency, exact opening cash, unique cash events, insufficient-cash rejection, invalid negative reservation rejection, and concurrent reservations that cannot overspend.
- [ ] **Step 2: Run the focused tests** with `pytest tests/test_portfolio.py -k ledger -q`; confirm missing schema/APIs fail.
- [ ] **Step 3: Implement SQLite tables** `paper_portfolio` (singleton account) and `paper_capital_events` (unique event ID, sleeve, signed amount, reference, timestamp). Use a transaction that updates cash only if the account exists and has enough funds for reservations; insert the unique event in that same transaction.
- [ ] **Step 4: Implement `PaperPortfolio`** as a thin typed wrapper over the DB functions. Reject non-finite amounts, zero/negative reservations, empty event IDs, and unknown sleeves; do not keep an independent in-memory cash copy.
- [ ] **Step 5: Rerun the focused ledger tests** and add a concurrent two-sleeve reservation assertion that total successful reservations never exceed opening cash.

### Task 2: Migration Gate and Persistence

**Files:**
- Modify: `app/settings.py`
- Modify: `app/main.py`
- Modify: `app/persistence.py`
- Modify: `app/db.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Startup calls `portfolio.bootstrap(opening_cash=crypto_broker.cash, migration_id="unified-paper-v1", legacy_pm_confirmed=...)` only after loading the old persisted crypto state.
- `POST /api/portfolio/migration/confirm` is authenticated, explicitly records that legacy Polymarket positions are empty or settled, and runs the one-time migration; no secret is involved.
- `persistence.capture()` stores the shared-account identifier/version, not another mutable cash balance. PM positions and closed-trade state are captured/restored with the PM sleeve.
- `PaperBroker.bind_portfolio(portfolio)` and `PMBroker.bind_portfolio(portfolio)` rebind already-imported module globals after state restore; startup must not replace broker objects referenced by other modules.

- [ ] **Step 1: Add failing migration tests** for restored crypto cash being used exactly once, unconfirmed migration leaving trading disabled, confirmed migration not adding `$10,000`, duplicate startup not reapplying opening cash, and existing kill/halt fields remaining unchanged.
- [ ] **Step 2: Run `pytest tests/test_portfolio.py -k migration -q`** and confirm the tests fail before implementation.
- [ ] **Step 3: Add a pending-migration status** that keeps all account trading paused and exposes the reason without disabling read-only dashboard/API startup.
- [ ] **Step 3a: Gate both the crypto orchestrator and PM engine on `PaperPortfolio.ready`** before any order-capable branch; add tests proving no sleeve opens while migration is pending.
- [ ] **Step 4: Implement one-time bootstrap** after `persistence.load()`. Source opening cash from the restored crypto broker balance; write account row, migration marker, and opening event atomically. Never infer old PM flatness from missing state.
- [ ] **Step 5: Add authenticated `POST /api/portfolio/migration/confirm`**; it records explicit operator confirmation, then performs migration only if it has not already run. Repeated calls return the completed status and cannot add cash.
- [ ] **Step 6: Persist/restore PM positions** in the shared atomic state snapshot and add version checks so malformed PM snapshots do not partially mutate broker state.
- [ ] **Step 7: Run focused migration plus existing persistence/API-auth tests**; verify unauthenticated confirmation is rejected and failed migration does not rewrite source state.

### Task 3: Make Crypto Use Shared Cash

**Files:**
- Modify: `app/execution/paper.py`
- Modify: `app/orchestrator.py`
- Modify: `app/strategies/hedge.py`
- Test: `tests/test_portfolio.py`
- Test: `tests/test_broker_risk.py`

**Interfaces:**
- `PaperBroker.cash` becomes a read-through property backed by `PaperPortfolio.cash`; it is not separately assignable after bootstrap.
- `PaperBroker.bind_portfolio(portfolio)` binds the imported global broker after persistence restore and before background loops start.
- A successful crypto open reserves notional plus fee with a stable event ID before publishing the position. A rejected/failed open leaves cash unchanged.
- Close, short cover, and funding use unique settlement event IDs and return/apply deltas once.

- [ ] **Step 1: Add failing tests** that open a long, open a short, close each, accrue funding twice with the same event identity, and verify shared cash moves by exact fills, fees, margin, and funding.
- [ ] **Step 2: Run the targeted broker tests** and confirm the shared-ledger behavior is absent.
- [ ] **Step 3: Add `PaperBroker.bind_portfolio(portfolio)`** for the global broker initialized before startup. Keep standalone brokers usable by existing unit tests with a test account fixture; production startup binds the singleton DB-backed portfolio only after restoring persisted crypto cash.
- [ ] **Step 4: Replace direct crypto cash mutations** in `open`, `sell`, and `_accrue_funding` with reserve/apply events. Only add a position after a successful reserve. Use per-position close IDs and per-funding-interval IDs to prevent double settlement.
- [ ] **Step 5: Update reset behavior** so resetting one strategy sleeve cannot reset shared cash or silently erase another sleeve's positions; any full-account reset requires every sleeve flat and an explicit whole-account reset operation.
- [ ] **Step 6: Run `pytest tests/test_portfolio.py tests/test_broker_risk.py -q`** and the hedge tests.

### Task 4: Make Polymarket Use Shared Cash

**Files:**
- Modify: `app/markets/polymarket/broker.py`
- Modify: `app/markets/polymarket/engine.py`
- Test: `tests/test_polymarket.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- `PMBroker.cash` reads `PaperPortfolio.cash`; `PM_START_CASH` is retired from production initialization.
- PM open reserves stake plus fee before storing a position; exit and authoritative resolution apply proceeds/PnL exactly once.
- PM `equity(price_lookup)` returns a sleeve view for compatibility; it must not be added to crypto equity because both views contain the same shared cash.

- [ ] **Step 1: Add failing cross-sleeve tests** where a crypto reservation reduces PM buying power, a PM stake reduces crypto buying power, and simultaneous crypto/PM opens cannot overspend.
- [ ] **Step 2: Run `pytest tests/test_portfolio.py -k cross_sleeve -q`** and confirm independent cash still allows double spending.
- [ ] **Step 3: Inject the shared portfolio into `PMBroker`** and replace open, early exit, resolution, and reset cash mutations with idempotent ledger events.
- [ ] **Step 4: Ensure resolution uses one event ID per PM position** and that replaying an already-settled position cannot credit the account twice.
- [ ] **Step 5: Run the full `tests/test_polymarket.py` module** and the new cross-sleeve tests.

### Task 5: Aggregate Equity, Risk, and Reporting

**Files:**
- Modify: `app/portfolio.py`
- Modify: `app/orchestrator.py`
- Modify: `app/markets/polymarket/engine.py`
- Modify: `app/main.py`
- Modify: `static/index.html`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- `PaperPortfolio.total_equity(crypto_market, pm_price_lookup) -> float` equals shared cash plus crypto marked position values plus PM marked position values, counted once.
- `PaperPortfolio.total_exposure(crypto_market, pm_price_lookup) -> float` sums both sleeves; shadow mirror is excluded.
- Existing `/api/status` and PM status expose the same canonical total cash/equity plus explicit per-sleeve breakdown.

- [ ] **Step 1: Add failing valuation tests** for long, short, PM outcome shares, mixed sleeves, missing price fallback, and shadow exclusion.
- [ ] **Step 2: Run `pytest tests/test_portfolio.py -k aggregate -q`** and confirm the current two-account/equity math does not satisfy the shared-account contract.
- [ ] **Step 3: Implement aggregate equity/exposure** using shared cash exactly once and each broker's positions without broker-local cash.
- [ ] **Step 4: Feed aggregate equity into the account-level risk manager** and aggregate committed exposure into global exposure checks; preserve meme-specific size, count, exposure gates and kill/halt behavior.
- [ ] **Step 5: Update snapshots, equity logs, PM responses, and dashboard** to label unified local paper capital separately from any Webull sandbox remote buying power.
- [ ] **Step 6: Run API/reporting tests and static checks** for all changed Python and HTML files.

### Task 6: Meme Research and Decision Priority

**Files:**
- Modify: `app/data/research.py`
- Modify: `app/data/universe.py`
- Modify: `app/signals/engine.py`
- Modify: `app/tunables.py`
- Test: `tests/test_memes.py`
- Test: `tests/test_meme_lifecycle.py`

**Interfaces:**
- Meme candidates receive reserved research-queue capacity and a tie-break priority only after existing listing/liquidity/performance eligibility checks.
- `meme_strategy_influence` defaults to a conservative `1.5`; it multiplies only the active `meme` strategy's effective vote weight on classified memes. The learned bandit weight remains the base weight, and non-meme signals are unchanged.
- Existing meme position size, stop/target, max concurrent positions, and total exposure caps are untouched.

- [ ] **Step 1: Add failing tests** proving fresh meme narratives receive reserved research slots, meme ranking cannot bypass listing/liquidity filters, the active meme arm gets the configured relative-weight boost only on memes, and the same signal on non-memes is unchanged.
- [ ] **Step 2: Run `pytest tests/test_memes.py tests/test_meme_lifecycle.py -k priority -q`** and confirm the priority behavior is missing.
- [ ] **Step 3: Reserve two of the existing bounded research tasks for fresh meme/event candidates** when present; keep total queue size, feed cadence, and request limits unchanged.
- [ ] **Step 4: Add a deterministic meme tie-breaker** to eligible universe candidates after effective heat/performance score; do not bypass Coinbase online status, liquidity floor, universe cap, or current held-position retention.
- [ ] **Step 5: Add `meme_strategy_influence` to `app/tunables.py`** and apply it only to the effective `meme` arm for meme products when that arm has a nonzero vote. Do not impose a weight floor and do not change cash sizing/risk gates.
- [ ] **Step 6: Run meme lifecycle/universe/signal tests** and confirm exact non-meme signal parity plus unchanged meme risk caps.

### Task 7: Whole-Account Safety Regression

**Files:**
- Test: `tests/test_portfolio.py`
- Test: existing persistence, crypto risk, paper broker, Polymarket, and API suites

- [ ] **Step 1: Test restart round-trip** after mixed crypto/PM holdings and cash events; assert exact shared cash, aggregate equity, event uniqueness, positions, and active kill/halt state.
- [ ] **Step 2: Test migration refusal** with confirmation false; assert account remains unavailable, no capital event is applied, and trading stays off.
- [ ] **Step 3: Run focused gates:** `pytest tests/test_portfolio.py tests/test_broker_risk.py tests/test_polymarket.py -q`.
- [ ] **Step 4: Run risk/persistence/API regression suites** and the full repository suite; record unrelated baseline failures without broadening scope.
- [ ] **Step 5: Inspect `git diff --check` and safety-sensitive diffs**; verify no change enables live execution, alters meme risk caps, or counts shadow funds.

## Follow-on

The Webull sandbox adapter is specified separately in
`docs/superpowers/plans/2026-09-30-webull-sandbox-adapter.md` and must not start
order routing until Tasks 1–7 establish this shared account contract.