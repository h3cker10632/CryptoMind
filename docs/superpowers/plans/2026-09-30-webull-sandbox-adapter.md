# Webull Sandbox Crypto Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let CryptoMind optionally route its crypto and meme paper decisions to Webull's sandbox Trading API while accounting every fill against the shared local paper portfolio.

**Architecture:** Preserve one imported broker facade and add an explicit `local_paper` versus `webull_sandbox` execution mode. The Webull path uses the official SDK, a sandbox-pinned client, a `Venue` implementation for order lifecycle, and a broker adapter that exposes the crypto broker interface to the orchestrator. Webull sandbox buying power is only an additional venue limit; the local portfolio ledger remains the sole capital source. Coinbase supplies broad discovery/features, and only the active Webull-supported overlap is tradable in this phase.

**Tech Stack:** Python, official `webull-openapi-python-sdk`, existing `app.execution.oms.Venue`/`OMS`, shared `PaperPortfolio`, Webull sandbox HTTP/market-data/trade-event endpoints, pytest with mocked transport.

**Spec:** `docs/superpowers/specs/2026-09-30-unified-paper-portfolio-webull-design.md`  
**Prerequisite plan:** `docs/superpowers/plans/2026-09-30-unified-paper-portfolio.md` (all seven tasks must pass first)

## Global Constraints

- Only `api.sandbox.webull.com` and `events-api.sandbox.webull.com` are permitted; production hosts are rejected in code and tests.
- The default execution mode remains `local_paper`; Webull sandbox routing is opt-in and disabled by default.
- Never send credentials in chat, source, logs, state snapshots, or API responses; keep App Key/Secret in local secret storage/environment only.
- Webull supported crypto assets are a strict routing allowlist; Webull-only symbols remain watch-only until the system has valid price history, features, and risk metadata for them.
- The remote Webull test-account buying power does not increase local portfolio cash/equity; it can only further restrict order size.
- Crypto preview and direct replace are unsupported by Webull. Safe order changes require confirmed cancellation followed by a new unique client order ID; any ambiguous cancellation blocks a replacement.
- Every account mutation must use the shared ledger's idempotent reserve/settlement API.
- Do not route local-paper and Webull-sandbox orders simultaneously for one signal; no automatic fallback after Webull failure.
- Production/live trading and real API endpoints require a separate reviewed design and explicit approval.

---

### Task 1: Sandbox Configuration and Credential-Safe Client

**Files:**
- Modify: `requirements.txt`
- Modify: `app/settings.py`
- Create: `app/execution/webull.py`
- Modify: `app/execution/__init__.py`
- Test: `tests/test_webull_execution.py`

**Interfaces:**
- `WebullSandboxClient.from_local_secrets()` loads Webull credentials locally and constructs the official SDK client with region `us` and endpoint exactly `api.sandbox.webull.com`.
- `WebullSandboxClient.status() -> dict` returns configured/connected/account-readiness booleans and sanitized errors, never credentials or response headers.
- `execution_mode` accepts only `local_paper` or `webull_sandbox`; default is `local_paper`.
- `webull_sandbox_enabled` defaults false and is an additional guard; production URLs are not configurable.

- [ ] **Step 1: Add failing tests** for default local mode, sandbox-disabled status, missing credentials, production-host rejection, and secret masking in settings/status/state serialization.
- [ ] **Step 2: Run `pytest tests/test_webull_execution.py -k config -q`** and confirm the new interfaces do not exist.
- [ ] **Step 3: Verify the official SDK package and compatible version metadata**; add that official package to `requirements.txt` without importing it in local-paper startup when sandbox mode is disabled.
- [ ] **Step 4: Add settings and local secret keys** `webull_app_key` and `webull_app_secret` to the existing secret storage/masking path. Add `execution_mode` to validated string settings and the disabled-by-default sandbox toggle.
- [ ] **Step 5: Implement the sandbox-pinned client** with the SDK's `ApiClient(app_key, app_secret, "us")` and `add_endpoint("us", "api.sandbox.webull.com")`; fail closed if the resolved host is not the exact sandbox host.
- [ ] **Step 6: Rerun the config tests** and assert credential sentinel strings never occur in JSON-serialized status, settings payloads, or persisted app state.

### Task 2: Account, Instruments, and Quote Preflight

**Files:**
- Modify: `app/execution/webull.py`
- Modify: `app/main.py`
- Modify: `app/orchestrator.py`
- Test: `tests/test_webull_execution.py`

**Interfaces:**
- `WebullSandboxClient.account_context() -> dict` returns the selected crypto account ID, available sandbox buying power, and current remote positions/open-order count.
- `WebullSandboxClient.crypto_instruments() -> dict[str, dict]` returns only `US_CRYPTO` instruments with status `OC` (tradable), including IDs, quantity/price increments, and order limits.
- `WebullSandboxClient.snapshots(symbols: list[str]) -> dict[str, dict]` accepts at most 20 instruments per call and records quote timestamps; active scans remain within the existing 14-market cap and the documented one-request-per-second limit.
- Webull sandbox account status is read-only until an explicit order mode is selected; sandbox account state is shown separately from the local shared ledger.

- [ ] **Step 1: Add mocked tests** for account-list selection, crypto-account identification, balance/position retrieval, paginated instrument discovery, tradable-status filtering, snapshot freshness, 20-symbol limit, and Webull-only watch-only behavior.
- [ ] **Step 2: Run `pytest tests/test_webull_execution.py -k preflight -q`** and confirm missing client methods fail.
- [ ] **Step 3: Use documented SDK operations** `account_v2.get_account_list()`, `account_v2.get_account_balance(account_id)`, and `account_v2.get_account_position(account_id)`; require one explicitly selected crypto account rather than guessing among multiple accounts.
- [ ] **Step 4: Fetch instruments** from the official `US_CRYPTO` crypto-instrument profiles endpoint, follow its pagination key, and accept only `OC` rows. Keep `CO` (liquidate-only) and `NT` (non-tradable) out of new entries.
- [ ] **Step 5: Fetch sandbox crypto snapshots** for currently active supported symbols, reject timestamps outside the configured freshness limit, and fail closed when Webull and Coinbase mid-prices diverge beyond the configured basis guard.
- [ ] **Step 6: Add a startup preflight** requiring the dedicated Webull test crypto account to have no untracked positions or open orders; surface any mismatch and keep sandbox orders disabled until reconciled.
- [ ] **Step 7: Rerun the mocked preflight tests**; no test makes an external HTTP or gRPC request.

### Task 3: Webull OMS Venue

**Files:**
- Create: `app/execution/webull.py` (venue/client sections)
- Modify: `app/execution/oms.py` only if a venue capability is required by tests
- Test: `tests/test_webull_execution.py`
- Test: `tests/test_oms.py`

**Interfaces:**
- `WebullSandboxVenue(Venue).place(intent) -> list[dict]` sends one sandbox crypto order with the OMS client order ID and maps any immediately returned executions into `{exec_id, qty, price, fee}`.
- `WebullSandboxVenue.cancel(intent) -> bool` returns true only after the sandbox confirms cancellation; timeout/ambiguous response returns false.
- `WebullSandboxVenue.fetch_open() -> dict` lists sandbox open orders keyed by CryptoMind client order ID.
- `WebullSandboxVenue.fetch_fills(intent) -> list[dict]` queries order detail by client order ID, returns only unseen Webull execution IDs, and never fabricates a fill from an order acknowledgement.
- Sandbox trade-event callbacks enqueue client order IDs only; the reconciliation worker queries order detail and applies fills, so callback retries cannot directly mutate portfolio cash.

- [ ] **Step 1: Add failing OMS contract tests** for sandbox order payloads, stable client IDs, immediate fills, partial fills, duplicate execution IDs, open acknowledgements, confirmed cancel, timeout/unknown state, and unsupported instrument rejection.
- [ ] **Step 2: Run `pytest tests/test_webull_execution.py tests/test_oms.py -k webull -q`** and confirm the adapter is missing.
- [ ] **Step 3: Place sandbox orders** through official SDK `order_v3.place_order(account_id, new_orders)` using only crypto `instrument_type`, `US` market, and documented crypto order types `MARKET`, `LIMIT`, or `STOP_LOSS_LIMIT`; reject unsupported parameters rather than silently coercing them.
- [ ] **Step 4: Implement cancel and reconciliation** using documented cancel-by-`account_id`/`client_order_id`, open-order listing, and order-detail-by-client-order-ID. Use order detail after acknowledgement because Webull warns that list endpoints can lag.
- [ ] **Step 5: Subscribe with the official `TradeEventsClient`** to sandbox order-status events. The callback only queues the order/client ID; a bounded worker queries authoritative order detail, maps stable Webull execution IDs and order states into OMS fills, and deduplicates before shared-cash settlement.
- [ ] **Step 6: Map execution IDs and order states** into OMS statuses. Keep partial/unfilled intents visible and block another same-product entry while one is pending or `UNKNOWN`.
- [ ] **Step 7: Do not implement direct crypto replace or preview calls.** For a requested price/quantity change, cancel first, query detail until canceled is confirmed, and then submit a new intent; if status remains ambiguous, keep the product blocked.
- [ ] **Step 8: Run Webull venue tests and all OMS tests** with mocked SDK responses and fill streams.

### Task 4: Route the Main Crypto/Meme Broker Through One Mode

**Files:**
- Modify: `app/execution/paper.py`
- Modify: `app/execution/webull.py`
- Modify: `app/execution/oms.py`
- Modify: `app/orchestrator.py`
- Modify: `app/main.py`
- Modify: `app/persistence.py`
- Test: `tests/test_webull_execution.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- The stable imported `broker` facade delegates to exactly one backend selected at startup: local `PaperBroker` or `WebullSandboxBroker`.
- `WebullSandboxBroker` implements the crypto broker surface consumed by the orchestrator: `cash`, `positions`, `closed_trades`, `open`, `sell`, `manage`, `equity`, and `exposure`.
- `WebullSandboxBroker.open(...)` returns a filled position only after confirmed execution; accepted-but-unfilled orders remain pending and prevent duplicate same-product entries.
- Both execution modes debit/settle through the already implemented shared `PaperPortfolio`; remote sandbox buying power is checked as a separate upper bound.

- [ ] **Step 1: Add failing routing tests** for local-paper default, sandbox-mode selection, no duplicate order across backends, pending-order suppression, shared-cash reservation, fee/fill settlement exactly once, and no local-paper fallback after sandbox rejection.
- [ ] **Step 2: Run `pytest tests/test_webull_execution.py -k routing -q`** and confirm the broker facade is absent.
- [ ] **Step 3: Introduce a stable `BrokerRouter` object** at `app.execution.paper.broker` so current imports in alerts, universe, persistence, risk stance, hedging, and orchestration continue to reference one delegating object. Bind/restore the selected backend before background loops start.
- [ ] **Step 4: Implement `WebullSandboxBroker`** as an OMS-backed broker adapter. Convert OMS fills into the existing position/trade shape; reserve shared capital before submission; release a reservation on a definitive rejection; retain it for accepted/pending orders; on partial fills settle the filled amount and release only the confirmed canceled remainder; retain pending intents without classifying an accepted order as rejected.
- [ ] **Step 5: Reconcile Webull positions/orders before each trading cycle** and use fresh Webull snapshots for Webull-mode stop management and order reference prices. Keep Coinbase candles/features as research input only for the supported overlap; Webull-only instruments remain non-tradable in this phase.
- [ ] **Step 6: Ensure kill switch, shutdown, manual close, stop, take-profit, signal flip, predictive exit, pattern exit, and hedge paths use the router and never bypass OMS in sandbox mode.** Disable duplicate shadow mirroring while Webull is the selected backend.
- [ ] **Step 7: Persist router mode and local portfolio state only.** Do not serialize App Key/Secret or trust local positions over Webull reconciliation; mode changes require a flat/reconciled crypto book and restart.
- [ ] **Step 8: Run routing tests, portfolio tests, broker-risk tests, hedge tests, and OMS tests.**

### Task 5: Operator Visibility and Sandbox Smoke Test

**Files:**
- Modify: `app/main.py`
- Modify: `app/orchestrator.py`
- Modify: `static/index.html`
- Modify: `README.md`
- Test: `tests/test_webull_execution.py`
- Test: `tests/test_tier1_safety.py`

**Interfaces:**
- `/api/execution/status` reports selected mode, `sandbox` environment, account-ready state, supported/tradable count, pending/unknown intents, and sanitized blocker/error fields.
- The dashboard distinguishes shared local portfolio cash/equity from Webull sandbox buying power and clearly labels test execution.

- [ ] **Step 1: Add failing API tests** for disabled/default mode, sandbox mode status, unsupported account, stale/mismatched balances, pending intents, and secret redaction.
- [ ] **Step 2: Implement the read-only status endpoint** and dashboard label without adding a production toggle or exposing account tokens.
- [ ] **Step 3: Document local setup**: Webull crypto account + approved Trading API test credentials stored in local secret storage, `execution_mode=webull_sandbox`, sandbox enabled, and explicit dedicated-account preflight. Never ask for secrets in chat.
- [ ] **Step 4: Run the entire test suite** and confirm no request targets `api.webull.com`, `events-api.webull.com`, or a production OAuth hostname.
- [ ] **Step 5: Only after the operator has an approved sandbox account, run a human-supervised sandbox smoke test** with a tiny supported crypto order, confirm fill/fee/position reconciliation, cancel a resting test order, close the position, and verify the shared paper ledger returns to the expected balance. Do not run this smoke test without explicit instruction and local credentials.

## Official References

- Trading API: `https://developer.webull.com/apis/docs/trade-api/crypto`
- Test endpoints and test accounts: `https://developer.webull.com/apis/docs/sdk`
- Account list/balance/positions: `https://developer.webull.com/apis/docs/trade-api/account`
- Crypto instrument profiles: `https://developer.webull.com/apis/docs/reference/crypto-instrument-list`
- Crypto snapshots: `https://developer.webull.com/apis/docs/reference/crypto-snapshot`
- Order detail/open/history query: `https://developer.webull.com/apis/docs/reference/order-query`