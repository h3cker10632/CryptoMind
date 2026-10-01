# Unified Paper Portfolio and Webull Sandbox Design

**Status:** Approved  
**Date:** 2026-09-30

## Goal

Make CryptoMind's existing paper portfolio the single source of simulated
capital for crypto, meme coins, and Polymarket. Use Webull's official sandbox
Trading API as the crypto execution venue for instruments Webull supports,
while retaining broad Coinbase market data for discovery and learning. Keep
production/live trading disabled and outside this change.

## Current Constraints

- The crypto paper broker owns cash and positions independently.
- Polymarket owns a separate fixed-$10,000 paper bankroll.
- The shadow OMS account is an execution mirror, not portfolio capital.
- Webull sandbox has its own remote test-account buying power. It cannot share
  literal cash with CryptoMind's local ledger or a Polymarket account.
- Coinbase currently provides a broader product universe than Webull. The live
  product count is dynamic; the recent public endpoint check returned 401
  online USD products, while Webull documents 70+ crypto assets.
- Existing crypto positions are persisted. Existing Polymarket broker positions
  are not included in the shared state snapshot. After a process restart, the
  old PM position state cannot be recovered or automatically verified.

## Recommended Architecture

### 1. CryptoMind Portfolio Ledger

Introduce one persisted paper-capital owner for cash, reserves, account-level
equity, and auditable capital movements. Crypto and Polymarket brokers retain
their instrument-specific position and fill logic, but they no longer own
independent cash balances. Each order must reserve/debit the shared ledger
atomically before it can consume capital; close, resolution, fees, and partial
fills must return or consume capital exactly once.

The dashboard and API report one total account view: shared cash, marked crypto
positions, marked Polymarket positions, total equity, committed exposure, and
per-sleeve attribution. The existing shadow account remains excluded. Account
risk checks use aggregate equity and exposure so the sleeves cannot each spend
the same free cash or independently exceed the shared budget.

### 2. Webull Sandbox Crypto Venue

Add a Webull adapter behind the existing OMS `Venue` boundary, using the
official Python SDK and the documented sandbox endpoint only. It will retrieve
test-account balances, list supported crypto instruments, place and cancel
orders, consume fill events, and reconcile open orders and positions. Webull's
crypto API does not support preview or direct replace; changing an order must
cancel it, confirm cancellation, and only then submit a new client order ID.
The adapter fails closed on unknown order status, timeout, duplicate fill,
stale account state, or unsupported symbols.

The CryptoMind ledger is the canonical paper budget. Before submitting a
crypto order, it checks both shared local buying power and Webull sandbox
buying power; the remote sandbox balance is an additional venue constraint, not
extra portfolio capital. Webull fills and fees are reflected in the shared
portfolio once, keyed by stable client/order/execution IDs. Webull sandbox
assets absent from Coinbase remain eligible only when CryptoMind has valid
market data and risk metadata for them.

The remote Webull test account is not merged with CryptoMind cash. Dashboard
copy must distinguish the unified CryptoMind paper portfolio from remote
Webull sandbox buying power to avoid presenting them as one real account.

### 3. Broad Research, Narrow Execution

Keep Coinbase products and existing public research feeds as discovery and
feature sources. Do not attempt to stream or train on all 401 products in the
decision loop. Preserve the current bounded active-universe cap and rank
candidates using liquidity, freshness, meme/event priority, and data quality.
Only assets present in Webull's refreshed crypto-instrument list may be routed
to Webull; other candidates remain watch-only and can contribute labeled
research only where reliable price/outcome data exists.

Meme status increases research and candidate-selection priority, not the
capital or risk limits. Existing meme size, position-count, and exposure caps
remain unchanged until paper evidence justifies a separately reviewed change.

### 4. Migration and Persistence

On first startup with the new schema, migrate the current crypto paper cash and
positions as the starting shared portfolio without resetting the account or
adding the legacy Polymarket $10,000 seed as new money. The legacy PM seed is
retired. Before cutover, require explicit operator confirmation that legacy PM
positions are empty or settled. Because the old PM positions were not
persisted, the new process cannot verify this itself; until confirmation, keep
the shared account and PM trading disabled. Do not infer that the PM book is
empty from a missing state snapshot. Write a migration marker and opening
ledger event atomically so restart cannot apply the migration twice. Persist
shared ledger events and PM positions for all subsequent starts.

The migration preserves the existing crypto risk kill/halt state and computes
aggregate baselines deliberately; it must not clear an active kill switch or
implicitly re-arm trading.

## Alternatives Considered

1. **Webull-only portfolio:** simpler venue ownership, but excludes Polymarket
   from the shared budget and loses Coinbase's broader discovery surface.
2. **Keep separate local books and add Webull shadow only:** lower integration
   risk, but does not meet the requirement that all paper strategies share one
   budget.
3. **Recommended: shared CryptoMind ledger plus Webull sandbox venue:** gives
   one canonical paper budget across crypto, memes, and Polymarket while
   respecting Webull's separate sandbox account and supported-coin boundary.

## Safety Boundaries

- Only the Webull sandbox endpoint is supported by this phase; production
  endpoint selection is rejected by configuration and tests.
- No production trading, withdrawals, or credential collection in chat.
- Webull credentials remain in local secret storage or environment variables,
  never in source, logs, API responses, state snapshots, or the ledger.
- The existing crypto and meme risk gates are not increased.
- The local ledger is the sole portfolio accounting source; remote sandbox
  test buying power is displayed separately and cannot inflate local equity.
- Live trading requires a new design, user approval, API permission review,
  small-size canary, reconciliation and kill-switch tests, and separately
  approved production credentials.

## Verification Criteria

- Crypto and Polymarket opens contend for the same free cash; concurrent orders
  cannot overspend it.
- Partial fills, fees, exits, resolutions, restarts, and duplicate Webull fill
  events preserve cash/equity exactly once.
- Aggregate equity includes each position once; shadow mirror positions do not
  enter the main account.
- Legacy crypto state migrates once; absence of an explicit legacy PM-book
  confirmation blocks the migration and leaves source state untouched.
- Webull sandbox adapter rejects production hosts, unsupported symbols, and
  ambiguous order outcomes; integration tests use mocked HTTP/events and no
  user credentials.
- Polymarket settlement remains local paper execution and uses the shared
  ledger.
- Existing paper risk limits and all unrelated safety protections remain
  unchanged.
- No test can submit a production Webull order.

## Rollout

1. Implement and test the shared local ledger and migration first, with all
   trading still locally paper-filled.
2. Add the Webull sandbox adapter and validate it with an approved Webull test
   account; keep it sandbox-only and reconcile every order.
3. Observe combined crypto/meme/Polymarket portfolio accounting and execution
   quality in paper mode. Do not change risk limits based on sample count
   alone.
4. Treat any production Webull connection as a separate, reviewed project.