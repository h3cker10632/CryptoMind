# Polymarket Research and Learning Design

**Status:** Draft for user review  
**Date:** 2026-09-29

## Problem and Current Evidence

The Polymarket engine is enabled, auto-paper trading is enabled, and it is scanning markets. The observed live snapshot had zero bets, zero closed trades, zero scored outcomes, and zero online-model updates. The recent candidates were skipped because their estimated edge was below the configured 3% minimum or there was no edge. The optional LLM had made calls, but its current prompt contains market question, outcomes, prices, time to resolution, and category; it receives no supporting research.

The learning path explains the empty counters. `PolymarketEngine.tick()` calls `PMLearner.score_resolution()` only while settling an open position. It does not retain forecasts for markets the engine considered and skipped. Consequently, no positions means no resolution labels reach either the bandit or online model. The PM learner also has no visible capture/restore integration in `app/persistence.py`, so PM-specific learned state is not durable across process restarts.

## Goals

- Cover up to the top 200 active, liquid, binary markets, using the existing market filters.
- Scan/update quotes every 20 seconds without increasing the frequency of news-source requests.
- Attach time-stamped, market-relevant research evidence to forecasts and make that evidence inspectable.
- Learn from all tracked markets when they resolve, including those the bot did not bet on.
- Preserve conservative paper sizing and all current edge, confidence, liquidity, and position-count gates.
- Keep on-chain execution disabled.

## Non-Goals

- Guarantee profits or treat a research summary as verified truth.
- Lower the minimum edge or confidence threshold to force more bets.
- Place real-money orders or configure a wallet.
- Treat repeated 20-second quote snapshots as independent training samples.
- Make the existing crypto learner consume Polymarket outcomes.

## Proposed Design

### Market and Research Pipeline

The existing Gamma/CLOB client remains the read-only source for market metadata and quotes. Set the PM universe to at most 200 markets, still requiring active binary markets, enabled order books, accepting orders, and the existing minimum-liquidity condition. Use a 20-second engine cadence.

Add a bounded background research cache. For each market, derive a query from its question, category, and named entities, and retrieve keyless Google News RSS results. Fetch/refresh feeds on a slower independent cadence, cache results by normalized query/source, deduplicate by canonical URL, and cap both concurrent requests and retained results. Keep publisher, URL, publication time, retrieval time, matched terms, and a short excerpt. Relevance and recency checks must run before evidence is attached. Research errors, stale documents, and irrelevant results yield no research lean and never stop quote scanning.

The optional evidence-aware PM advisor receives only the market fields and retrieved evidence actually stored for that market. Its output must be bounded, structured, and cite evidence IDs from the supplied list; unsupported citations are discarded. If evidence is absent or too weak, it abstains with a zero lean. Keep this research-conditioned prediction as its own `research` strategy arm, separate from the existing market-only `llm` arm, so resolution results can measure whether research adds value. No generated claim is stored as a source fact.

### Forecast Ledger and Learning

Persist one forecast per `(condition_id, time_to_resolution_bucket)` to SQLite, captured on the first eligible observation after entering that bucket. Store forecast timestamp, the composed predicted probability for outcome 0, market-implied probability, signal votes, feature snapshot, research evidence IDs/metadata, and nullable resolved outcome. A uniqueness constraint prevents the 20-second loop from writing repeated samples for the same market and bucket. A market may receive a new forecast when it enters a new time-to-resolution bucket.

When a forecasted market has an authoritative resolution, label it exactly once, whether or not the PM broker held a position. Update the bandit from the saved strategy votes and update the online model using the saved feature snapshot, not current/reconstructed features. Preserve the existing trade-PnL teacher for positions that were actually opened and closed. Resolution polling must be limited to outstanding forecasts that are plausibly due, bounded in concurrency/rate, and retryable; an unavailable or ambiguous resolution remains unlabeled rather than being treated as a loss.

Add PM learner `capture`/`restore` support for bandit arms, online-model weights/statistics, and counters. Restore only compatible/versioned state. Keep forecast rows in SQLite so pending labels and audit provenance survive restarts independently of the in-memory broker.

Report scored forecast count, unresolved count, research-covered count, abstentions, Brier score/log loss, per-strategy resolution performance, and research evidence freshness/coverage through the existing Polymarket learning/status endpoints. Compute Brier score/log loss for both the bot forecast and the saved market-implied probability on exactly the same resolved forecast set, and report their difference. Learning samples are not proof of profitability. Keep existing trade performance as a separate report.

### Fast-Paced Paper Betting and Safety

Keep the 20-second scan cadence but do not relax the 3% minimum edge, confidence gate, fractional-Kelly sizing, per-position cap, maximum positions, or minimum liquidity. The engine may open only candidates that already pass those gates and the existing duplicate-position check. Evidence refresh does not run on every quote tick. The live execution switch remains off and no code path in this change enables it.

## Failure Handling

- RSS/provider failure: retain only unexpired cached evidence, record a bounded diagnostic, and continue market scanning.
- LLM unavailable or malformed output: research lean is zero; heuristic and market-only signals continue under existing gates.
- Duplicate articles or forecasts: reject by canonical URL and unique forecast key.
- Unresolved/ambiguous outcomes: do not label; retry with bounded backoff.
- Restart: restore compatible PM learner state and continue unresolved forecasts without duplicating updates.
- SQLite migration: create new tables/indexes idempotently through the existing database initialization pattern.

## Verification Criteria

- Offline tests verify query/evidence relevance and expiry, cache reuse, source provenance, invalid-citation rejection, and abstention on missing evidence.
- Forecast tests verify one record per market/bucket, persistence of the point-in-time feature/vote/evidence snapshot, scoring skipped markets at resolution, exactly-once labeling, and no label on unresolved markets.
- Learner tests verify research is a separate bandit arm and compatible state survives capture/restore.
- Engine tests verify 200-market cap, 20-second configured cadence, unchanged bet gates, no duplicate positions, and continued paper-only execution.
- API tests verify the dashboard/status surfaces expose forecasts, research coverage, learning scores, and last errors without exposing secrets.
- No test requires network access or live order placement.

## Scope and Implementation Boundaries

Expected surfaces are `app/markets/polymarket/client.py`, `engine.py`, `llm.py`, `signals.py`, `learner.py`, `app/db.py`, `app/persistence.py`, Polymarket routes in `app/main.py`, Polymarket dashboard rendering, relevant tunables/settings, and existing Polymarket tests. Exact module placement for the RSS cache should follow the repository's existing research/feed abstractions after implementation review.
