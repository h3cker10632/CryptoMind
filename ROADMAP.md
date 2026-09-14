# CryptoMind — Path to Reliable Live Trading

A staged progression with **hard gates**. You only advance when the current
stage's exit criteria are met. Expect the full path to take 6–12 months.
Skipping stages is how automated trading accounts die.

---

## Stage 0 — Honest baseline (now → 2 weeks)

The system works, but recognize what it is today: a research prototype on a
sandbox that sleeps, with ~40 days of hourly backtest history and minutes of
live learning. None of its intelligence is statistically proven yet.

**Do:**
- [ ] Deploy to an always-on machine (a $5–10/mo VPS: Hetzner, DigitalOcean,
      OVH — or a spare mini-PC). The system must run 24/7 uninterrupted;
      learning and trade statistics are meaningless with gaps.
- [ ] Add process supervision: `systemd` unit or Docker with
      `restart: always` so crashes self-recover (state persistence already
      handles the resume).
- [ ] Set up alerting you'll actually see: a Telegram bot or email on
      kill-switch, daily-loss halt, data-feed outage, and crash. You cannot
      supervise what you don't see.
- [ ] Start a trading journal: weekly snapshot of equity, trade count,
      win rate, per-strategy bandit posteriors.

**Exit gate:** 14 consecutive days of uptime > 99%, alerts verified working.

---## Stage 1 — Statistical validation on paper (1–3 months)

The single biggest risk is deploying a system whose edge is noise. You need
enough samples to distinguish skill from luck.

**Do:**
- [ ] Accumulate **≥ 300 closed trades** across regimes (this is why
      exploration trading matters).
- [ ] Track these numbers weekly:
      - Expectancy per trade **after fees/slippage** (must be positive)
      - Profit factor (gross wins / gross losses; want > 1.3)
      - Max drawdown vs the 15% kill threshold
      - Sharpe on daily equity returns (want > 1.0 sustained)
      - **Benchmark: did it beat holding BTC over the same period?**
        Risk-adjusted, at minimum. If buy-and-hold wins after 3 months,
        the system isn't ready.
- [ ] Kill or fix what the bandit demotes: if a strategy's posterior mean
      is negative after 100+ samples, remove or rework it.
- [ ] Extend backtesting depth: fetch 1+ year of daily and 90 days of hourly
      candles; re-run walk-forward per strategy per regime.
- [ ] Watch the paper-fill assumption: our 5 bps slippage is optimistic for
      thin coins. Compare paper fill prices vs the live order book spread on
      every trade (the data is already there — spread_bps).

**Exit gate:** 3 consecutive months of positive expectancy after costs,
max DD < 10%, and outperformance (risk-adjusted) vs BTC buy-and-hold.
**If it fails: iterate here. Do not proceed. Most systems never pass this
gate — that's the point of it.**

---

## Stage 2 — Live-market plumbing with zero risk (2–4 weeks, overlaps Stage 1)

Build and battle-test the real exchange path without risking a dollar.

**Do:**
- [ ] Pick the live venue (Coinbase Advanced Trade / Kraken / OKX). Open the
      account, complete KYC, understand fee tiers (real taker fees may be
      0.4–0.6% at low volume — 4–6x our simulated 10 bps! This alone can
      erase a thin edge; factor it into Stage 1 math).
- [ ] Create **read-only API keys** first. Build a `LiveBroker` implementing
      the same interface as `PaperBroker` (buy/sell/equity/exposure), but
      run it in **shadow mode**: it fetches real balances and simulates
      orders against the real order book, logging what WOULD have filled.
- [ ] Upgrade market data from REST polling (30s) to **WebSocket streams**
      for the live venue. 30s polling is fine for research, too slow for
      real execution of stop-losses.
- [ ] Add order-lifecycle safety: idempotent client order IDs, timeout +
      retry logic, reconciliation loop (every minute: does the exchange's
      position state match ours? If not — alert and halt).
- [ ] Secrets: API keys in environment variables or a secrets manager,
      never in code or state.json. IP-whitelist the VPS. **Never enable
      withdrawal permissions on trading keys.**
- [ ] Chaos-test: kill the process mid-order, disconnect the network,
      feed it malformed API responses. It must fail safe (halt + alert),
      never fail open (orphaned positions, duplicate orders).

**Exit gate:** 2+ weeks of shadow trading where shadow fills track paper
fills within a known, small tracking error, plus passing chaos tests.

---

## Stage 3 — Canary live (2–3 months)

Real money, trivial size. The goal is not profit — it's discovering
everything that's different about live execution.

**Do:**
- [ ] Fund with an amount you can lose without caring: **$500–$2,000 max.**
- [ ] Scale all position sizing down proportionally (config change).
- [ ] Add exchange minimums handling (min order size, lot size, precision).
- [ ] Run paper and live **in parallel** — same signals, both executing.
      Track the live-vs-paper divergence (slippage reality check, fill
      rates, partial fills, fees). This divergence number is your most
      important discovery of this stage.
- [ ] Keep the human-in-the-loop: daily check-ins for the first month.
      Kill-switch drills — actually press it, verify it flattens.
- [ ] Taxes/records: every fill is a taxable event in the US. Export the
      trade log monthly. Talk to a CPA about trader tax treatment in Iowa.

**Exit gate:** 2–3 months where (a) live expectancy is still positive after
REAL fees, (b) live-vs-paper divergence is understood and bounded, (c) zero
operational incidents (orphaned orders, reconciliation failures, missed
stops).

---

## Stage 4 — Graduated scaling (6+ months, ongoing)

**Do:**
- [ ] Scale in steps: 2x capital only after each 4–8 week window with
      positive after-cost results and no gate violations. E.g.:
      $2k → $5k → $10k → $25k...
- [ ] Watch capacity constraints: at $10k+ positions on thin discovered
      coins, YOUR OWN orders move the book. Enforce per-coin liquidity
      caps (position ≤ 0.5–1% of coin's 24h volume).
- [ ] Never scale after a hot streak. Scale on schedule, or not at all —
      hot-streak scaling is how you maximize exposure right before regime
      change.
- [ ] Set an absolute account ceiling you never exceed with an
      experimental system (a small % of your investable assets).

**Ongoing forever:**
- Edge decays. The bandit/retraining loop helps, but expect strategies to
  die and need replacement. The research infrastructure is the durable
  asset, not any particular strategy.
- Quarterly review: is the system still beating its benchmark after costs?
  If it fails 2 consecutive quarters, drop back to paper and iterate.

---

## Reality checks (read this twice)

1. **The most likely outcome, statistically, is that the system does not
   beat buy-and-hold BTC after costs.** That's true for most retail algo
   systems. The gates above exist to discover this cheaply on paper rather
   than expensively live.
2. **Fees are the boss fight.** At 0.5% round-trip real-world cost, a
   strategy trading daily needs to overcome ~180%/year in friction. Fewer,
   better trades beat many mediocre ones — the confidence gates and
   blackout windows already push this direction.
3. **Learning ≠ proven.** The neural net/bandit/RL adapt, but adaptation on
   noise is just expensive curve-fitting. Sample counts and out-of-sample
   discipline (already built into the GA promotion gate) are your defense.
4. **The kill switch is the most valuable feature in the system.** Reliable
   live trading is less about the upside engine and more about guaranteed,
   tested, bounded downside.
5. **This is not financial advice** — it's an engineering progression for
   a system you built. Size everything so total loss is an acceptable
   tuition cost.

## Priority build list (technical debt before live)

| Priority | Item | Why |
|---|---|---|
| P0 | Always-on deployment + supervision + alerting | everything else is meaningless with gaps |
| P0 | Real-fee modeling (0.4–0.6% taker) in paper + backtests | current 10 bps flatters every result |
| P1 | WebSocket market data for execution path | 30s polling can't manage stops live |
| P1 | LiveBroker (shadow mode) + reconciliation loop | the actual live plumbing, tested safely |
| P1 | 1-year backtest history + per-regime walk-forward | 40 days proves nothing |
| P2 | Per-coin liquidity caps (vs 24h volume) | discovered coins are thin |
| P2 | Deflated Sharpe / PBO overfitting stats on backtests | blueprint item, matters at scale-up |
| P3 | Multi-venue support | redundancy + fee optimization, later |
