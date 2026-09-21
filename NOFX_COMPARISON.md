# What we can learn from NoFxAiOS/nofx (NOFX) — analysis for CryptoMind

Researched: 2026-09-17 · Source: https://github.com/NoFxAiOS/nofx (dev branch,
~12.9k stars, AGPL-3.0, Go backend + React frontend).

> ⚠️ **License note first.** NOFX is **AGPL-3.0**. We can *read it for ideas and
> reimplement concepts ourselves*, but we must **not copy its code** into
> CryptoMind unless we're prepared to make our whole system AGPL. Everything
> below is "adopt the idea, write our own code," never "paste theirs."

---

## 1. What NOFX actually is (so we compare apples to apples)

NOFX's tagline is *"the strategy is a language model."* Each "trader" runs a loop:
**read market structure → ask an LLM for a decision → a Go runtime clamps the
order to hard risk limits the model cannot override → execute → store the full
reasoning.** Key architectural facts:

- **The alpha is an LLM prompt**, not a learned model. It assembles multi-timeframe
  candles + indicators + OI/funding + recent trades into a big prompt, asks
  DeepSeek/Claude/GPT/Qwen/etc., and parses a JSON decision array.
- **A deterministic Go "runtime" is the safety layer** — position caps, leverage
  clamps, exchange-side SL/TP, drawdown auto-close, trade throttling, "safe mode."
- **Multi-trader**: run many model/strategy combos side by side, ranked on a
  public leaderboard by realized return.
- **Every decision is persisted with the model's full chain-of-thought** — "no
  position without a paper trail."
- Nine exchanges, MCP-style AI client abstraction, Telegram agent, x402 micro-payments.

**This is almost the philosophical opposite of CryptoMind.** We are a *quantitative
self-learning* system (bandit + online NN committee + GA evolution + RL sizing,
all learning from realized PnL). They are an *LLM-reasoning* system with a
hard-coded safety cage and no learning loop. That contrast is exactly why some of
their ideas are worth stealing — they've invested heavily in the parts we're
thinner on (the execution/safety cage, the decision audit trail, config-driven
strategy variants), while we're far ahead on the parts they don't have at all
(actual machine learning, walk-forward validation, forgetting bandits).

---

## 2. Side-by-side

| Capability | NOFX | CryptoMind (today) |
| --- | --- | --- |
| Decision engine | LLM prompt → JSON decisions | Ensemble of quant strategies + online NN + GA champions, bandit-weighted |
| Learning from outcomes | ❌ none (stateless LLM each cycle) | ✅ bandit forgetting, NN committee, GA walk-forward, RL sizer |
| Risk "cage" outside the decider | ✅ strong, explicit, code-enforced | ⚠️ good (risk.py sizing + cost gate + kill switch) but less unified |
| Exchange-side SL/TP after entry | ✅ placed immediately on the exchange | ❌ paper stops are simulated in-process |
| Drawdown-from-peak auto-close | ✅ per-position peak-giveback close | ⚠️ portfolio kill switch only, not per-position trailing |
| Trade throttling (min-hold, re-entry cooldown, per-cycle/hr caps) | ✅ explicit | ⚠️ partial (cooldowns exist; no min-hold / per-hour cap) |
| "Safe mode" on repeated decider failures | ✅ blocks new entries until recovery | ⚠️ we gate a broken ML head, but no global decider-health circuit breaker |
| Multi-timeframe context (5m/15m/1h/4h) | ✅ | ⚠️ 5m candles + folded swing-ATR; no true multi-TF feature set |
| Coin selection sources (static / AI-rated / OI-growth / mixed) | ✅ pluggable | ⚠️ narrative-heat discovery + core list (one blended source) |
| Per-decision audit trail w/ full reasoning | ✅ first-class (`DecisionRecord`) | ⚠️ we store per-trade `votes` + regime, not a full rationale record |
| Multiple strategies compared on a leaderboard | ✅ core feature | ❌ single portfolio; GA champions aren't raced head-to-head live |
| Launch preflight (verify funds/model/strategy/balances before start) | ✅ | ⚠️ implicit; no single preflight gate |
| Config-driven "strategy studio" (presets, indicators, leverage, prompts) | ✅ | ⚠️ tunables registry exists but no per-strategy profile objects |

Legend: ✅ have/strong · ⚠️ partial · ❌ absent

---

## 3. Worth incorporating — ranked by value/effort

### Tier 1 — high value, low/medium effort (do these)

**1. Per-decision audit trail ("no position without a paper trail").**
NOFX writes a `DecisionRecord` every cycle: prompt inputs, the reasoning, the
parsed decision, the execution result, and which candidates were considered. We
already store per-trade `votes` + `regime_at_entry`, but not a *cycle-level*
record of "here's every signal, the composite, why we did/didn't act, and what
the risk cage did to the order." **Add a `decisions` table** (cycle #, ts,
per-strategy raw votes, composite, confidence, chosen action, size pre/post risk
clamp, gate reasons, outcome later backfilled). This is the single biggest
debuggability win and it directly serves "why is nothing trading / why did it
size that small" questions like the ones we've been chasing. Low effort, huge
payoff.

**2. Per-position drawdown-from-peak auto-close (trailing giveback).**
NOFX closes a *profitable* position that gives back too much from its peak PnL.
We only have a portfolio-level kill switch. A per-position "peak PnL − current ≥
X%" trailing exit (armed only after a real move, learned from their
2026-07-23 bug: evaluate on **price-basis** PnL, not leverage-multiplied margin
PnL) would lock in winners the current stop-only model lets round-trip. Medium
effort; slots into `risk.py` + the position dict (we already track entry/mark).

**3. Explicit trade throttling: min-hold + per-hour entry cap.**
We have re-entry cooldowns but no minimum hold time or per-hour entry ceiling.
NOFX added these after a live blowup (43 fills, <1h avg hold in a day). Cheap
tunables (`min_hold_sec`, `max_entries_per_hour`) that prevent overtrading /
churn. Low effort.

**4. A unified "safe mode" / decider-health circuit breaker.**
NOFX blocks *new entries* when the model repeatedly fails, until it recovers. We
gate a broken ML head, but nothing globally halts entries when the whole decision
stack is unhealthy (e.g. data stale, N consecutive tick errors, model + bandit
both degenerate). Add a `safe_mode` flag in the orchestrator that suspends new
opens (but keeps managing/closing existing risk) on a health score. Medium effort.

**5. Launch preflight gate.**
One function that verifies, before the loop is allowed to open anything: data
feed healthy, enough history per product, model dimensionality matches, risk
config sane, not killed/halted. Turns a class of silent "nothing works" states
into one explicit, logged reason. Low effort — and would have surfaced the ML
`n_updates=0` and GA schema bugs faster.

### Tier 2 — high value, higher effort (plan these)

**6. True multi-timeframe feature context (5m/15m/1h/4h).**
NOFX feeds the decider aligned OHLCV+indicators across four timeframes. We use 5m
candles + a folded swing-ATR. Giving the NN committee and the strategies a proper
multi-TF view (e.g. 15m/1h/4h EMA/RSI/MACD alongside 5m) is a legitimate alpha
upgrade — regime and trend confirmation are much cleaner across timeframes.
Medium-high effort (feature-vector expansion → model input dim change →
retrain/migrate; our persistence already guards dim mismatch).

**7. Pluggable coin-selection sources ("mixed" mode).**
NOFX composes candidates from static + AI-rated pool + OI-growth ranking, tagging
multi-source coins. We have narrative-heat discovery + a core list. Adding an
**OI-growth / capital-flow source** (we already pull OI & funding in
`derivatives.py`!) and letting the universe be a *blend with source tags* would
improve what we even look at. The dual-signal idea ("this coin showed up in two
independent sources → higher prior") is a nice, cheap edge. Medium effort.

**8. Optional LLM "advisor" as ONE more bandit arm — not the driver.**
This is the interesting philosophical merge. We should **not** hand the wheel to
an LLM (we'd throw away our learning edge and add latency/cost/nondeterminism).
But we *could* add an optional strategy sleeve `strat_llm` that, on a slow cadence,
asks an LLM for a directional lean per coin given our compiled context, and feeds
that in as **one vote the bandit weights like any other.** If the LLM is good,
the bandit rewards it; if not, forgetting decays it to zero — exactly the safe way
to test it. It rides all our existing validation instead of bypassing it. Medium
effort, and fully reversible.

### Tier 3 — architecturally interesting, probably not for us

- **Exchange-side SL/TP after entry.** Critical for *live* trading on a real
  venue (their #1 safety feature), but we're paper/shadow today. Worth adopting
  *when* we wire a live exchange — align with the OMS/idempotency work already in
  our critique backlog. Not now.
- **Multi-trader leaderboard.** We could race our GA champion portfolio + the
  ensemble + (optionally) the LLM sleeve as separate paper "traders" ranked by
  realized return — a nicer version of what we do internally. Neat, not urgent.
- **x402 micropayments / MCP client zoo / 9 exchanges / Telegram AI agent.**
  Out of scope for our goals.

---

## 4. Proposed first steps — ✅ ALL DELIVERED

> Every item below was implemented, tested, and pushed. Current HEAD **`4afb09a`**,
> **197 tests passing**. Kept here as a record of what was adopted from the analysis.

1. ✅ **`decisions` audit table + writer** in the orchestrator cycle (Tier-1 #1),
   with a `/decisions` view and Telegram command. *Complements `/brains`.*
2. ✅ **Per-position trailing giveback exit** (`paper.py`/`risk.py`), price-basis,
   armed after a real move (Tier-1 #2), with tunables + tests.
3. ✅ **`min_hold_sec` + `max_entries_per_hour` throttles** (Tier-1 #3).
4. ✅ **`safe_mode` circuit breaker + launch preflight** (Tier-1 #4 & #5) in the
   `guardian` "decider health" module.
5. ✅ **Multi-timeframe features** (`mtf_align`, Tier-2 #6) and the **OI-growth
   universe source** (Tier-2 #7).
6. ✅ **LLM-as-a-bandit-arm** sleeve (Tier-2 #8) — now defaults to **Gemini**
   (`gemini-2.5-flash`), still OFF by default and inert without a key. Measured
   honestly against the audit trail; the forgetting bandit decays it if it adds
   no edge.

### Beyond the original plan (also shipped)
- **Adaptive loss-cut exit + hold-vs-fold learning** and a **direction learner**
  (per-regime long/short edge + higher-TF veto) — both learn from realized/
  counterfactual PnL. Extends the "runtime disposes" idea with a *learned*,
  not just hard-coded, exit.
- **Meme-coin trading sleeve** — curated seed + CoinGecko meme-category
  auto-discovery under a contained risk envelope (this is the kind of
  config-driven "strategy variant" NOFX leans on, done our way).
- **NumPy-vectorized backtester** — ~5× faster GA evaluation, parity-tested.

## 5. What NOT to copy
- Don't replace our quant/learning core with an LLM decider — that's their whole
  design and it has *no* learning loop; our edge is precisely the learning.
- Don't copy any NOFX source (AGPL). Reimplement concepts from these notes.
- Don't add leverage/perp complexity or 9-exchange breadth chasing feature parity;
  it's scope we don't need.
