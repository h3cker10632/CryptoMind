# Data Purity Audit — CryptoMind

**Scope:** every `.py` file under `app/` (20k LOC, 108 files), plus stray repo artifacts.
**Question asked:** is every number REAL — sourced from a real feed or computed from real
data — with nothing generated, randomized, mocked, or faked and presented as real?

**Bottom line: the data is pure.** Every number surfaced to you or used in a trading
decision traces back to a real API response or a real computation on real inputs. The only
randomness in the live path is legitimate *algorithmic* randomness (learning exploration,
model initialization, request jitter, security tokens) — none of it fabricates market data,
prices, sentiment, or P&L. Two housekeeping issues (stray committed temp files) are noted
at the end; neither is fake data.

---

## 1. Real data sources (verified hitting real endpoints)

| Feed | Source | Notes |
|---|---|---|
| Candles / ticker / order book | `api.exchange.coinbase.com` | `market.py` — `float(t["price"])` etc. from live REST; WebSocket top-of-book in `ws_market.py` |
| Derivatives (funding, OI, L/S ratio) | `www.okx.com` | `derivatives.py` — all `float()`-parsed from real `data` payloads |
| Fear & Greed index | `api.alternative.me/fng` | `research.py` |
| News / social sentiment text | CoinDesk, CoinTelegraph, Decrypt, Reddit, Google News RSS, etc. | `research.py` — real RSS pulled, then lexicon-scored |
| Macro calendar | `nfs.faireconomy.media` | `calendar.py` |
| Universe / meme discovery | CoinGecko + OKX | `universe.py`, `memes.py` |

When a feed is unavailable these return `None` / skip / neutral — **never a fabricated price.**

## 2. Money, P&L, equity — all computed from real fills at real prices

`execution/paper.py`:
- `equity = cash + Σ position_value(pos, market.price(p))` — marked at **live price**.
- Fills use the **real price passed from the market ticker**, plus a configurable slippage
  *model* and real fees.
- `realized_pnl` accumulates from actual close events: `gross − fee − cost_basis`.
- `START_CASH` is a declared paper-account starting balance (config), not fabricated data.

## 3. Randomness inventory — every use is legitimate, none fabricates data

| Location | Use | Legitimate? |
|---|---|---|
| `learn/bandit.py` | Thompson-sampling `gauss(mu,sd)` | ✅ exploration |
| `learn/rl_risk.py` | ε-greedy action pick | ✅ exploration |
| `learn/evolution.py` | GA random genome / mutation / crossover | ✅ optimizer |
| `learn/online_model.py` | NN weight init + prioritized-replay sampling | ✅ ML internals |
| `orchestrator.py` | `random < explore_prob` gate | ✅ exploration |
| `data/research.py` | `uniform` sleep jitter | ✅ polite rate-limiting |
| `security.py` | `secrets.token_urlsafe` | ✅ auth token |
| `execution/shadow.py` | simulated partial fills | ✅ **shadow mode only** — never real risk, explicitly labeled |
| `backtest/*`, `tools/measure_scaleout.py`, `tests/*` | synthetic paths | ✅ backtests/tests are *supposed* to be synthetic |

## 4. Transparency notes (real-but-derived values — by design, not "fake")

These are honest computed stand-ins, not fabricated data, but you should know they exist:
- **Gap mark:** `equity()`/`exposure()` mark a position at its `entry` price if the live
  price is momentarily `None` (a brief, conservative gap-fill during a data hiccup).
- **Neutral indicator fills:** RSI → 50.0, hype-velocity → 0.0, trend-sign → 0.0 when there
  aren't enough bars yet. Standard neutral defaults, clearly bounded by MIN_BARS logic.
- **Slippage is a model** (10 bps, configurable) not measured real slippage — inherent to
  paper trading; labeled "simulated slippage" in config/tunables.

## 5. Housekeeping issues found (NOT fake data — repo hygiene)

- ⚠️ **Two stray temp files are tracked by git** (committed in `31f4589`):
  - `tmpt1py54uh` — a stale account-state snapshot (real, but stale/duplicate of `state.json`).
  - `tmph44l_e8_` — a settings blob.
  These are accidental commits; recommend `git rm` + `.gitignore` them.
- `api_token.txt` exists on disk but **is correctly gitignored** (not tracked). Fine, though
  it's a secret at rest.

---

**Conclusion:** no generated, randomized, or mocked value is ever passed off as real market
data, price, sentiment, or P&L. The learning stack's randomness is exploration/initialization
only. Data integrity: **clean.**
