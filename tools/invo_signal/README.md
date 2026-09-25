# Invo Signal Edge Study (standalone harness)

Decides — on evidence, not hope — whether aggregated Invo top-trader positioning
should become an input feature for CryptoMind's learner. It answers three
questions:

1. **Does it predict?** Information Coefficient (Spearman), Pearson, mutual
   information, with a permutation p-value.
2. **Is it *new* information?** `partial_ic` — IC after removing what your
   existing funding / open-interest / long-short baseline already explains. If
   this collapses toward zero, the Invo signal is redundant with data you have.
3. **Does adding it actually help?** A walk-forward logistic-regression
   **ablation**: directional accuracy + log-loss, baseline vs baseline+Invo.

This mirrors how the scale-out prototype was decided: measure first, ship only
if the numbers earn it.

## Important: this ships NO scraper

`collector.py` deliberately contains no scraper and no reverse-engineered Invo
API client. Scraping behind Invo's login / hitting their private API likely
violates their ToS — that's your decision with your own authorized access. The
harness consumes **data you've already collected** as JSON (schemas in
`collector.py`). It is fully decoupled from `app/` (no imports either way), so it
never touches live trading and keeps CryptoMind clear of Maxun's AGPL-3.0.

## Inputs (all files you provide)

- `--snapshots snaps.json` — list of positioning snapshots (schema in
  `collector.py`): each has `ts` and a list of `traders`, each with `rank`,
  optional `score`, and `positions` (`asset`, `direction`, `notional`,
  `leverage`).
- `--prices prices.json` — `{"BTC": [[ts, close], ...], ...}` for the assets you
  trade. Forward returns are computed no-lookahead.
- `--baseline baseline.json` *(optional but recommended)* —
  `{"BTC": [[ts, funding_norm, oi_change, long_short], ...]}`, the same features
  `app/data/derivatives.py` already produces. Enables the orthogonality test.

## Collecting snapshots — the authorized poller

`poller.py` is a standalone scheduler that produces `snapshots.json` for you. It
provides all the reusable plumbing — bearer auth, retry/backoff, 429 rate-limit
handling, atomic append, graceful shutdown — and leaves exactly one thing to you:
a **mapper** that turns *your* account's response into the snapshot schema, plus
your own authorized token. It ships no reverse-engineered endpoints.

```bash
export INVO_API_BASE="https://app.invoapp.com"
export INVO_TOKEN="<your authorized session token>"     # never logged
export INVO_LEADERBOARD="/<path returning ranked traders>"
export INVO_POSITIONS_TMPL="/<path per trader>/{id}/positions"   # optional
export INVO_TOP_N=25 INVO_INTERVAL_SEC=300 INVO_OUT=snapshots.json

python -m tools.invo_signal.poller --once        # smoke test (one snapshot)
python -m tools.invo_signal.poller               # loop on the interval
```

Implement `default_mapper` in `poller.py` (it raises with a worked skeleton
until you do). Whether your access is within Invo's ToS is your call — the poller
only wires up transport + scheduling, and stays fully decoupled from `app/`.

## Run the study

```bash
python -m tools.invo_signal.run_study \
    --snapshots snaps.json --prices prices.json \
    --baseline baseline.json --horizon-hours 4 --rank-decay 1.0
```

Tuning knobs: `--horizon-hours` (prediction horizon), `--rank-decay` (how much
top traders outweigh the rest), `--use-score` (weight by trader quality score).

## Verify the harness itself

```bash
python -m pytest tests/test_invo_study.py -q      # edge detected, noise rejected
# or eyeball it:
python -m tools.invo_signal.make_synthetic --mode edge --rho 0.35 --out-prefix /tmp/edge
python -m tools.invo_signal.run_study --snapshots /tmp/edge_snapshots.json --prices /tmp/edge_prices.json --horizon-hours 1
```

`make_synthetic.py` is a **correctness fixture only** — a dataset with a known
embedded (or absent) relationship, never real Invo data and never fed to the
live model.

## Reading the verdict

- **NO EDGE / INCONCLUSIVE** → don't add the feature.
- **NEGLIGIBLE** → statistically real but too small to survive cost/latency.
- **PROMISING** → then check per-asset `partial_ic` and ablation `d_acc`: if the
  edge *survives the baseline* and the ablation accuracy delta stays positive,
  it's earned a place as a measured feature in `app/learn/online_model.py`
  (`build_x` + `N_IN` bump), subject to the same drift/gating as everything else.
