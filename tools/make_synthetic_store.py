"""Fill a data store with a SYNTHETIC market, so the improvement pipeline can
be run start to finish without network access (tests, CI, a sandbox).

    python tools/make_synthetic_store.py                 # ~40 coins, 2017 -> today
    python tools/make_synthetic_store.py --coins 25 --start 2018-01-01 --seed 3

Writes into THIS checkout's .cache/store (candles + funding via
app/data/store.py, external series via app/data/series.py) — so run it in a
throwaway copy of the repo, as `tools/run_pipeline.py --synthetic` does,
never in the checkout whose store holds real history. It refuses to touch a
store that already has data unless --force.

The market: a common factor that switches between bull and bear regimes
(so trend rules have something to find), coins with different betas and
volatilities, staggered listings, alts that crash and get delisted, BTC/ETH
hourly bars for the last year, hourly funding, and Fear & Greed / DVOL-like
series tied to the market. Nothing here is evidence about real markets.
"""
import argparse
import datetime as dt
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D, H = 86400, 3600


def build(n_coins=40, start="2017-01-01", seed=7, now=None, say=print, plant_signal=0.0):
    import numpy as np
    from app.data import store, series
    rng = np.random.default_rng(seed)
    now = int(now or (dt.datetime.now(dt.timezone.utc).timestamp()))
    end_day = now // D - 1                                  # last CLOSED day
    d0 = (dt.date.fromisoformat(start) - dt.date(1970, 1, 1)).days
    T = end_day - d0 + 1
    # market factor with regime switches every ~150-400 days
    drift = np.zeros(T)
    t, bull = 0, True
    while t < T:
        n = int(rng.integers(150, 400))
        drift[t:t + n] = 0.0025 if bull else -0.002
        bull = not bull
        t += n
    mkt = drift + rng.normal(0, 0.032, T)
    names = ["BTC-USD", "ETH-USD"] + [f"SYN{i:02d}-USD" for i in range(n_coins - 2)]
    betas = np.r_[1.0, 1.2, rng.uniform(0.9, 1.8, n_coins - 2)]
    idio = np.r_[0.012, 0.02, rng.uniform(0.025, 0.07, n_coins - 2)]
    listed = np.r_[0, 0, rng.integers(0, int(T * 0.7), n_coins - 2)]
    # optional planted cross-coin signal: calmer coins earn a daily premium
    # (a low-volatility effect that momentum doesn't capture) — for exercising
    # the model -> candidate -> research -> live path end to end
    premium = plant_signal * (idio.mean() - idio) / idio.std()
    total_rows = 0
    for j, p in enumerate(names):
        r = betas[j] * mkt + rng.normal(0, idio[j], T) + premium[j]
        lp = np.cumsum(r)
        close = 100 * np.exp(lp - lp[listed[j]])
        alive = np.arange(T) >= listed[j]
        if j >= 2:                                         # crashed alts get delisted
            peak = np.maximum.accumulate(np.where(alive, close, 0))
            dead = np.flatnonzero(alive & (close < 0.08 * peak) & (np.arange(T) > listed[j] + 200))
            if len(dead):
                alive &= np.arange(T) < dead[0]
        dv = 5e9 / (1 + j) ** 1.3
        rows = []
        for t in np.flatnonzero(alive):
            c = float(close[t])
            o = float(close[t - 1]) if t > 0 and alive[t - 1] else c
            hi, lo = max(o, c) * (1 + abs(rng.normal(0, 0.01))), min(o, c) * (1 - abs(rng.normal(0, 0.01)))
            rows.append([(d0 + int(t)) * D, lo, hi, o, c, dv / c * float(np.exp(rng.normal(0, 0.3)))])
        store.ingest_candles(p, D, rows, now=now, source="synthetic")
        total_rows += len(rows)
    say(f"daily: {n_coins} coins, {total_rows:,} bars ({names[0]} .. {names[-1]})")
    # hourly bars for BTC / ETH, last ~365 days
    h_end = now // H - 1
    hours = 365 * 24
    for j, p in enumerate(names[:2]):
        r = rng.normal(0.00005, 0.0065 * betas[j], hours)
        c = 30000.0 / (1 + j * 14) * np.exp(np.cumsum(r))
        rows = []
        for k in range(hours):
            cc = float(c[k])
            oo = float(c[k - 1]) if k else cc
            rows.append([(h_end - hours + 1 + k) * H, min(oo, cc) * 0.998, max(oo, cc) * 1.002,
                         oo, cc, 1e8 / cc])
        store.ingest_candles(p, H, rows, now=now, source="synthetic")
    say(f"hourly: BTC-USD, ETH-USD, {hours} bars each")
    # hourly funding tied to the market's 30-day trend
    mom = np.convolve(mkt, np.ones(30) / 30, mode="full")[:T]
    for venue, coins, first in (("deribit", ("BTC", "ETH"), max(0, T - 5 * 365)),
                                ("hyperliquid", ("BTC", "ETH"), max(0, T - 2 * 365))):
        for coin in coins:
            rows = [[(d0 + t) * D + h * H, float(1e-5 + 5e-3 * mom[t] + rng.normal(0, 5e-6))]
                    for t in range(first, T) for h in range(24)]
            store.ingest_funding(venue, coin, rows, now=now, source="synthetic")
    say("funding: deribit + hyperliquid BTC/ETH, hourly")
    # external series known one day after their stamp
    btc = np.cumsum(mkt)
    vol30 = np.array([mkt[max(0, t - 29):t + 1].std() for t in range(T)]) * math.sqrt(365) * 100
    fg = [((d0 + t) * D, float(np.clip(50 + 40 * np.tanh(5 * (btc[t] - btc[max(0, t - 30)]))
                                       + rng.normal(0, 5), 0, 100)), (d0 + t + 1) * D)
          for t in range(T) if t > 400]
    dvol = [((d0 + t) * D, float(vol30[t] + rng.normal(0, 3)), (d0 + t + 1) * D)
            for t in range(T) if t > 1500]
    series.ingest("fear_greed", fg, source="synthetic", now=now)
    series.ingest("dvol_btc", dvol, source="synthetic", now=now)
    say(f"series: fear_greed ({len(fg)} days), dvol_btc ({len(dvol)} days)")
    return {"coins": n_coins, "daily_bars": total_rows, "days": T}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", type=int, default=40)
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plant-signal", type=float, default=0.0,
                    help="daily low-volatility premium per cross-coin sd (e.g. 0.003)")
    a = ap.parse_args()
    from app.data import store
    if store.products("candles", D) and not a.force:
        sys.exit(f"{store.STORE} already holds candles; refusing to mix synthetic data in "
                 f"(use a throwaway copy, or --force)")
    build(a.coins, a.start, a.seed, plant_signal=a.plant_signal)


if __name__ == "__main__":
    main()
