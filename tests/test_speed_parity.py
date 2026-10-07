"""Speedups must not change results: each fast path is checked against the
original computation."""
import json
import math
import random
import statistics
import sys
import types

import numpy as np
import pytest


def _candles(n, seed, step=3600, t0=1_700_000_000, vol=0.008):
    r = random.Random(seed)
    px, out = 100.0 + seed, []
    for i in range(n):
        op = px
        px *= math.exp(0.0002 * (seed % 3 - 1) + r.gauss(0, vol))
        out.append([t0 + i * step, min(op, px) * 0.997, max(op, px) * 1.003, op, px,
                    1000.0 + 10 * (i % 7)])
    return out


def _universe(k=6, n=320):
    names = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "DOGE-USD"][:k]
    return {p: _candles(n, i) for i, p in enumerate(names)}


def _report(r):
    return json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                      sort_keys=True, default=str)


# ------------------------------------------------------------------ stdev
def test_fsum_stdev_matches_statistics_to_one_ulp():
    from app.data.features import stdev
    rng = random.Random(0)
    worst = 0.0
    for _ in range(3000):
        xs = [rng.gauss(0, 10 ** rng.uniform(-5, 0)) for _ in range(rng.randint(3, 80))]
        for ddof, ref in ((1, statistics.stdev), (0, statistics.pstdev)):
            a, b = ref(xs), stdev(xs, ddof=ddof)
            worst = max(worst, abs(a - b) / a)
    assert worst < 1e-15


# ------------------------------------------------------------------ disk bar cache
def test_persisted_replay_equals_fresh_replay_after_new_revised_and_dropped_bars():
    from app.backtest import replay as rp
    from app.config import CANDLE_HISTORY
    n = CANDLE_HISTORY + 120                 # long enough for full-length windows
    base = _universe(n=n)
    c1 = {}
    first = rp.run_replay(base, cache=c1, maker=False, shorts=True, persist=True)
    assert first["ok"]
    assert c1["disk"]["reused_bars"] == 0

    # a day later: 24 new bars, one revised bar late in history, the window's
    # first 10 bars dropped (rolling download)
    later = {}
    for i, (p, rows) in enumerate(base.items()):
        rows = [list(r) for r in rows] + _candles(n + 24, i)[n:]
        if p == "SOL-USD":
            rows[n - 20][4] *= 1.01
        later[p] = rows[10:]
    c2 = {}
    warm = rp.run_replay(later, cache=c2, maker=False, shorts=True, persist=True)
    fresh = rp.run_replay(later, cache={}, maker=False, shorts=True)
    assert _report(warm) == _report(fresh)
    d = c2["disk"]
    # bars whose window lost no front rows and that precede the revision are
    # reused; everything that read a changed or missing row is recomputed
    assert d["reused_bars"] > 0 and d["computed_bars"] > 0
    assert d["reused_bars"] < len(c2["bars"])


@pytest.fixture
def store_stub(monkeypatch):
    """run_report records the data fingerprint from app/data/store.py; stand
    in for it when that module isn't present in this checkout."""
    try:
        import app.data.store  # noqa: F401
        return
    except ImportError:
        import app.data
        stub = types.ModuleType("app.data.store")
        stub.fingerprint = lambda *a, **k: "test"
        monkeypatch.setitem(sys.modules, "app.data.store", stub)
        monkeypatch.setattr(app.data, "store", stub, raising=False)


def test_persisted_report_equals_fresh_report(store_stub):
    from app.backtest import replay as rp
    from app.backtest import bar_cache as BC
    import shutil
    shutil.rmtree(BC.DIR, ignore_errors=True)
    data = _universe(n=300)
    a = rp.run_report(data, chop_ab=False, filter_ab=False)
    b = rp.run_report(data, chop_ab=False, filter_ab=False)      # all bars from disk
    for k in ("full", "first_half", "second_half", "verdict"):
        assert json.dumps(a[k], sort_keys=True) == json.dumps(b[k], sort_keys=True)


def test_cache_key_changes_with_tunables():
    from app.backtest import bar_cache as BC
    from app import tunables
    off = {"only": ("trend",), "gate": 0.3, "shorts": True, "veto_align": 0.8}
    k1 = BC.cache_key(off, 24, 300, False)
    tunables.update({"stop_atr_mult": tunables.tv("stop_atr_mult") + 0.5})
    try:
        assert BC.cache_key(off, 24, 300, False) != k1
    finally:
        tunables.reset()
    assert BC.cache_key(off, 24, 300, True) != k1


# ------------------------------------------------------------------ parallel seeds
def test_parallel_ml_tracks_equal_serial():
    from app.backtest import replay as rp, ablation as ab
    cache = {}
    rp.run_replay(_universe(k=5, n=230), cache=cache, extras=True, maker=False, shorts=True)
    serial = ab.ml_tracks(cache, 2, workers=1)
    parallel = ab.ml_tracks(cache, 2, workers=2)
    assert serial == parallel


# ------------------------------------------------------------------ vol forecast parity
def test_challenger_vol_forecast_matches_live_core_panel():
    """The research loop scores a vol_forecast candidate on the full store
    panel; live trading loads only the candidate's coins. Both must size with
    the same forecast."""
    from app.engine import panel as P, challengers as C
    day = 86400
    cs = {"BTC-USD": _candles(800, 1, step=day, vol=0.03),
          "ETH-USD": _candles(800, 2, step=day, vol=0.04),
          # an older coin: the full panel starts 200 days before BTC/ETH
          "OLD-USD": _candles(1000, 3, step=day, t0=1_700_000_000 - 200 * day, vol=0.05)}
    now = 1_700_000_000 + 900 * day
    full = P.from_candles(cs, now=now)
    live = P.from_candles({k: cs[k] for k in ("BTC-USD", "ETH-USD")}, now=now)
    cfg = C.CANDIDATES["btc_eth_trend_volf"]
    W_full = C.weights(cfg, full, vol=C.vol_forecast_for(cfg, full))
    W_live = C.weights(cfg, live)
    off = int(np.flatnonzero(full.days == live.days[0])[0])
    for j, c in enumerate(live.coins):
        np.testing.assert_allclose(W_full[off:off + live.T, full.coins.index(c)],
                                   W_live[:, j], rtol=0, atol=1e-12)
    assert W_live.sum() > 0
