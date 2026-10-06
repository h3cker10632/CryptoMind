"""Daily lab: features use only past data, the rank model is walk-forward
(no label from after a refit is ever trained on), the decision rule needs
both halves, and the live core runs the same weights the lab simulated."""
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest import daily_lab as dl

COINS = ("BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD", "DOGE-USD", "XRP-USD")


def _daily(n=900, seed=0, drift=0.0, t0=None):
    r = random.Random(seed)
    t0 = t0 or int(time.time() // 86400 - n - 2) * 86400
    px, out = 100.0, []
    for i in range(n):
        px *= math.exp(drift + r.gauss(0, 0.03))
        out.append([t0 + i * 86400, px * 0.98, px * 1.02, px, px, 1000 + r.random() * 100])
    return out


def _universe(n=900):
    return {p: _daily(n, seed=i, drift=0.002 * (i - 3) / 3) for i, p in enumerate(COINS)}


def test_features_use_only_history_up_to_the_bar():
    U = _universe(400)
    days, panel = dl.daily_panel(U)
    k200 = days[200]
    base = dict(dl._features_by_day(days, panel))[k200]
    # wreck every price AFTER day 200: day-200 features must not change
    U2 = {p: [r if r[0] // 86400 <= k200 else [r[0]] + [x * 7 for x in r[1:5]] + [r[5]]
              for r in rows] for p, rows in U.items()}
    days2, panel2 = dl.daily_panel(U2)
    assert dict(dl._features_by_day(days2, panel2))[k200] == base
    assert dl.coin_features([1.0] * 20) is None        # too little history


def test_walk_forward_never_trains_on_unfinished_labels(monkeypatch):
    days, panel = dl.daily_panel(_universe(800))
    fbd = dl._features_by_day(days, panel)
    labels = dl._labels(days, panel)
    refits = []
    real = dl.train_set

    def spy(samples, labels_, k):
        used = [j for j, p, x in samples if j + dl.H < k]
        assert used and max(used) + dl.H < k          # label ended before the refit
        refits.append(k)
        return real(samples, labels_, k)
    monkeypatch.setattr(dl, "train_set", spy)
    preds = dl.walk_forward_preds(fbd, labels, min_train_days=365, refit_days=60)
    assert len(refits) >= 5
    assert min(k for k, _ in preds) >= fbd[0][0] + 365   # nothing scored before training


def test_decision_requires_both_halves_and_rank_beats_momentum():
    def v(h1, h2, full=1.0):
        return {"first_half": {"sharpe": h1}, "second_half": {"sharpe": h2},
                "full": {"sharpe": full}}
    base = {f"{s}+{z}": v(0.5, 0.5) for s in dl.SELECTIONS for z in dl.SIZINGS}
    vs = dict(base, **{"momentum+equal": v(0.9, 0.4)})            # one half only
    assert dl.decide(vs)["variant"] == "trend+equal"
    vs = dict(base, **{"rank+equal": v(0.9, 0.9), "momentum+equal": v(1.0, 1.0, 0.9)})
    d = dl.decide(vs)
    assert d["variant"] == "momentum+equal"                       # rank lost to momentum
    assert "rank+equal" not in d["passed"]


def test_target_weights_trend_equal_and_vol_target_caps():
    days, panel = dl.daily_panel(_universe(400))
    fbd = dl._features_by_day(days, panel)
    k, feats = fbd[-1]
    w, _ = dl.target_weights(feats, "trend", "equal", day=k)
    elig = [p for p, f in feats.items() if f["n_days"] >= 100]
    assert all(abs(x - 1 / len(elig)) < 1e-12 for x in w.values())
    wv, _ = dl.target_weights(feats, "trend", "vol_target", day=k, vol_target=5.0)
    assert sum(wv.values()) <= 1.0 + 1e-9                         # never above 100%
    wm, prev = dl.target_weights(feats, "momentum", "equal", day=k, top_k=3)
    assert len(wm) <= 3 and prev["picks"] == list(wm)
    wm2, prev2 = dl.target_weights(feats, "momentum", "equal", prev=prev, day=k + 3)
    assert prev2 is prev                                          # weekly: no re-pick


def test_run_lab_report_shape():
    rep = dl.run_lab(_universe(900), min_train_days=365)
    assert rep["ok"], rep
    assert set(rep["variants"]) == {f"{s}+{z}" for s in dl.SELECTIONS for z in dl.SIZINGS}
    assert rep["decision"]["selection"] in dl.SELECTIONS


def _settings(monkeypatch, **vals):
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k: vals[k] if k in vals else real(k))


def test_core_runs_the_backtested_weights(monkeypatch):
    from app.strategies.core import CoreBook
    monkeypatch.setattr("app.config.PRODUCTS", list(COINS))
    _settings(monkeypatch, core_allocation_pct=50, core_assets="UNIVERSE",
              core_trend_filter=True, core_sma_days=100,
              core_selection="momentum", core_sizing="vol_target",
              core_top_k=3, core_vol_target=0.5, core_tranches=7, core_hysteresis=0.02,
              core_strategy="settings")
    U = _universe(400)
    c = CoreBook()
    t = c.targets(100_000, U)
    # the live core must hold exactly the backtest's weights for today
    from app.engine import panel as P, strategies as S
    W = S.trend_portfolio(P.from_candles(U), "momentum", "vol_target", sma=100, top_k=3,
                          vol_target=0.5, tranches=7, hysteresis=0.02)
    w = dict(zip(sorted(U), W[-1]))
    assert sum(1 for v in w.values() if v > 0) > 0
    for a in COINS:
        assert abs(t[a] - 50_000 * w.get(a, 0.0)) < 1e-6
    assert c.mode_used == ("momentum", "vol_target")


def test_core_auto_falls_back_to_original_rule_without_a_report(monkeypatch, tmp_path):
    from app.strategies import core as core_mod
    monkeypatch.setattr(core_mod, "_reports_dir", lambda: str(tmp_path))
    core_mod._LAB.update(mtime=None, report=None)
    _settings(monkeypatch, core_selection="auto", core_sizing="auto")
    sel, siz, why = core_mod.CoreBook.modes()
    assert (sel, siz) == ("trend", "equal") and "no daily-lab report" in why
