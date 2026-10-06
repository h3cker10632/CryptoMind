"""Phase 3 model deep-dive:
  * quantile heads (P10/P50/P90) via pinball loss -> aleatoric band
  * committee of seeded members -> epistemic (disagreement) uncertainty
  * uncertainty feeds the risk-manager sizer (unsure -> smaller position)
  * persistence round-trips the whole committee incl. quantile heads
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn.online_model import TinyMLP, Committee, N_IN


from app.learn.online_model import TARGET_SCALE


def _train(net_or_committee, n=500, seed=0):
    r = random.Random(seed)
    for _ in range(n):
        f0 = r.gauss(0, 1)
        x = [f0] + [r.gauss(0, 1) for _ in range(N_IN - 1)]
        fwd = (0.5 * f0 + r.gauss(0, 0.1)) * TARGET_SCALE
        net_or_committee.update(x, fwd)
    return r


# ---------- quantile heads ----------
def test_predict_quantiles_ordered():
    m = TinyMLP()
    _train(m)
    lo, med, hi = m.predict_quantiles([1.5] + [0.0] * (N_IN - 1))
    assert lo <= med <= hi                      # band is always ordered
    assert hi - lo >= 0


def test_quantile_band_has_width_after_training_on_noise():
    m = TinyMLP()
    _train(m, n=600)
    # noisy target => the P10/P90 heads should separate (non-degenerate band)
    widths = [m.predict_quantiles([random.gauss(0, 1)] + [0.0] * (N_IN - 1))
              for _ in range(20)]
    avg_w = sum(hi - lo for lo, _, hi in widths) / len(widths)
    assert avg_w > 0.01


# ---------- committee ----------
def test_committee_members_are_distinct():
    c = Committee(n_members=3, base_seed=5)
    # different seeds -> different initial weights
    assert c.members[0].W2 != c.members[1].W2


def test_committee_mean_matches_member_average():
    c = Committee(n_members=3)
    _train(c, n=200)
    x = [1.0] + [0.0] * (N_IN - 1)
    expect = sum(m.predict(x) for m in c.members) / 3
    assert abs(c.predict(x) - expect) < 1e-9


def test_uncertainty_higher_out_of_distribution():
    c = Committee(n_members=3, base_seed=3)
    _train(c, n=600)
    u_in = c.predict_with_uncertainty([1.0] + [0.0] * (N_IN - 1))
    u_ood = c.predict_with_uncertainty([9.0] * N_IN)      # far from training data
    # members disagree more on unfamiliar inputs -> lower confidence there
    assert u_ood["epistemic"] > u_in["epistemic"]
    assert u_ood["confidence"] <= u_in["confidence"]
    assert 0.0 <= u_in["confidence"] <= 1.0


def test_committee_lr_boost_propagates():
    c = Committee(n_members=3)
    c.lr_boost = 2.5
    assert all(m.lr_boost == 2.5 for m in c.members)


# ---------- uncertainty feeds sizing ----------
def test_low_ml_confidence_shrinks_position():
    from app.risk.manager import risk
    rs = {"effective_risk_scale": 1.0}
    common = dict(equity=100_000.0, price=100.0, atr=2.0, confidence=0.8,
                  risk_status=rs, direction=1, product="BTC-USD")
    n_sure, _, _ = risk.size(ml_confidence=1.0, **common)
    n_unsure, _, _ = risk.size(ml_confidence=0.0, **common)
    assert n_unsure < n_sure                    # unsure model bets smaller
    # default (no ML arg) must equal the fully-confident case (no regression)
    n_default, _, _ = risk.size(**common)
    assert abs(n_default - n_sure) < 1e-6


# ---------- persistence round-trip ----------
def test_persistence_roundtrips_committee(tmp_path, monkeypatch):
    import app.persistence as P
    from app.learn import online_model as om
    from app import settings
    _train(om.committee, n=120, seed=2)
    x = [1.3] + [0.0] * (N_IN - 1)
    before = om.committee.predict(x)
    q_before = om.committee.members[-1].predict_quantiles(x)

    snap = tmp_path / "state.json"
    monkeypatch.setattr(P, "STATE_PATH", str(snap), raising=False)
    monkeypatch.setattr(settings, "get", lambda k, *a: True)  # carry_equity on
    assert P.save()

    # wipe the committee, then restore
    om.committee = om.Committee(n_members=3)
    om.model = om.committee.primary
    monkeypatch.setattr(P, "STATE_PATH", str(snap), raising=False)
    assert P.load()
    after = om.committee.predict(x)
    q_after = om.committee.members[-1].predict_quantiles(x)
    assert abs(after - before) < 1e-9
    assert all(abs(a - b) < 1e-9 for a, b in zip(q_after, q_before))
