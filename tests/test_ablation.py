"""Learner ablation: the no-learner hooks reproduce the plain replay exactly,
each learner runs causally from a fresh state, and the report has the shape
the CLI relies on."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_replay_auto import _universe


def _cache(U):
    from app.backtest.replay import run_replay
    cache = {}
    run_replay(U, 0.0, 1.0, cache=cache, extras=True)
    return cache


def test_no_learner_hooks_reproduce_plain_replay():
    from app.backtest.replay import run_replay
    from app.backtest.ablation import LearnerHooks
    U = _universe(7, 500)
    plain = run_replay(U, 0.0, 1.0)
    cache = _cache(U)
    hooked = run_replay(U, 0.0, 1.0, cache=cache, extras=True,
                        hooks=LearnerHooks(cache, ()))
    for k in ("return_pct", "trades", "max_drawdown_pct", "win_rate_pct"):
        assert hooked[k] == plain[k], k
    assert len(hooked["_equity"]) == len(hooked["_ts"]) > 0


def test_remix_with_default_weights_matches_engine():
    """Re-mixing the cached raw votes with the engine's default weights and the
    cold-start direction learner must give the engine's own signals."""
    from app.backtest.ablation import LearnerHooks
    U = _universe(7, 400)
    cache = _cache(U)
    h = LearnerHooks(cache, {"direction"})
    for t in sorted(cache["bars"])[:150]:
        bd = cache["bars"][t]
        got = {x[0]: (x[1], x[3]) for x in h.signals(t, bd)}
        want = {x[0]: (x[1], x[3]) for x in bd[0]}
        assert got == want


def test_bandit_only_learns_from_matured_labels():
    from app.backtest.ablation import LearnerHooks
    U = _universe(7, 400)
    cache = _cache(U)
    h = LearnerHooks(cache, {"bandit"})
    ts = sorted(cache["bars"])
    for t in ts[:h.H]:                    # before any 24h label can exist
        h.signals(t, cache["bars"][t])
        h.on_bar(t, 100_000.0, {})
    assert h.bandit.arms == {}
    assert all(ready > ts[h.H - 1] for ready, *_ in h.sig_pending)


def test_learners_leave_live_singletons_alone():
    from app.backtest import ablation as ab
    from app.learn.direction import direction_learner
    from app.learn.exit_advisor import exit_advisor
    from app.learn.rl_risk import agent
    before = (direction_learner.n_updates, exit_advisor.n_updates, agent.n_updates)
    U = _universe(7, 400)
    cache = _cache(U)
    for name in ("bandit", "direction", "exit_advisor", "rl_risk"):
        r, _ = ab._run(U, cache, {name})
        assert r["full"]["trades"] >= 0
    assert (direction_learner.n_updates, exit_advisor.n_updates, agent.n_updates) == before


def test_run_ablation_report_shape():
    from app.backtest import ablation as ab
    rep = ab.run_ablation(_universe(7, 450), seeds=1,
                          learners=("bandit", "direction", "exit_advisor", "rl_risk"),
                          include_filters=False)
    assert rep["ok"], rep
    assert set(rep["variants"]) == {"bandit", "direction", "exit_advisor", "rl_risk", "all"}
    for v in rep["variants"].values():
        assert set(v["mean"]) == {"full", "first_half", "second_half"}
        assert isinstance(v["helps_in_both_halves"], bool)
    assert set(rep["verdict"]) == set(rep["variants"])
    assert "evolved" in rep["not_testable"]


def test_sizing_learner_gets_size_matched_control():
    """A learner that shrinks size must beat a FIXED shrink of the same average
    size, or it doesn't count as helping."""
    from app.backtest import ablation as ab
    rep = ab.run_ablation(_universe(7, 450), seeds=1, learners=("rl_risk",),
                          include_filters=False)
    v = rep["variants"]["rl_risk"]
    ctrl = v.get("size_matched_control")
    assert ctrl is not None and 0 < ctrl["avg_size_mult"] < 1.25
    if not ctrl["beats_control_in_both_halves"]:
        assert v["helps_in_both_halves"] is False
