"""The online model's trust must come from skill, not from the market's
direction: replays of how live trading scores predictions."""
import numpy as np


def _market(days=400, seed=0, N=21):
    """Hourly returns for N correlated coins (zero drift), 24h forward labels."""
    rng = np.random.default_rng(seed)
    H, hours = 24, days * 24
    mkt = rng.normal(0, 0.006, hours)
    r = mkt[:, None] + rng.normal(0, 0.0045, (hours, N))
    lp = np.cumsum(r, axis=0)
    return np.exp(lp[H:] - lp[:-H]) - 1                 # (hours-24, N)


def _replay(fwd, predict, keep_every=24, start_day=30):
    """Score every coin's prediction through TinyMLP.update with cluster = label
    window, as live trading does; return the trust over time. Training is
    stubbed out: only the scoring path matters here."""
    from app.learn.online_model import N_IN, TinyMLP, trust
    m = TinyMLP()
    m._sgd = lambda x, target: 0.0
    m._observe_features = lambda x: None
    m._sample_replay_idx = lambda: 0
    x = [0.0] * N_IN
    out = []
    for t in range(len(fwd)):
        for j, y in enumerate(fwd[t]):
            m.update(x, y, pred_at_record=predict(t, j, y), cluster=t // 24)
        if t % keep_every == 0 and t >= 24 * start_day:
            out.append(trust(m.stats()))
    return np.array(out)


def test_zero_skill_models_are_not_trusted():
    fwd = _market()
    always_up = _replay(fwd, lambda t, j, y: 1.0)
    assert always_up.max() == 0.0                    # drift alone can never be skill
    rng = np.random.default_rng(1)
    coin = _replay(fwd, lambda t, j, y: rng.choice([-1.0, 1.0]))
    assert (coin > 0).mean() <= 0.05


def test_a_daily_market_call_with_no_skill_is_not_trusted():
    """One coin-flip call per day shared by every coin: a day's ~400 scored
    predictions ride one market move, so they are one observation. Only the
    clustered SE keeps this from reading as skill: with an iid or zero SE,
    3-4 of these 6 seeds are trusted on 7-56% of days."""
    for seed in range(6):
        fwd = _market(seed=seed)
        calls = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(fwd) // 24 + 1)
        tr = _replay(fwd, lambda t, j, y, c=calls: c[t // 24])
        assert (tr > 0).mean() <= 0.05, seed


def test_a_model_that_calls_the_market_gets_trusted_with_enough_evidence():
    fwd = _market(days=300, seed=2)
    rng = np.random.default_rng(3)
    calls = {}

    def predict(t, j, y):            # right about the day's direction 75% of the time
        day = t // 24
        if day not in calls:
            calls[day] = rng.random() < 0.75
        mkt_up = np.median(fwd[t]) > 0
        return (1.0 if mkt_up else -1.0) * (1 if calls[day] else -1)
    tr = _replay(fwd, predict, start_day=1)
    assert tr[-30:].mean() > 0.5
    assert tr[:15].max() == 0.0          # first 15 days: < 20 label windows, no trust yet


def test_old_rule_would_have_trusted_the_zero_skill_model():
    """Documents the bug the skill metric fixes."""
    fwd = _market()
    acc, window = [], []
    for t in range(len(fwd)):
        for y in fwd[t]:
            if abs(y) >= 0.006:
                window.append(1 if y > 0 else 0)
        window = window[-300:]
        if t > 500 and t % 24 == 0:
            acc.append(np.mean(window))
    assert (np.array(acc) > 0.6).mean() > 0.25         # old trust = (acc - .5) / .1 >= 1
