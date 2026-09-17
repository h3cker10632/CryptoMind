"""Phase 1 learning upgrades:
  * bandit forgetting (discounted posteriors + prune)
  * online model: prioritized replay + online feature standardization
  * concept-drift detection (Page-Hinkley)
  * GA warm-start from the standing champion
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------- bandit forgetting ----------------
def test_bandit_decay_shrinks_evidence():
    from app.learn.bandit import RegimeBandit
    b = RegimeBandit(["trend", "meanrev"])
    for _ in range(50):
        b.update("bull", "trend", 0.002)
    n_before = b.arms[("bull", "trend")][0]
    mean_before = b.arms[("bull", "trend")][1]
    b.decay(gamma=0.9)
    n_after, mean_after, _ = b.arms[("bull", "trend")]
    assert n_after < n_before                      # evidence shrank
    # idle mean forgets toward 0 (same gamma) — a stale +edge must not
    # coast forever just because n is decaying slowly
    assert abs(mean_after) < abs(mean_before)
    assert mean_after * mean_before > 0            # sign preserved


def test_bandit_decay_prunes_idle_arms():
    from app.learn.bandit import RegimeBandit
    b = RegimeBandit(["trend"])
    b.update("rare_regime", "trend", 0.001)        # n=1
    # decay hard many times -> arm should fall below prune threshold and vanish
    for _ in range(200):
        b.decay(gamma=0.9)
    assert ("rare_regime", "trend") not in b.arms


def test_recent_evidence_outweighs_stale_after_decay():
    from app.learn.bandit import RegimeBandit
    b = RegimeBandit(["a", "b"])
    # 'a' was great long ago; 'b' is good now. Decay between the two epochs.
    for _ in range(40):
        b.update("bull", "a", 0.003)
    for _ in range(140):
        b.decay(gamma=0.99)          # ~ a couple half-lives
    for _ in range(40):
        b.update("bull", "b", 0.003)
    na = b.arms[("bull", "a")][0]
    nb = b.arms[("bull", "b")][0]
    assert nb > na, "stale winner did not lose effective weight to the recent one"


def test_idle_mean_shrinks_each_decay_cycle():
    """Each learning cycle must move an idle posterior toward 0, not keep a
    frozen +8bps mean while n slowly decays (the ghost-evolved bug)."""
    from app.learn.bandit import RegimeBandit
    b = RegimeBandit(["evolved"])
    for _ in range(80):
        b.update("sideways", "evolved", 0.000855)   # leftover +8.55 bps
    abs_means = []
    for _ in range(8):
        b.decay(gamma=0.9)
        abs_means.append(abs(b.arms[("sideways", "evolved")][1]))
    assert all(abs_means[i] > abs_means[i + 1] for i in range(len(abs_means) - 1))
    assert abs_means[-1] < 0.5 * abs_means[0]


def test_drop_strategy_clears_every_regime():
    from app.learn.bandit import RegimeBandit
    b = RegimeBandit(["evolved", "trend"])
    b.update("bull", "evolved", 0.002)
    b.update("bear", "evolved", -0.001)
    b.update("bull", "trend", 0.001)
    n = b.drop_strategy("evolved")
    assert n == 2
    assert all(s != "evolved" for (_, s) in b.arms)
    assert ("bull", "trend") in b.arms


# ---------------- online model ----------------
def test_prioritized_replay_populates_priorities():
    from app.learn.online_model import TinyMLP, N_IN
    m = TinyMLP()
    x = [0.1] * N_IN
    for _ in range(20):
        m.update(x, 0.01, pred_at_record=0.0)
    assert len(m.replay) == len(m.replay_pr)
    assert all(p > 0 for p in m.replay_pr)


def test_online_standardization_tracks_feature_stats():
    from app.learn.online_model import TinyMLP, N_IN
    m = TinyMLP()
    # feed a feature with a strong offset; running mean should move toward it
    for _ in range(60):
        x = [5.0] + [0.0] * (N_IN - 1)
        m.update(x, 0.0)
    assert abs(m.feat_mean[0] - 5.0) < 0.5
    # standardized value of the mean input is ~0
    z = m._standardize([5.0] + [0.0] * (N_IN - 1))
    assert abs(z[0]) < 0.5


# ---------------- concept drift ----------------
def test_page_hinkley_flags_rising_error():
    from app.learn.drift import PageHinkley
    ph = PageHinkley(delta=0.005, lambda_=0.3)
    # stable low error -> no drift
    fired_low = any(ph.add(0.05) for _ in range(100))
    assert not fired_low
    # a sustained jump in error should trip it
    fired_high = any(ph.add(1.5) for _ in range(50))
    assert fired_high
    assert ph.n_events >= 1


# ---------------- GA warm-start ----------------
def test_ga_warm_starts_from_champion(monkeypatch):
    from app.learn import evolution as ev
    e = ev.Evolution(pop_size=12, generations=1, seed=1)
    champ = {"ema_fast": 8, "ema_slow": 30, "rsi_buy": 30, "rsi_sell": 70,
             "breakout_n": 20, "stop_atr": 2.0, "take_atr": 4.0,
             "mom_w": 0.6, "short_w": 0.0}
    e.champions["BTC-USD"] = champ

    seen = {"has_champ": False}
    real_sim = ev.simulate
    def spy(genome, candles, **k):
        if all(genome.get(kk) == vv for kk, vv in champ.items()):
            seen["has_champ"] = True
        return real_sim(genome, candles, **k)
    monkeypatch.setattr(ev, "simulate", spy)

    # minimal synthetic candle history [ts, low, high, open, close, vol]
    candles = []
    px = 100.0
    for i in range(300):
        px *= 1 + (0.001 if i % 3 else -0.0008)
        candles.append([i * 300, px * 0.99, px * 1.01, px, px, 1000.0])
    e.evolve(candles, product="BTC-USD")
    assert seen["has_champ"], "GA did not seed the population with the champion"
