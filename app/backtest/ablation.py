"""Learner ablation — does each learning component actually help?

Runs the hourly replay (app/backtest/replay.py) once with NO learners (the
baseline) and then once per learner switched ON, each starting from a fresh,
empty state and learning causally as the replay walks forward:

  bandit        Thompson strategy weights. Learns from the per-bar signal
                stream (each sleeve's vote scored on its NET 24h forward
                return once that return has fully happened) and from the
                replay's own closed trades (vote attribution, side-aware) —
                the same two feeds as live (loop.run / on_trade_closed).
  direction     DirectionLearner: regime/side bias + conformal HTF veto,
                learning from the replay's closed trades.
  exit_advisor  predictive loss-cut, learning from counterfactual moves
                scored `exit_horizon_sec` later.
  rl_risk       Q-learning risk scale (floored at the live conviction floor),
                learning from replay equity.
  online_ml     the online MLP committee as an extra sleeve vote, trained in
                time order on matured 24h labels (as live: gated on measured
                accuracy > 50%, reset when broken). Its predictions don't
                depend on the portfolio, so they're computed once and shared.
  all           every learner above together.

The existing optional layers (trade filter, chop filter) are scored the same
way for comparison.

Scoring: each variant is ONE continuous run over the whole window (a learner
needs its history), then the equity curve is split at the midpoint. A learner
"helps" only if it beats the baseline in BOTH halves (averaged over seeds for
the stochastic ones); one good half is regime luck.

Not testable here (no history to replay them on): evolved (GA champions are
fitted on the same history), llm, model, discovered, derivatives,
exit_throttle (acts on live-only pattern exits).
"""
from __future__ import annotations
import math
import random
import statistics
import time
from collections import deque
from types import SimpleNamespace

LEARNERS = ("bandit", "direction", "exit_advisor", "rl_risk", "online_ml")
# learners whose result depends on a random seed (Thompson draws, epsilon-
# greedy, network initialization): scored over several seeds, and EVERY seed
# must beat the baseline in both halves.
STOCHASTIC = {"bandit", "rl_risk", "online_ml"}
NOT_TESTABLE = {
    "evolved": "GA champions are evolved on this same history (in-sample)",
    "llm": "no historical LLM opinions",
    "model": "no historical crypto_ml_lab predictions",
    "discovered": "no historical discovered-strategy records",
    "derivatives": "no funding/open-interest history",
    "exit_throttle": "throttles pattern exits, which the replay doesn't have",
}
BAR_SEC = 3600
LEARN_CYCLES_PER_BAR = 20           # live learner.run cadence: every ~3 min


def _tv(key):
    from ..tunables import tv
    return tv(key)


# --------------------------------------------------------------- online ML
def ml_track(cache, horizon_bars=None, seed=7, progress=None):
    """Walk the cached bars in time order with a FRESH committee, exactly like
    live: predict every coin each bar, train on a sample only once its label
    horizon has passed, reset a head stuck at/below a coin flip.

    Returns {(t, product): (vote, mean_pred)} for bars where the committee was
    live (warmed up and measured accuracy > 50%), plus a summary."""
    from ..learn import online_model as om
    from ..learn.loop import ML_HORIZON_SEC
    from ..signals.engine import _clip
    random.seed(seed)
    H = horizon_bars or int(ML_HORIZON_SEC // BAR_SEC)
    C, pos_of = cache["C"], cache["pos_of"]
    committee = om.Committee(n_members=3, base_seed=seed)
    pending, out = deque(), {}
    n_train = n_resets = live_bars = 0
    acc_trace = []
    ts = sorted(cache["bars"])
    for k, t in enumerate(ts):
        # train matured samples (label fully known at t)
        while pending and pending[0][0] <= t:
            ready, p, i, x, pred = pending.popleft()
            fwd = C[p][i + H][4] / C[p][i][4] - 1
            committee.observe_outcome(list(x), fwd)
            committee.update(list(x), fwd, pred_at_record=pred,
                             cluster=int(ready // (H * BAR_SEC)))
            n_train += 1
        st = committee.primary.stats()
        acc = st["directional_accuracy"]
        if om.confidently_broken(st):
            committee.reset()
            n_resets += 1
            acc = None
        ext = cache["bars"][t][4]
        tr = om.trust(st)
        live = committee.primary.n_updates >= 40 and tr > 0
        live_bars += live
        if k % 24 == 0:
            acc_trace.append(acc)
        for p, x in ext["x"].items():
            i = pos_of[p].get(t)
            if i is None:
                continue
            xl = list(x)
            if live:
                u = committee.predict_with_uncertainty(xl)
                out[(t, p)] = (_clip(u["mean"] * 1.3) * tr * u["confidence"], u["mean"])
            if i + H < len(C[p]):
                pred = committee.predict(xl) if committee.primary.n_updates >= 10 else 0.0
                pending.append((t + H * BAR_SEC, p, i, x, pred))
        if progress and k % 500 == 0:
            progress(f"online_ml track {k}/{len(ts)} bars")
    scored = [a for a in acc_trace if a is not None]
    return out, {"samples_trained": n_train, "resets": n_resets,
                 "live_pct_of_bars": round(100 * live_bars / max(1, len(ts)), 1),
                 "median_dir_accuracy": round(statistics.median(scored), 3) if scored else None}


def _workers(n_jobs):
    """Processes for independent jobs: CRYPTOMIND_ABLATION_WORKERS, else one
    per job up to (CPUs - 1). 1 = run serially."""
    import os
    try:
        w = int(os.environ.get("CRYPTOMIND_ABLATION_WORKERS", "0"))
    except ValueError:
        w = 0
    if w <= 0:
        w = max(1, (os.cpu_count() or 1) - 1)
    return max(1, min(w, n_jobs))


_WORKER_CACHE = None


def _init_ml_worker(payload):
    global _WORKER_CACHE
    _WORKER_CACHE = payload


def _ml_track_job(seed):
    return ml_track(_WORKER_CACHE, seed=seed)


def ml_tracks(cache, seeds, progress=None, workers=None):
    """`ml_track` for seeds 7, 8, ... — the dominant cost of an ablation
    (a pure-Python MLP, ~96% of the run). Seeds are independent and each
    seeds its own random state, so they run in parallel processes with results
    identical to the serial loop (tests/test_speed_parity.py). Falls back to
    serial on any multiprocessing failure."""
    say = progress or (lambda m: None)
    seed_list = [7 + s for s in range(seeds)]
    n = _workers(len(seed_list)) if workers is None else max(1, min(workers, len(seed_list)))
    if n > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            payload = {k: cache[k] for k in ("C", "pos_of", "bars")}
            say(f"training the online model through history: {len(seed_list)} seeds "
                f"on {n} processes (pure-Python MLP)")
            with ProcessPoolExecutor(max_workers=n, initializer=_init_ml_worker,
                                     initargs=(payload,)) as ex:
                return list(ex.map(_ml_track_job, seed_list))
        except Exception as e:
            say(f"parallel online-model training failed ({e}); running serially")
    out = []
    for k, s in enumerate(seed_list):
        say(f"training the online model through history, seed {k + 1}/{len(seed_list)} "
            "(slow: pure-Python MLP)")
        out.append(ml_track(cache, seed=s, progress=say))
    return out


# --------------------------------------------------------------- hooks
class LearnerHooks:
    """Replay hooks (see replay.run_replay `hooks`) that switch on a chosen
    set of learners, each fresh, learning only from what has already happened
    at the current bar. With nothing enabled it reproduces the plain replay."""

    def __init__(self, cache, enabled=(), ml=None, shorts=True):
        from ..signals.engine import STRATEGIES
        from ..learn.bandit import RegimeBandit
        from ..learn.direction import DirectionLearner
        from ..learn.exit_advisor import ExitAdvisor
        from ..learn.rl_risk import QRiskAgent, CONVICTION_FLOOR
        from ..learn.loop import ML_HORIZON_SEC
        from ..execution.costs import round_trip_cost
        from ..config import DISABLED_STRATEGIES
        from .replay import HISTORY_SLEEVES
        self.on = set(enabled)
        self.C, self.pos_of = cache["C"], cache["pos_of"]
        self.ml = ml or {}
        self.shorts = shorts
        self.strategies = list(STRATEGIES)
        self.history_sleeves = {s for s in HISTORY_SLEEVES if s not in DISABLED_STRATEGIES}
        self.weights = {k: 1.0 / len(STRATEGIES) for k in STRATEGIES}   # engine default
        self.bandit = RegimeBandit(self.strategies)
        self.dl = DirectionLearner()
        self.dl.enabled = True         # tested regardless of the live switch
        self.xa = ExitAdvisor()
        self.rl = QRiskAgent()
        self.rl_floor = CONVICTION_FLOOR
        self.rl_scale = 1.0
        self.H = int(ML_HORIZON_SEC // BAR_SEC)
        self.rt_cost = round_trip_cost()
        self.gate = _tv("min_confidence")
        self.veto_align = _tv("mtf_veto_align")
        self.gamma = _tv("bandit_decay_gamma") ** LEARN_CYCLES_PER_BAR
        self.sig_w, self.sig_clip = _tv("signal_learn_weight"), _tv("signal_learn_clip")
        self.trade_w, self.loss_mult = _tv("trade_weight"), _tv("loss_lesson_mult")
        self.sig_pending = []          # (ready_t, regime, sleeve, product, i, dir, conf)
        self.ext = None
        self.votes = {}                # product -> votes used this bar
        self.closes = []               # (t, net)
        self.peak = 0.0
        self.streak = 0
        self.closed_this_bar = False
        self.n_cuts = self.n_flips = self.n_vetoes = self.n_changed = 0
        self.fixed_mult = 1.0          # constant size multiplier (size-matched control)
        self.size_used = []            # multiplier applied at each entry

    # ---- signals: re-mix votes with learned weights / ML / direction learner
    def signals(self, t, bd):
        sl, ext = bd[0], bd[4]
        self.ext = ext
        if ext is None:
            return sl
        if "bandit" in self.on:
            self._queue_signal_lessons(t, ext)
        if not self.on & {"bandit", "direction", "online_ml"}:
            self.votes = ext["raw"]
            return sl
        from ..signals.engine import combine
        out, self.votes = [], {}
        for p, d0, c0, a0, price, atr in sl:
            raw = dict(ext["raw"].get(p, {}))
            if "online_ml" in self.on:
                v = self.ml.get((t, p))
                if v and abs(v[0]) > 0:
                    raw["ml"] = v[0]
            self.votes[p] = raw
            active = {n: s for n, s in raw.items() if abs(s) > 0.05}
            comp, conf = combine(active, self.weights) if active else (0.0, 0.0)
            mtf = ext["mtf"].get(p, 0.0)
            if "direction" in self.on:
                comp, flipped = self.dl.adjust(ext["regime"], comp, mtf)
                self.n_flips += flipped
            if conf == 0.0 and comp != 0.0:
                conf = min(1.0, abs(comp) * 0.5)
            d = 1 if comp > 0 else -1
            dir_ok = d > 0 or self.shorts
            if "direction" in self.on:
                vetoed, _ = self.dl.veto(d, mtf)
                self.n_vetoes += vetoed
            else:
                vetoed = (d < 0 and mtf >= self.veto_align) or (d > 0 and mtf <= -self.veto_align)
            act = conf >= self.gate and dir_ok and not vetoed
            conf = round(conf, 3)
            if (d, act) != (d0, a0):
                self.n_changed += 1
            out.append((p, d, conf, act, price, atr))
        return out

    def _queue_signal_lessons(self, t, ext):
        for p, raw in ext["raw"].items():
            i = self.pos_of[p].get(t)
            if i is None or i + self.H >= len(self.C[p]):
                continue
            for name, s in raw.items():
                if abs(s) > 0.3:
                    self.sig_pending.append((t + self.H * BAR_SEC, ext["regime"], name, p, i,
                                             1 if s > 0 else -1, abs(s)))

    # ---- per bar: mature lessons, re-allocate, RL step, exit-advisor scoring
    def on_bar(self, t, eqv, book):
        if "bandit" in self.on:
            k = 0
            while k < len(self.sig_pending) and self.sig_pending[k][0] <= t:
                _, reg, name, p, i, d, c = self.sig_pending[k]
                fwd = (self.C[p][i + self.H][4] / self.C[p][i][4] - 1) * d
                net = max(-self.sig_clip, min(self.sig_clip, fwd - self.rt_cost))
                self.bandit.update(reg, name, net * c * self.sig_w)
                k += 1
            del self.sig_pending[:k]
            self.bandit.decay(gamma=self.gamma)
            self._allocate(t)
        if "rl_risk" in self.on and self.ext is not None:
            self.peak = max(self.peak, eqv)
            dd = 1 - eqv / self.peak if self.peak else 0.0
            tradable = bool(book) or self.closed_this_bar
            a = self.rl.act({"trend": self.ext["trend"], "vol_state": "normal"},
                            dd, self.streak, eqv, tradable=tradable)
            self.rl_scale = max(self.rl_floor, a)
        if "exit_advisor" in self.on:
            self.xa.score_pending(self._price_at, now=t)
        self.closed_this_bar = False

    def _allocate(self, t):
        from ..learn.loop import Learner
        ext = self.ext or {}
        live = set(self.history_sleeves)
        # ml is silent (weight 0, no floor) unless the committee is voting now
        if "online_ml" in self.on and any((t, p) in self.ml for p in ext.get("x", {})):
            live.add("ml")
        silent = set(self.strategies) - live
        shim = SimpleNamespace(bandit=self.bandit, weights=self.weights)
        self.weights = Learner._allocate(shim, ext.get("regime", "unknown"), silent)

    def _price_at(self, p, ts):
        i = self.pos_of.get(p, {}).get(int(ts) // BAR_SEC * BAR_SEC)
        return self.C[p][i][4] if i is not None else None

    # ---- exits, sizing, bookkeeping
    def exit(self, t, p, b, x, atr):
        if "exit_advisor" not in self.on or self.ext is None:
            return None
        side = b["side"]
        unreal = side * (x / b["e"] - 1)
        atr_pct = (atr or 0.0) / x if x else 0.0
        ml_pos = 0.0
        if "online_ml" in self.on:
            v = self.ml.get((t, p))
            if v:
                ml_pos = side * v[1] * 0.004          # same scaling as live
        regime = {"trend": self.ext["trend"], "vol_state": "normal"}
        self.xa.record(p, side, x, unreal, ml_pos, regime, atr_pct, ts=t)
        action, _, _ = self.xa.decide(p, side, unreal, ml_pos, regime, atr_pct)
        if action == "cut":
            self.n_cuts += 1
            return "exit-advisor cut"
        return None

    def size_mult(self, t):
        m = self.rl_scale if "rl_risk" in self.on else self.fixed_mult
        self.size_used.append(m)
        return m

    def on_entry(self, t, p, side):
        ext = self.ext or {}
        return {"regime": ext.get("regime", "unknown"),
                "mtf": ext.get("mtf", {}).get(p, 0.0),
                "votes": dict(self.votes.get(p, {}))}

    def on_close(self, t, p, b, net, why):
        self.closes.append((t, net))
        self.closed_this_bar = True
        self.streak = self.streak + 1 if net < 0 else 0
        meta = b.get("meta") or {}
        if "direction" in self.on:
            self.dl.on_trade_closed({
                "regime_at_entry": meta.get("regime", "unknown"), "side": b["side"],
                "qty": b["n"] / b["e"], "entry": b["e"], "pnl": b["n"] * net,
                "mtf_at_entry": meta.get("mtf")})
        if "bandit" in self.on:
            votes = meta.get("votes") or {}
            total = sum(abs(v) for v in votes.values()) or 1e-9
            for strat, v in votes.items():
                aligned = net if v * b["side"] > 0 else -net
                w = self.trade_w * (self.loss_mult if aligned < 0 else 1.0)
                self.bandit.update(meta.get("regime", "unknown"), strat,
                                   aligned * abs(v) / total * w)

    def capture(self):
        """Replay-trained state of each enabled learner, in the same shape the
        live learners persist (see gate.warm_start)."""
        out = {}
        if "direction" in self.on:
            out["direction"] = self.dl.capture()
        if "exit_advisor" in self.on:
            out["exit_advisor"] = self.xa.capture()
        if "rl_risk" in self.on:
            out["rl_risk"] = {"q": {"||".join(map(str, k)): v for k, v in self.rl.q.items()},
                              "n_updates": self.rl.n_updates}
        if "bandit" in self.on:
            out["bandit"] = {"arms": {f"{r}||{s}": list(v)
                                      for (r, s), v in self.bandit.arms.items()}}
        return out

    def stats(self):
        out = {"signals_changed": self.n_changed}
        if "bandit" in self.on:
            out["final_weights"] = {k: round(v, 3) for k, v in self.weights.items() if v > 0.001}
        if "direction" in self.on:
            out.update(direction_flips=self.n_flips, direction_vetoes=self.n_vetoes)
        if "exit_advisor" in self.on:
            out["exit_cuts"] = self.n_cuts
        if "rl_risk" in self.on:
            out["rl_updates"] = self.rl.n_updates
        return out


# --------------------------------------------------------------- scoring
def _halves(r, hooks):
    eq, ts = r["_equity"], r["_ts"]
    m = len(eq) // 2

    def seg(a, b):
        e = eq[a:b]
        peak, dd = e[0], 0.0
        for v in e:
            peak = max(peak, v)
            dd = min(dd, v / peak - 1)
        n = [net for t, net in hooks.closes if ts[a] < t <= ts[b - 1]]
        return {"return_pct": round((e[-1] / e[0] - 1) * 100, 2),
                "max_drawdown_pct": round(dd * 100, 1), "trades": len(n),
                "avg_net_bps": round(1e4 * statistics.mean(n), 1) if n else None}
    return {"full": seg(0, len(eq)), "first_half": seg(0, m + 1),
            "second_half": seg(m, len(eq))}


def _run(candles, cache, enabled=(), ml=None, seed=0, shorts=None, fixed_mult=1.0, **kw):
    from .replay import run_replay
    from .. import settings as app_settings
    if shorts is None:
        try:
            shorts = bool(app_settings.get("allow_shorts"))
        except Exception:
            shorts = True
    random.seed(seed)
    h = LearnerHooks(cache, enabled, ml=ml, shorts=shorts)
    h.fixed_mult = fixed_mult
    r = run_replay(candles, 0.0, 1.0, cache=cache, hooks=h, extras=True, shorts=shorts, **kw)
    if not r.get("ok"):
        return r, h
    return _halves(r, h), h


def run_ablation(candles, seeds=3, learners=LEARNERS, include_all=True,
                 include_filters=True, progress=None):
    """Baseline vs each learner (and all together, and the optional filters).
    Returns a JSON-able report; see the module docstring for the method."""
    from .replay import run_replay, filter_samples
    say = progress or (lambda m: None)
    t0 = time.time()
    cache = {}
    say("computing per-bar signals (one pass, ~1 min per year)")
    warm = run_replay(candles, 0.0, 1.0, cache=cache, extras=True, persist=True)
    if not warm.get("ok"):
        return {"ok": False, "error": warm.get("error")}
    base, _ = _run(candles, cache)
    states = {}
    from ..data.store import fingerprint
    out = {"ok": True, "ran_at": time.time(), "data_version": fingerprint(candles),
           "days": warm["days"], "products": warm["products"],
           "from_ts": warm["from_ts"], "to_ts": warm["to_ts"],
           "baseline": base, "variants": {}, "not_testable": NOT_TESTABLE}

    mls, ml_summary = [None] * seeds, None
    if "online_ml" in learners:
        tracks = ml_tracks(cache, seeds, progress=say)
        mls = [m for m, _ in tracks]
        ml_summary = {"per_seed": [sm for _, sm in tracks]}

    def score(name, enabled, extra_kw=None, stochastic=False):
        say(f"variant: {name}")
        runs, stats, used = [], None, []
        for s in (range(seeds) if stochastic else [0]):
            r, h = _run(candles, cache, enabled, ml=mls[s], seed=s, **(extra_kw or {}))
            runs.append(r)
            stats = h.stats()
            used += h.size_used
            if s == 0 and len(enabled) == 1:
                states.update(h.capture())
        mean = {w: {k: round(statistics.mean(r[w][k] for r in runs), 2)
                    for k in ("return_pct", "max_drawdown_pct", "trades")}
                for w in ("full", "first_half", "second_half")}
        delta = {w: round(mean[w]["return_pct"] - base[w]["return_pct"], 2)
                 for w in ("full", "first_half", "second_half")}
        seeds_both = sum(1 for r in runs
                         if all(r[w]["return_pct"] > base[w]["return_pct"]
                                for w in ("first_half", "second_half")))
        # every seed must beat the baseline in both halves (one seed for the
        # deterministic learners, where this is the same as the mean)
        helps = seeds_both == len(runs)
        v = out["variants"][name] = {
            "seeds": len(runs), "mean": mean, "delta_vs_baseline_pct_pts": delta,
            "helps_in_both_halves": helps,
            "seeds_helping_in_both_halves": f"{seeds_both}/{len(runs)}",
            "learner_stats": stats}
        # SIZE-MATCHED CONTROL: on a losing strategy, betting smaller "helps"
        # with no skill at all. A learner that changes position size must also
        # beat the baseline run at a FIXED multiplier equal to its average one.
        avg = statistics.mean(used) if used else 1.0
        if abs(avg - 1.0) > 0.02:
            ctrl, _ = _run(candles, cache, (), fixed_mult=avg, **(extra_kw or {}))
            # EVERY seed must beat the (deterministic) control in both halves:
            # a mean edge of a point or two is inside seed-to-seed noise.
            beats = all(r[w]["return_pct"] > ctrl[w]["return_pct"]
                        for r in runs for w in ("first_half", "second_half"))
            v["size_matched_control"] = {
                "avg_size_mult": round(avg, 3),
                "return_pct": {w: ctrl[w]["return_pct"]
                               for w in ("full", "first_half", "second_half")},
                "beats_control_in_both_halves": beats}
            v["helps_in_both_halves"] = helps and beats

    for name in learners:
        score(name, {name}, stochastic=name in STOCHASTIC)
    if include_all and len(learners) > 1:
        score("all", set(learners), stochastic=bool(STOCHASTIC & set(learners)))
    if ml_summary:
        out["variants"]["online_ml"]["learner_stats"].update(ml_summary)
    if include_filters:
        from ..learn.trade_filter import walk_forward_probs
        say("trade filter walk-forward")
        pr, bases = walk_forward_probs(filter_samples(cache))
        probs = {k: (v, bases[k]) for k, v in pr.items()}
        score("trade_filter (existing)", (), {"filt": probs})
        score("chop_filter (existing)", (), {"chop": True})
    out["verdict"] = {n: ("helps" if v["helps_in_both_halves"] else "does not help")
                      for n, v in out["variants"].items()}
    out["elapsed_sec"] = round(time.time() - t0, 1)
    out["_states"] = states            # popped by the caller -> gate.save_states
    return out
