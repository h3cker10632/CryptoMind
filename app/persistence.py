"""State persistence — snapshots the full system state to disk so paper
account, positions, learned models, Q-tables, bandit posteriors, GA champion
and discovered universe all survive restarts.

Saved to state.json next to the SQLite DB (atomic write via temp file).
"""
import json, os, time, tempfile
from collections import deque
from . import db
from .config import PRODUCTS

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state.json")
SAVE_EVERY_TICKS = 3          # ~once a minute at TICK_SEC=20


# ---------------- serialization helpers ----------------

def _tup_key(k, sep="||"):
    return sep.join(map(str, k))


def _save_dict_tupkeys(d):
    return {_tup_key(k): v for k, v in d.items()}


def _load_arms(d):
    out = {}
    for k, v in d.items():
        parts = k.split("||")
        out[(parts[0], parts[1])] = tuple(v)
    return out


def _load_q(d):
    out = {}
    for k, v in d.items():
        p = k.split("||")
        # state = (trend, vol, dd_bucket, streak_bucket)
        out[(p[0], p[1], int(p[2]), int(p[3]))] = list(v)
    return out


def _dump_mlp(m):
    """Full serialization of one TinyMLP member, incl. quantile heads and the
    online feature-standardization stats."""
    return {
        "W1": m.W1, "b1": m.b1, "W2": m.W2, "b2": m.b2,
        "gW1": m.gW1, "gb1": m.gb1, "gW2": m.gW2, "gb2": m.gb2,
        "qW": m.qW, "qb": m.qb, "gqW": m.gqW, "gqb": m.gqb,
        "n_updates": m.n_updates, "lr_boost": m.lr_boost,
        "replay": [[x, t] for x, t in list(m.replay)[-1500:]],
        "replay_pr": list(m.replay_pr)[-1500:],
        "acc_window": list(m.acc_window),
        "loss_window": list(m.loss_window),
        "feat_n": m.feat_n, "feat_mean": m.feat_mean, "feat_M2": m.feat_M2,
    }


def _load_mlp(m, d):
    """Restore one TinyMLP member from a dict; only if feature dims match.
    Tolerates snapshots written before quantile heads existed (keeps fresh
    heads in that case)."""
    if not d.get("W1") or len(d["W1"][0]) != len(m.W1[0]):
        return False
    m.W1, m.b1 = d["W1"], d["b1"]
    m.W2, m.b2 = d["W2"], d["b2"]
    m.gW1, m.gb1 = d["gW1"], d["gb1"]
    m.gW2, m.gb2 = d["gW2"], d["gb2"]
    if d.get("qW") and len(d["qW"]) == len(m.qW):
        m.qW, m.qb = d["qW"], d["qb"]
        m.gqW, m.gqb = d.get("gqW", m.gqW), d.get("gqb", m.gqb)
    m.n_updates = d.get("n_updates", 0)
    m.lr_boost = d.get("lr_boost", 1.0)
    m.replay = deque([(x, t) for x, t in d.get("replay", [])],
                     maxlen=m.replay.maxlen)
    pr = d.get("replay_pr", [])
    if len(pr) != len(m.replay):
        pr = [1e-2] * len(m.replay)
    m.replay_pr = deque(pr, maxlen=m.replay_pr.maxlen)
    m.acc_window = deque(d.get("acc_window", []), maxlen=m.acc_window.maxlen)
    m.loss_window = deque(d.get("loss_window", []), maxlen=m.loss_window.maxlen)
    if len(d.get("feat_mean", [])) == len(m.feat_mean):
        m.feat_n = d.get("feat_n", 0)
        m.feat_mean = d.get("feat_mean", m.feat_mean)
        m.feat_M2 = d.get("feat_M2", m.feat_M2)
    return True


# ---------------- capture ----------------

def capture():
    from .execution.paper import broker
    from .risk.manager import risk
    from .learn.loop import learner
    from .learn.online_model import model, committee
    from .learn.rl_risk import agent as rl_agent
    from .learn.evolution import evolution
    from .data.universe import universe

    return {
        "version": 3,
        "saved_at": time.time(),
        "broker": {
            "cash": broker.cash,
            "realized_pnl": broker.realized_pnl,
            # copy the dict/list first (feeds/threads mutate these live)
            "positions": {p: dict(pos) for p, pos in list(broker.positions.items())},
            "closed_trades": [dict(t) for t in list(broker.closed_trades)[-200:]],
        },
        "risk": {
            "peak_equity": risk.peak_equity,
            "kill_arm_peak": risk.kill_arm_peak,
            "day_start_equity": risk.day_start_equity,
            "day_start_ts": risk.day_start_ts,
            "killed": risk.killed,
            "kill_reason": risk.kill_reason,
            "kill_ts": risk.kill_ts,
            "halted_today": risk.halted_today,
            "halt_reason": risk.halt_reason,
            "day_index": getattr(risk, "day_index", None),
            "cooldowns": risk.cooldowns,
            "risk_scale": risk.risk_scale,
            "consecutive_losses": risk.consecutive_losses,
        },
        "learner": {
            "weights": learner.weights,
            "bandit_arms": _save_dict_tupkeys(
                {k: list(v) for k, v in learner.bandit.arms.items()}),
            "trade_attributions": learner.trade_attributions,
            # Pending ML training samples + price history are IN-MEMORY working
            # state, but the online model only labels a sample ML_HORIZON_SEC
            # (30 min) after it was recorded. The process restarts far more often
            # than that, so if these aren't persisted every pending label is
            # wiped before it matures and the model NEVER trains (n_updates=0).
            "pending_ml": [[ts, p, x, pred]
                           for (ts, p, x, pred) in list(learner.pending_ml)],
            "price_history": {p: hist[-400:]
                              for p, hist in learner.price_history.items()},
        },
        # `model` = the committee's PRIMARY member, kept under the same key/
        # schema (now incl. quantile heads) so old snapshots keep loading.
        "model": _dump_mlp(model),
        # `committee` = every member (primary included) for the full ensemble.
        "committee": [_dump_mlp(m) for m in committee.members],
        "rl": {
            "q": _save_dict_tupkeys(rl_agent.q),
            "eps": rl_agent.eps,
            "n_updates": rl_agent.n_updates,
        },
        "hedge": _capture_hedge(),
        "evolution": {
            "champions": evolution.champions,
            "champion_portfolios": evolution.champion_portfolios,
            "champion_reports": evolution.champion_reports,
            "last_attempt": evolution.last_attempt,
        },
        "universe": {
            "products": list(PRODUCTS),
            "mention_heat": universe.mention_heat,
            "name_to_sym": universe.name_to_sym,
            # sets aren't JSON-serializable → store source tags as sorted lists
            "sources": {k: sorted(v) for k, v in universe.sources.items()},
            "oi_growth": universe.oi_growth,
        },
        "research": _capture_research(),
        "calendar": _capture_calendar(),
    }


def _capture_hedge():
    from .strategies.hedge import hedger
    return {"active": {k: dict(v) for k, v in hedger.active.items()},
            "history": hedger.history[-50:],
            "cooldowns": dict(hedger.cooldowns)}


def _capture_calendar():
    from .data.calendar import calendar
    return {"events": calendar.events, "last_update": calendar.last_update}


def _feed_state(f):
    return {"name": f.name, "url": f.url, "kind": f.kind,
            "min_interval": f.min_interval, "candidate": f.candidate,
            "successes": f.successes, "failures": f.failures,
            "dead": f.dead, "backoff": f.backoff}


def _capture_research():
    from .data.research import research
    return {
        "documents": research.documents[-250:],
        "research_queue": research.research_queue,
        "fear_greed": research.fear_greed,
        "feeds": [_feed_state(f) for f in research.feeds],
        "candidates": [_feed_state(f) for f in research.candidates],
        "dynamic_feeds": {p: _feed_state(f)
                          for p, f in research.dynamic_feeds.items()},
    }


_save_lock = None


def save():
    # serialize concurrent saves (event loop + executor) so two writers can't
    # race on the temp file / shared collections.
    global _save_lock
    if _save_lock is None:
        import threading
        _save_lock = threading.Lock()
    with _save_lock:
        try:
            state = capture()
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATE_PATH))
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.replace(tmp, STATE_PATH)   # atomic
            return True
        except Exception as e:
            db.log_event("error", f"State save failed: {e}")
            return False


# ---------------- restore ----------------

def load():
    if not os.path.exists(STATE_PATH):
        return False
    try:
        with open(STATE_PATH) as f:
            s = json.load(f)
    except Exception as e:
        db.log_event("error", f"State load failed (corrupt file?): {e}")
        return False

    from .execution.paper import broker, Position
    from .risk.manager import risk
    from .learn.loop import learner
    from .learn.online_model import model, committee
    from .learn.rl_risk import agent as rl_agent
    from .learn.evolution import evolution
    from .data.universe import universe
    from . import settings

    carry = settings.get("carry_equity")

    try:
        if carry:
            b = s.get("broker", {})
            broker.cash = b.get("cash", broker.cash)
            broker.realized_pnl = b.get("realized_pnl", 0.0)
            broker.positions = {p: Position(pos) for p, pos in
                                b.get("positions", {}).items()}
            broker.closed_trades = b.get("closed_trades", [])

            r = s.get("risk", {})
            risk.peak_equity = r.get("peak_equity", 0.0)
            # fall back to the true peak for snapshots written before the
            # kill_arm_peak split existed.
            risk.kill_arm_peak = r.get("kill_arm_peak", risk.peak_equity)
            risk.day_start_equity = r.get("day_start_equity")
            risk.day_start_ts = r.get("day_start_ts", time.time())
            risk.killed = r.get("killed", False)
            risk.kill_ts = r.get("kill_ts")
            risk.halted_today = r.get("halted_today", False)
            risk.halt_reason = r.get("halt_reason", "")
            if risk.killed:
                # a kill flag reloaded from a snapshot is the sneaky "why is it
                # KILLED and I never pressed it?" case — make that explicit.
                prev = r.get("kill_reason", "")
                risk.kill_reason = (
                    (prev + " · restored from saved state on restart")
                    if prev else "Restored from saved state on restart "
                                 "(kill switch was active when last saved)")
            if r.get("day_index") is not None:
                risk.day_index = r["day_index"]
            risk.cooldowns = r.get("cooldowns", {})
            risk.risk_scale = r.get("risk_scale", 1.0)
            risk.consecutive_losses = r.get("consecutive_losses", 0)
        else:
            db.log_event("system", "carry_equity=OFF — fresh $100k paper "
                                   "account (learned state still restored)")

        l = s.get("learner", {})
        if l.get("weights"):
            # merge: keep defaults for any newly added strategies
            learner.weights.update(l["weights"])
        learner.bandit.arms = _load_arms(l.get("bandit_arms", {}))
        learner.trade_attributions = l.get("trade_attributions",
                                           learner.trade_attributions)
        # Restore the pending ML label queue + price history so samples recorded
        # before a restart still mature into training examples afterwards. This
        # is what actually lets the online model accumulate updates across the
        # frequent restarts (without it, n_updates is pinned at 0 forever).
        from .learn.online_model import N_IN
        pend = l.get("pending_ml") or []
        kept, dropped = [], 0
        for ts, p, x, pred in pend:
            if isinstance(x, (list, tuple)) and len(x) == N_IN:
                kept.append((ts, p, x, pred))
            else:
                dropped += 1
        learner.pending_ml = deque(kept, maxlen=learner.pending_ml.maxlen)
        if dropped:
            db.log_event("learn",
                         f"Dropped {dropped} restored pending ML sample(s) "
                         f"with stale feature dim (need {N_IN})")
        ph = l.get("price_history") or {}
        learner.price_history = {p: [tuple(pt) for pt in hist]
                                 for p, hist in ph.items()}

        # ---- online model / committee ----
        # Prefer the full committee snapshot; fall back to the legacy single
        # "model" dict (restored into the primary member) for old snapshots.
        comm = s.get("committee")
        m = s.get("model", {})
        if comm:
            restored = 0
            for i, member in enumerate(committee.members):
                if i < len(comm) and _load_mlp(member, comm[i]):
                    restored += 1
            if restored:
                # keep any extra live members warm-started from the primary so a
                # grown committee doesn't leave fresh members far behind.
                for member in committee.members[len(comm):]:
                    _load_mlp(member, comm[0])
            elif m.get("W1"):
                db.log_event("learn", "Online model NOT restored: feature "
                                      "dimension changed — starting fresh")
        elif m.get("W1"):
            if not _load_mlp(model, m):
                db.log_event("learn", "Online model NOT restored: feature "
                                      "dimension changed — starting fresh")
            else:
                # legacy snapshot had one net: seed the other members off it.
                for member in committee.members[1:]:
                    _load_mlp(member, m)

        q = s.get("rl", {})
        rl_agent.q = _load_q(q.get("q", {}))
        rl_agent.eps = q.get("eps", rl_agent.eps)
        rl_agent.n_updates = q.get("n_updates", 0)

        hd = s.get("hedge") or {}
        if hd:
            from .strategies.hedge import hedger
            hedger.active = {k: dict(v) for k, v in (hd.get("active") or {}).items()}
            for h in hedger.active.values():
                if isinstance(h.get("pair"), list):
                    h["pair"] = tuple(h["pair"])
            hedger.history = list(hd.get("history") or [])
            hedger.cooldowns = dict(hd.get("cooldowns") or {})

        e = s.get("evolution", {})
        evolution.champions = e.get("champions", {}) or {}
        evolution.champion_portfolios = e.get("champion_portfolios", {}) or {}
        evolution.champion_reports = e.get("champion_reports", {}) or {}
        evolution.last_attempt = e.get("last_attempt", {}) or {}
        # migrate old single-champion format
        if not evolution.champions and e.get("champion"):
            evolution.champions["BTC-USD"] = e["champion"]
            if e.get("champion_report"):
                evolution.champion_reports["BTC-USD"] = e["champion_report"]
        # back-fill portfolios for champions saved before portfolios existed
        for prod, g in evolution.champions.items():
            evolution.champion_portfolios.setdefault(prod, [g])

        u = s.get("universe", {})
        universe.mention_heat = u.get("mention_heat", {})
        universe.name_to_sym = u.get("name_to_sym", {})
        universe.sources = {k: set(v) for k, v in (u.get("sources") or {}).items()}
        universe.oi_growth = u.get("oi_growth", {}) or {}
        # restore discovered universe (mutate PRODUCTS in place)
        saved_products = u.get("products", [])
        for pid in saved_products:
            if pid not in PRODUCTS:
                PRODUCTS.append(pid)
        universe.discovered = [p for p in PRODUCTS if p not in _core()]

        # ---- research corpus + source-discovery memory ----
        _restore_research(s.get("research", {}))

        # ---- macro calendar (so a 429 at startup doesn't blind us) ----
        cal = s.get("calendar", {})
        if cal.get("events"):
            from .data.calendar import calendar as macro_cal
            macro_cal.events = cal["events"]
            macro_cal.last_update = cal.get("last_update", 0)
            macro_cal.healthy = True

        age_min = (time.time() - s.get("saved_at", 0)) / 60
        db.log_event("system",
                     f"State restored (snapshot {age_min:.1f} min old, "
                     f"equity {'carried over' if carry else 'RESET to fresh $100k'}): "
                     f"cash=${broker.cash:,.0f}, "
                     f"{len(broker.positions)} open positions, "
                     f"{len(broker.closed_trades)} trade history, "
                     f"model updates={model.n_updates}, "
                     f"Q-states={len(rl_agent.q)}, "
                     f"bandit arms={len(learner.bandit.arms)}, "
                     f"champions={len(evolution.champions)}, "
                     f"universe={len(PRODUCTS)} coins")
        return True
    except Exception as e:
        db.log_event("error", f"State restore failed: {e}")
        return False


def _core():
    from .data.universe import CORE
    return CORE


def _restore_research(r):
    """Restore document corpus, research queue and source-pool memory.
    Promoted feeds stay promoted; demoted feeds stay demoted; success
    counts carry over so trial sources keep their track record."""
    if not r:
        return
    from .data.research import research, Feed
    research.documents = r.get("documents", [])
    research.research_queue = r.get("research_queue", [])
    research.fear_greed = r.get("fear_greed")

    def apply(feed, saved):
        feed.successes = saved.get("successes", 0)
        feed.failures = saved.get("failures", 0)
        feed.dead = saved.get("dead", False)
        feed.backoff = saved.get("backoff", 0.0)
        feed.candidate = saved.get("candidate", feed.candidate)

    saved_by_name = {f["name"]: f for f in
                     r.get("feeds", []) + r.get("candidates", [])}
    # apply memory onto existing pool objects
    for f in research.feeds + research.candidates:
        if f.name in saved_by_name:
            apply(f, saved_by_name[f.name])
    # move earned promotions: candidates that were promoted last session
    for f in list(research.candidates):
        if f.name in saved_by_name and not saved_by_name[f.name]["candidate"]:
            f.candidate = False
            research.candidates.remove(f)
            research.feeds.append(f)
    # sources promoted last session that aren't in the default pools
    known = {f.name for f in research.feeds + research.candidates}
    for name, saved in saved_by_name.items():
        if name not in known and not saved.get("dead"):
            nf = Feed(saved["name"], saved["url"], kind=saved.get("kind", "rss"),
                      min_interval=saved.get("min_interval", 300),
                      candidate=saved.get("candidate", False))
            apply(nf, saved)
            (research.candidates if nf.candidate else research.feeds).append(nf)
    # dynamic per-coin feeds (recreated with their history)
    for p, saved in r.get("dynamic_feeds", {}).items():
        nf = Feed(saved["name"], saved["url"], kind=saved.get("kind", "rss"),
                  min_interval=saved.get("min_interval", 900))
        apply(nf, saved)
        research.dynamic_feeds[p] = nf
