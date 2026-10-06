"""Evidence gate for the learners: a learner only drives live decisions while
the latest learner ablation (app/backtest/ablation.py) shows it beating the
plain strategy in BOTH halves of the replay window — and, for sizing
learners, beating a fixed size of the same average multiplier.

Each learner has a `<name>_mode` setting: "off", "on", or "auto" (default).
"auto" reads reports/learner_ablation_latest.json; a missing, failed or stale
(> `learner_gate_max_age_days`) report means OFF — no evidence, no influence.

Learners keep learning while gated off, so a learner switching on doesn't
start cold; `warm_start()` additionally seeds it with the state it reached in
the replay when that is more experienced than the live one.
"""
import json
import os
import time

LEARNERS = ("bandit", "direction", "exit_advisor", "rl_risk", "online_ml")
MODE_KEY = {"bandit": "bandit_mode", "direction": "direction_learner_mode",
            "exit_advisor": "exit_advisor_mode", "rl_risk": "rl_risk_mode",
            "online_ml": "ml_vote_mode"}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, "reports", "learner_ablation_latest.json")
STATES = os.path.join(ROOT, "reports", "learner_states.json")

_cache = {"mtime": None, "report": None}


def _report():
    try:
        m = os.path.getmtime(REPORT)
    except OSError:
        return None
    if _cache["mtime"] != m:
        try:
            with open(REPORT) as f:
                _cache["report"] = json.load(f)
        except Exception:
            _cache["report"] = None
        _cache["mtime"] = m
    return _cache["report"]


def evidence(name, now=None):
    """(helps: bool, why: str) from the latest ablation report."""
    rep = _report()
    if not rep or not rep.get("ok"):
        return False, "no learner ablation report yet"
    try:
        from .. import settings
        max_age = float(settings.get("learner_gate_max_age_days")) * 86400
    except Exception:
        max_age = 10 * 86400
    age = (now or time.time()) - rep.get("ran_at", 0)
    if age > max_age:
        return False, f"ablation report is {age / 86400:.0f} days old"
    v = (rep.get("variants") or {}).get(name)
    if not v:
        return False, f"{name} not in the ablation report"
    d = v.get("delta_vs_baseline_pct_pts") or {}
    why = (f"replay vs no-learner: {d.get('first_half', 0):+.1f} / "
           f"{d.get('second_half', 0):+.1f} pts by half")
    ctrl = v.get("size_matched_control")
    if ctrl and not ctrl.get("beats_control_in_both_halves"):
        why += f"; no better than a fixed {ctrl.get('avg_size_mult')}x size"
    return bool(v.get("helps_in_both_halves")), why


def mode(name):
    try:
        from .. import settings
        m = settings.get(MODE_KEY[name])
    except Exception:
        m = "auto"
    return m if m in ("off", "on", "auto") else "auto"


def active(name):
    m = mode(name)
    if m == "on":
        return True
    if m == "off":
        return False
    return evidence(name)[0]


def status():
    out = {}
    for n in LEARNERS:
        helps, why = evidence(n)
        out[n] = {"mode": mode(n), "active": active(n), "evidence_helps": helps, "why": why}
    return out


# ------------------------------------------------------------- warm start
def save_states(states):
    """Persist the learner states the ablation reached (see ablation.py)."""
    os.makedirs(os.path.dirname(STATES), exist_ok=True)
    tmp = STATES + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"saved_at": time.time(), "states": states}, f)
    os.replace(tmp, STATES)


def warm_start(name):
    """Seed a live learner with its replay-trained state when the replay one
    has seen more updates than the live one. Returns True if it did."""
    try:
        with open(STATES) as f:
            st = (json.load(f).get("states") or {}).get(name)
    except Exception:
        return False
    if not st:
        return False
    if name == "direction":
        from .direction import direction_learner as live
        if st.get("n_updates", 0) <= live.n_updates:
            return False
        live.restore(st)
        return True
    if name == "exit_advisor":
        from .exit_advisor import exit_advisor as live
        if st.get("n_updates", 0) <= live.n_updates:
            return False
        st = dict(st, pending=[])           # replay timestamps aren't live ones
        live.restore(st)
        return True
    if name == "rl_risk":
        from .rl_risk import agent as live
        if st.get("n_updates", 0) <= live.n_updates:
            return False
        live.q = {tuple(k.split("||")[:2]) + tuple(int(x) for x in k.split("||")[2:]): v
                  for k, v in st["q"].items()}
        live.n_updates = st["n_updates"]
        return True
    if name == "bandit":
        from .loop import learner as live
        n_live = sum(a[0] for a in live.bandit.arms.values())
        arms = {tuple(k.split("||", 1)): tuple(v) for k, v in st["arms"].items()}
        if sum(a[0] for a in arms.values()) <= n_live:
            return False
        live.bandit.arms.update(arms)
        return True
    return False
