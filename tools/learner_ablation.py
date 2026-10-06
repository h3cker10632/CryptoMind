"""Learner ablation — does each learning component earn its keep?

Replays the cached hourly history (same engine as tools/replay_backtest.py)
with no learners, then with each learner switched on from a fresh state, and
reports which ones beat the baseline in BOTH halves of the window. Method:
app/backtest/ablation.py. Read-only: never touches live state or settings.

    python tools/learner_ablation.py                  # everything (~20-30 min: online_ml is slow)
    python tools/learner_ablation.py --skip online_ml # fast (~2 min)
    python tools/learner_ablation.py --only bandit --seeds 5
    python tools/learner_ablation.py --out reports/learner_ablation.json
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _table(rep):
    b = rep["baseline"]
    rows = [("baseline", b["full"]["return_pct"], b["first_half"]["return_pct"],
             b["second_half"]["return_pct"], b["full"]["trades"], "", "")]
    for name, v in rep["variants"].items():
        m, d = v["mean"], v["delta_vs_baseline_pct_pts"]
        ctrl = v.get("size_matched_control")
        verdict = "HELPS" if v["helps_in_both_halves"] else "no"
        if ctrl and not ctrl["beats_control_in_both_halves"]:
            verdict += f" (just smaller bets: fixed {ctrl['avg_size_mult']}x size does as well)"
        rows.append((name, m["full"]["return_pct"], m["first_half"]["return_pct"],
                     m["second_half"]["return_pct"], m["full"]["trades"],
                     f"{d['first_half']:+.1f} / {d['second_half']:+.1f}",
                     verdict + (f" ({v['seeds_helping_in_both_halves']} seeds beat baseline)"
                                if v["seeds"] > 1 else "")))
    head = ("variant", "full %", "1st half %", "2nd half %", "trades",
            "vs baseline (pts)", "helps both halves")
    w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    line = lambda r: "  ".join(str(c).rjust(w[i]) if i else str(c).ljust(w[i])
                               for i, c in enumerate(r))
    return "\n".join([line(head), line(["-" * x for x in w])] + [line(r) for r in rows])


def main():
    from app.backtest import ablation as ab
    from app.backtest.replay import load_cached_history
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=ab.LEARNERS, default=[])
    ap.add_argument("--skip", action="append", choices=ab.LEARNERS, default=[])
    ap.add_argument("--seeds", type=int, default=3, help="runs per stochastic learner")
    ap.add_argument("--no-all", action="store_true", help="skip the all-learners run")
    ap.add_argument("--no-filters", action="store_true", help="skip trade/chop filter rows")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "learner_ablation_latest.json"))
    a = ap.parse_args()
    learners = tuple(x for x in (a.only or ab.LEARNERS) if x not in a.skip)
    t0 = time.time()
    rep = ab.run_ablation(load_cached_history(ROOT), seeds=a.seeds, learners=learners,
                          include_all=not a.no_all, include_filters=not a.no_filters,
                          progress=lambda m: print(f"[{time.time() - t0:6.0f}s] {m}",
                                                   flush=True))
    if not rep.get("ok"):
        print(json.dumps(rep, indent=1))
        sys.exit(1)
    from app.learn import gate
    states = rep.pop("_states", None)
    if states:
        gate.save_states(states)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, a.out)                 # the live gate reads this file
    print()
    print(f"{rep['days']} days, {rep['products']} coins. Each learner starts empty and "
          f"learns as the replay walks forward.")
    print(_table(rep))
    print("\nNot testable in replay: " + ", ".join(f"{k} ({v})" for k, v in rep["not_testable"].items()))
    print(f"\nFull report: {a.out}")


if __name__ == "__main__":
    main()
