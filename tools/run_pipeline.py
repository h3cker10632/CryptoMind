"""Run the whole improvement pipeline start to finish, once, and summarize it
(docs/IMPROVEMENT_PIPELINE.md):

    data -> screen -> model -> candidate -> promote

    python tools/run_pipeline.py                      # real data: sync, then every stage
    python tools/run_pipeline.py --skip-sync          # stages on the data already stored
    python tools/run_pipeline.py --synthetic          # offline dry run on a synthetic market
    python tools/run_pipeline.py --synthetic --ml-model ridge --json --cleanup

Stages (each a separate process, so one failing doesn't stop the rest):
  sync       tools/data_sync.py + tools/series_sync.py   (market data, external series)
  screen     tools/signal_screen.py                      (does a feature predict anything?)
  ml         tools/ml_lab.py                             (pooled models vs a baseline;
                                                          queues a candidate if it passes)
  research   tools/research_loop.py                      (backtest + forward test + promote)
  promotion  tools/promotion_backtest.py                 (would the promotion rule have helped?)

`--synthetic` never touches this checkout's data or reports: it copies app/
and tools/ into a throwaway workspace, fills that workspace's store with a
synthetic market (tools/make_synthetic_store.py) and runs every stage there.
It proves the plumbing works end to end; its numbers are not evidence about
real markets. The orchestrator also runs each stage on its own schedule.

Summary: reports/pipeline_latest.json (in the workspace for --synthetic).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ("sync", "screen", "ml", "research", "promotion")


def _run(ws, script, args, say, timeout):
    cmd = [sys.executable, os.path.join(ws, "tools", script)] + list(args)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout)
        code, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        code, out = -1, f"timed out after {timeout}s\n{e.stdout or ''}"
    dt = round(time.time() - t0, 1)
    say(f"  {script} {' '.join(args)} -> exit {code} in {dt}s")
    return {"script": script, "args": list(args), "exit": code, "seconds": dt,
            "tail": out.strip().splitlines()[-12:]}


def _load(ws, name):
    try:
        with open(os.path.join(ws, "reports", name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summarize(ws):
    """The headline of every stage's report."""
    out = {}
    scr = _load(ws, "signal_screen_latest.json")
    if scr:
        res = scr.get("results") or {}
        out["screen"] = {"screened": len(res),
                         "passed": sorted(k for k, v in res.items() if v.get("passes"))}
    ml = _load(ws, "ml_lab_latest.json")
    if ml:
        rk, mt = ml.get("rank_model") or {}, ml.get("trend_meta_model") or {}
        out["ml"] = {"rank_model": {k: rk.get(k) for k in ("ic", "ic_t", "baseline_ic", "passes")},
                     "trend_meta_passes": mt.get("passes"),
                     "queued_candidates": ml.get("registered")}
    ch = _load(ws, "challengers.json")
    rep = (ch or {}).get("last_report")
    if rep:
        cands = rep.get("candidates") or {}
        out["research"] = {
            "champion": rep.get("champion"), "promoted": rep.get("promoted"),
            "trials": rep.get("trials"),
            "candidates": {n: {"sharpe": ((c.get("backtest") or {}).get("full") or {}).get("sharpe"),
                               "deflated_sharpe": (c.get("backtest") or {}).get("deflated_sharpe"),
                               "forward_test": (c.get("forward_test") or {}).get("decision"),
                               "eligible": c.get("eligible_for_promotion")}
                           for n, c in cands.items()},
            "champion_costs_taxes": {k: v.get("cagr_pct") for k, v in
                                     ((rep.get("costs_taxes") or {}).get("scenarios") or {}).items()}}
    pb = _load(ws, "promotion_backtest_latest.json")
    if pb:
        out["promotion"] = {"verdict": pb.get("verdict"), "promotions": len(pb.get("promotions") or []),
                            **{k: ((pb.get(k) or {}).get("full") or {}).get("sharpe")
                               for k in ("process", "never_switch", "hold_btc_eth")}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--coins", type=int, default=30, help="synthetic coins")
    ap.add_argument("--start", default="2017-06-01", help="synthetic history start")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plant-signal", type=float, default=0.0,
                    help="synthetic: plant a cross-coin signal the ML lab should find")
    ap.add_argument("--skip-sync", action="store_true")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--ml-model", default="ensemble", choices=["ridge", "gbt", "ensemble"])
    ap.add_argument("--top", type=int, default=20, help="universe size for screen / ML")
    ap.add_argument("--timeout", type=int, default=3600, help="per-stage seconds")
    ap.add_argument("--json", action="store_true", help="print only the JSON summary")
    ap.add_argument("--cleanup", action="store_true", help="delete the synthetic workspace")
    a = ap.parse_args()
    say = (lambda m: None) if a.json else (lambda m: print(m, flush=True))
    stages = [s for s in a.stages.split(",") if s in STAGES]
    t0 = time.time()
    ws = ROOT
    run = {"started": t0, "synthetic": a.synthetic, "stages": {}}
    if a.synthetic:
        ws = tempfile.mkdtemp(prefix="cryptomind-synthetic-")
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for d in ("app", "tools"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(ws, d), ignore=ignore)
        say(f"synthetic workspace: {ws}")
        run["stages"]["synthetic_store"] = _run(
            ws, "make_synthetic_store.py",
            ["--coins", str(a.coins), "--start", a.start, "--seed", str(a.seed),
             "--plant-signal", str(a.plant_signal)], say, a.timeout)
    run["workspace"] = ws
    top = ["--top", str(a.top)]
    plan = {
        "sync": [] if (a.synthetic or a.skip_sync) else
                [("data_sync.py", []), ("series_sync.py", [])],
        "screen": [("signal_screen.py", top)],
        "ml": [("ml_lab.py", top + ["--model", a.ml_model])],
        "research": [("research_loop.py", [])],
        "promotion": [("promotion_backtest.py", [])],
    }
    for st in stages:
        if not plan[st]:
            say(f"[{st}] skipped")
            run["stages"][st] = {"skipped": True}
            continue
        say(f"[{st}]")
        res = [_run(ws, s, args, say, a.timeout) for s, args in plan[st]]
        run["stages"][st] = res[0] if len(res) == 1 else {"steps": res,
                                                          "exit": max(r["exit"] for r in res)}
    run["summary"] = summarize(ws)
    run["seconds"] = round(time.time() - t0, 1)
    run["ok"] = all(v.get("skipped") or v.get("exit") == 0 for v in run["stages"].values())
    os.makedirs(os.path.join(ws, "reports"), exist_ok=True)
    with open(os.path.join(ws, "reports", "pipeline_latest.json"), "w") as f:
        json.dump(run, f, indent=1, default=str)
    if a.json:
        print(json.dumps({k: run[k] for k in ("ok", "synthetic", "workspace", "seconds",
                                              "summary", "stages")}, default=str))
    else:
        say(json.dumps(run["summary"], indent=1, default=str))
        say(f"\n{'OK' if run['ok'] else 'SOME STAGES FAILED'} in {run['seconds']}s — "
            f"report: {os.path.join(ws, 'reports', 'pipeline_latest.json')}")
    if a.synthetic and a.cleanup:
        shutil.rmtree(ws, ignore_errors=True)
    sys.exit(0 if run["ok"] else 1)


if __name__ == "__main__":
    main()
