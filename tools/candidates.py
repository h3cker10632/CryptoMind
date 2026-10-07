"""Manage the research loop's candidate queue (app/engine/challengers.py).

    python tools/candidates.py list
    python tools/candidates.py add NAME '{"assets": ["BTC-USD","ETH-USD"], "selection": "trend", "sma": 150}'
    python tools/candidates.py remove NAME

A queued config is frozen: its forward clock starts at the next research-loop
run, and it counts as one more trial in the deflated Sharpe. Changing it later
restarts its clock.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from app.engine import challengers as C
    args = sys.argv[1:]
    if not args or args[0] == "list":
        champ, champ_cfg = C.champion()
        st = C._load(C.STATE, {}).get("candidates", {})
        for name, cfg in dict(C.candidates(), **{champ: champ_cfg}).items():
            reg = st.get(name, {})
            tag = " (champion)" if name == champ else ""
            ret = " RETIRED" if reg.get("retired") else ""
            print(f"{name}{tag}{ret}: {json.dumps(cfg)}")
        return
    if args[0] == "add" and len(args) == 3:
        ok, msg = C.register(args[1], json.loads(args[2]), source="cli")
        print(msg)
        sys.exit(0 if ok else 1)
    if args[0] == "remove" and len(args) == 2:
        print("removed" if C.unregister(args[1]) else "not queued")
        return
    print(__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
