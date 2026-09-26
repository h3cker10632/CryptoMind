"""Invo copy-signal integration — dashboard-driven collection + edge study.

This is the app-side glue that makes the standalone `tools/invo_signal` harness
usable from the Settings tab with no code:

  * a CONFIG-DRIVEN mapper (field names come from settings, not a Python edit),
  * a background snapshot collector (start/stop/status),
  * peek (see the raw API response so you can fill in the field map),
  * run the edge study (verdict + per-asset IC), using CryptoMind's own live
    candle history for the forward-return prices.

It NEVER wires the Invo signal into live trading. Everything here is measurement
plumbing; the signal becomes a learner feature only via a separate explicit step
after the study earns it. `tools/invo_signal` is our own numpy-only code (no
Maxun / AGPL), so importing it here is fine.
"""
from __future__ import annotations
import asyncio, json, os, tempfile, time

from .. import db
from .. import settings as app_settings
from ..config import PRODUCTS
from tools.invo_signal.poller import InvoPoller, PollerConfig
from tools.invo_signal import run_study as _study

SNAP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         "invo_snapshots.json")


def _dig(obj, path):
    """Fetch a possibly-dotted key path from a dict; None if missing."""
    if not path:
        return None
    cur = obj
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _as_list(obj, list_key):
    """Resolve the traders/positions list from a response of unknown shape."""
    if list_key:
        v = _dig(obj, list_key)
        if isinstance(v, list):
            return v
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("data", "results", "items", "positions", "traders"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return []


def config_mapper(leaderboard_raw, fetch_positions, top_n):
    """No-code mapper: builds the snapshot schema from the field map in settings."""
    s = app_settings.load()
    rows = _as_list(leaderboard_raw, s.get("invo_map_list", ""))[:top_n]
    long_val = str(s.get("invo_map_long_value", "long")).lower()
    traders = []
    for i, row in enumerate(rows, start=1):
        tid = _dig(row, s.get("invo_map_id", "id"))
        try:
            score = float(_dig(row, s.get("invo_map_score")) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        pos_key = s.get("invo_map_positions", "")
        if pos_key:
            pos_raw = _as_list(row, pos_key)
        elif fetch_positions and tid is not None:
            pos_raw = _as_list(fetch_positions(tid), s.get("invo_map_list", ""))
        else:
            pos_raw = []
        positions = []
        for p in pos_raw:
            side = str(_dig(p, s.get("invo_map_side", "side")) or "").lower()
            try:
                notional = float(_dig(p, s.get("invo_map_size")) or 0.0)
            except (TypeError, ValueError):
                notional = 0.0
            try:
                lev = float(_dig(p, s.get("invo_map_leverage")) or 1.0)
            except (TypeError, ValueError):
                lev = 1.0
            asset = _dig(p, s.get("invo_map_asset", "coin"))
            if not asset or notional <= 0:
                continue
            positions.append({"asset": str(asset).upper(),
                              "direction": 1 if side == long_val else -1,
                              "notional": notional, "leverage": lev})
        traders.append({"id": tid, "rank": i, "score": score, "positions": positions})
    return traders


def _cfg() -> PollerConfig:
    s = app_settings.load()
    return PollerConfig(
        base_url=str(s["invo_api_base"]).rstrip("/"),
        token=s["invo_token"],
        leaderboard_path=s["invo_leaderboard_path"],
        positions_tmpl=s["invo_positions_tmpl"] or None,
        top_n=int(s["invo_top_n"]),
        interval_sec=float(s["invo_interval_sec"]),
        out_path=SNAP_PATH,
        refresh_path=s.get("invo_refresh_path", ""),
        refresh_token=s.get("invo_refresh_token", ""),
        refresh_body=s.get("invo_refresh_body", ""),
        token_json_path=s.get("invo_token_path", "access_token") or "access_token",
        refresh_rotates_path=s.get("invo_refresh_rotates_path", ""),
    )


def _persist_tokens(access: str, refresh: str | None = None):
    """Persist an auto-refreshed access token (and rotated refresh token, if any)
    back into the git-ignored secret store, so it survives restarts."""
    upd = {"invo_token": access}
    if refresh:
        upd["invo_refresh_token"] = refresh
    try:
        app_settings.update(upd)
        db.log_event("system", "Invo access token auto-refreshed")
    except Exception:                               # best-effort
        pass


def _make_poller() -> InvoPoller:
    return InvoPoller(_cfg(), mapper=config_mapper, on_new_token=_persist_tokens)


def _ready() -> str | None:
    s = app_settings.load()
    if not s.get("invo_enabled"):
        return "Invo study is disabled — enable it in Settings."
    for k, label in (("invo_api_base", "API base URL"),
                     ("invo_token", "API token"),
                     ("invo_leaderboard_path", "leaderboard path")):
        if not s.get(k):
            return f"Missing {label} in Settings."
    return None


# ----------------------------- peek -----------------------------
def peek_sync() -> dict:
    err = _ready()
    if err:
        return {"ok": False, "error": err}
    try:
        p = _make_poller()
        import httpx
        s = app_settings.load()
        with httpx.Client() as c:
            lb = p._get(c, s["invo_leaderboard_path"])
            sample_positions = None
            tid = None
            rows = _as_list(lb, s.get("invo_map_list", ""))
            if rows:
                tid = _dig(rows[0], s.get("invo_map_id", "id"))
            if s["invo_positions_tmpl"] and tid is not None:
                sample_positions = p._get(c, s["invo_positions_tmpl"].format(id=tid))
            # also show what the current field-map extracts, so mapping is verifiable
            mapped = config_mapper(lb, None, 3)
        return {"ok": True, "leaderboard_sample": _truncate(lb),
                "positions_sample": _truncate(sample_positions),
                "mapped_preview": mapped}
    except Exception as e:                      # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _truncate(obj, limit=4000):
    try:
        txt = json.dumps(obj, indent=2)
    except Exception:
        return None
    return txt[:limit] + ("\n… (truncated)" if len(txt) > limit else "")


# --------------------------- collector ---------------------------
class _Collector:
    def __init__(self):
        self.task: asyncio.Task | None = None
        self.running = False
        self.count = self._file_count()
        self.last_ts = 0.0
        self.last_error = ""

    def _file_count(self) -> int:
        try:
            with open(SNAP_PATH) as f:
                return len(json.load(f))
        except Exception:
            return 0

    def status(self) -> dict:
        return {"running": self.running, "snapshots": self._file_count(),
                "last_snapshot_ts": self.last_ts, "last_error": self.last_error,
                "path": SNAP_PATH}

    async def _loop(self):
        poller = _make_poller()
        db.log_event("system", "Invo collector started")
        while self.running:
            try:
                snap = await asyncio.to_thread(poller.fetch_once)
                self.count = await asyncio.to_thread(poller.append_snapshot, snap)
                self.last_ts = snap["ts"]
                self.last_error = ""
            except Exception as e:                  # noqa: BLE001
                self.last_error = str(e)
                db.log_event("warn", f"Invo collector snapshot failed: {e}")
            interval = float(app_settings.get("invo_interval_sec"))
            for _ in range(int(interval)):
                if not self.running:
                    break
                await asyncio.sleep(1)
        db.log_event("system", "Invo collector stopped")

    def start(self) -> dict:
        err = _ready()
        if err:
            return {"ok": False, "error": err}
        if self.running:
            return {"ok": True, "already": True, **self.status()}
        self.running = True
        self.task = asyncio.create_task(self._loop())
        return {"ok": True, **self.status()}

    def stop(self) -> dict:
        self.running = False
        return {"ok": True, **self.status()}


collector = _Collector()


# ----------------------------- study -----------------------------
def _prices_from_market() -> dict:
    """Build {SYMBOL: [[ts, close], ...]} from CryptoMind's own live candles."""
    from .market import market
    out = {}
    for prod, candles in market.candles.items():
        sym = prod.split("-")[0].upper()
        rows = [[float(c[0]), float(c[4])] for c in candles if len(c) >= 5]
        if rows:
            out[sym] = rows
    return out


def run_study_sync() -> dict:
    if collector._file_count() < 2:
        return {"ok": False, "error": "Not enough snapshots collected yet — "
                                      "start the collector and let it run."}
    prices = _prices_from_market()
    if not prices:
        return {"ok": False, "error": "No candle history available yet for "
                                      "forward-return prices."}
    s = app_settings.load()
    fd, ptmp = tempfile.mkstemp(suffix="_prices.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(prices, f)
        rep = _study.run(SNAP_PATH, ptmp,
                         horizon_hours=float(s["invo_horizon_hours"]),
                         rank_decay=float(s["invo_rank_decay"]),
                         use_score=bool(s["invo_use_score"]))
    finally:
        try:
            os.remove(ptmp)
        except OSError:
            pass
    rep["verdict"] = _study._verdict(rep.get("pooled", {}))
    db.log_event("system", "Invo edge study run",
                 {"verdict": rep["verdict"], "pooled": rep.get("pooled", {})})
    return {"ok": True, **rep}
