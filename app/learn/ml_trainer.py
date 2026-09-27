"""Autonomous, metric-gated ML retraining pipeline.

Closes the crypto_ml_lab loop WITHOUT operator babysitting: when enough NEW
labeled data has matured, this runs the full offline pipeline

    export -> validate -> prepare -> train -> backtest

then AUTO-PROMOTES the freshly trained artifact into the live model advisor —
but ONLY if the backtest's measured gates pass (out-of-sample Sharpe, trade
count, beats-baseline). If the gates fail, the candidate is kept in staging for
inspection and the currently-live model is left untouched. Nothing is ever
promoted on vibes or on unreadable metrics.

Design (same fail-safe governance as the LLM / model advisors):
  * OFF by default (`ml_autotrain_enabled`). Enabling it on a machine WITHOUT
    crypto_ml installed simply records a "lab command not found" status and
    promotes nothing.
  * DATA-DRIVEN trigger: only retrains once >= `ml_autotrain_min_new_labels`
    new labeled rows have accumulated since the last successful train, so it
    never churns on unchanged data (labels need the forward horizon to elapse).
  * The heavy work runs in a background thread; the async loop only kicks it.
  * Promotion is atomic (temp file + os.replace) and resets the advisor cache.
  * Unknown/unreadable backtest metrics => gate FAIL => no promotion. Ever.

The lab CLI is invoked exactly as documented in
docs/crypto_ml_lab_integration.md (`python -m crypto_ml.cli ...`), configurable
via the `ml_lab_cmd` setting. The backtest is expected to write a small JSON
metrics file into its run/output dir (default `metrics.json`); see
`_read_metrics` / `_evaluate_gates` for the accepted keys and aliases.
"""
import json
import os
import shlex
import shutil
import subprocess
import threading
import time


# metric-name aliases so we tolerate small differences in the lab's output
_RETURN_KEYS = ("total_return", "oos_return", "cum_return", "net_return", "return")
_DRAWDOWN_KEYS = ("max_drawdown", "max_dd", "drawdown", "mdd")
_SHARPE_KEYS = ("oos_sharpe", "test_sharpe", "sharpe", "sharpe_ratio")
_TRADES_KEYS = ("n_oos_trades", "oos_trades", "n_trades", "trades", "num_trades")
_BEATS_KEYS = ("beats_baseline", "beats_bench", "beats_benchmark", "outperforms_baseline")

_STEP_TIMEOUT = 60 * 60  # 1h per CLI step — training can be slow


class MLTrainer:
    def __init__(self):
        self._running = False
        self._lock = threading.Lock()
        self.last_run = None        # dict report of the most recent pipeline run
        self.last_error = ""

    # ---------------- config ----------------
    @staticmethod
    def _s(key, default=None):
        try:
            from .. import settings
            v = settings.get(key)
            return default if v is None else v
        except Exception:
            return default

    @classmethod
    def enabled(cls):
        return bool(cls._s("ml_autotrain_enabled", False))

    @classmethod
    def _min_new_labels(cls):
        return int(cls._s("ml_autotrain_min_new_labels", 200))

    @classmethod
    def _lab_cmd(cls):
        return str(cls._s("ml_lab_cmd", "python -m crypto_ml.cli")).strip()

    @classmethod
    def _metrics_file(cls):
        return str(cls._s("ml_backtest_metrics_file", "metrics.json")).strip() or "metrics.json"

    @classmethod
    def _gate(cls):
        return {
            # primary risk gate (metrics the lab's backtest actually emits)
            "min_return": float(cls._s("ml_gate_min_return", 0.0)),
            "max_drawdown": float(cls._s("ml_gate_max_drawdown", 0.25)),
            # walk-forward robustness (only enforced when validation ran)
            "min_frac_folds_positive": float(cls._s("ml_gate_min_frac_folds_positive", 0.75)),
            # optional extra gates — only enforced if the backtest reports them
            "min_oos_sharpe": float(cls._s("ml_gate_min_oos_sharpe", 0.5)),
            "min_oos_trades": int(cls._s("ml_gate_min_oos_trades", 20)),
            "require_beats_baseline": bool(cls._s("ml_gate_require_beats_baseline", False)),
        }

    @classmethod
    def _val_folds(cls):
        return int(cls._s("ml_val_folds", 4))

    @classmethod
    def _val_embargo(cls):
        return int(cls._s("ml_val_embargo", 24))

    @classmethod
    def _allow_backtest_fallback(cls):
        return bool(cls._s("ml_gate_allow_backtest_fallback", False))

    # ---------------- paths ----------------
    @staticmethod
    def _model_dir():
        from .model_advisor import ModelAdvisor
        return ModelAdvisor.model_dir()

    @classmethod
    def _staging_dir(cls):
        return os.path.join(cls._model_dir(), "staging")

    @classmethod
    def _state_path(cls):
        # lives inside the (git-ignored) model_artifact dir so it persists
        # locally across restarts but is never committed
        return os.path.join(cls._model_dir(), ".autotrain_state.json")

    # ---------------- persistent trigger state ----------------
    @classmethod
    def _load_state(cls):
        try:
            with open(cls._state_path()) as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def _save_state(cls, state):
        try:
            os.makedirs(cls._model_dir(), exist_ok=True)
            tmp = cls._state_path() + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, cls._state_path())
        except Exception as e:  # pragma: no cover - defensive
            cls_last = f"could not persist autotrain state: {e}"
            print(f"[ml_trainer] {cls_last}")

    # ---------------- trigger ----------------
    @classmethod
    def _labeled_count(cls):
        try:
            from .. import db
            return int(db.labeled_signals_count())
        except Exception:
            return 0

    @classmethod
    def trigger_status(cls):
        """How close we are to the next data-driven retrain."""
        labeled = cls._labeled_count()
        base = int(cls._load_state().get("last_trained_label_count", 0))
        new = max(0, labeled - base)
        thresh = cls._min_new_labels()
        return {
            "labeled_total": labeled,
            "baseline_at_last_train": base,
            "new_since_last_train": new,
            "threshold": thresh,
            "ready": new >= thresh,
        }

    def should_run(self):
        """(ok, reason) for whether a scheduled (non-forced) run should fire."""
        if not self.enabled():
            return False, "disabled"
        if self._running:
            return False, "already_running"
        t = self.trigger_status()
        if not t["ready"]:
            return (False,
                    f"waiting_for_labels ({t['new_since_last_train']}/{t['threshold']})")
        return True, "ready"

    # ---------------- CLI runner (mockable) ----------------
    def _run_cli(self, args, cwd, timeout=_STEP_TIMEOUT):
        """Run one lab CLI sub-command. Returns (rc, out_tail, err_tail).
        rc=127 signals 'lab command not found'. Isolated here so tests can stub
        the whole pipeline without crypto_ml installed."""
        cmd = shlex.split(self._lab_cmd()) + list(args)
        try:
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                               timeout=timeout)
            return p.returncode, (p.stdout or "")[-2000:], (p.stderr or "")[-2000:]
        except FileNotFoundError as e:
            return 127, "", f"lab command not found: {e}"
        except subprocess.TimeoutExpired:
            return 124, "", f"step timed out after {timeout}s"
        except Exception as e:  # pragma: no cover - defensive
            return 1, "", f"{type(e).__name__}: {e}"

    # ---------------- dataset export (in-process, our own code) ----------------
    def _export_dataset(self, staging):
        from .. import ml_export
        data = ml_export.build_ml_dataset()
        os.makedirs(staging, exist_ok=True)
        ev = os.path.join(staging, "events.json")
        lb = os.path.join(staging, "labels.json")
        with open(ev, "w") as f:
            json.dump({"events": data["events"]}, f, default=str)
        with open(lb, "w") as f:
            json.dump({"meta": data["meta"], "labels": data["labels"]}, f, default=str)
        return ev, lb, data["meta"]

    # ---------------- purged walk-forward validation ----------------
    def _run_purged_validation(self, staging, features_path):
        """Estimate GENUINE out-of-sample performance via purged, embargoed
        walk-forward: split the prepared feature table into expanding folds,
        train a throwaway model per fold, and score it only on the held-out slice
        it never saw. Returns the pooled gate-metric dict (return/drawdown +
        frac_folds_positive), or None if it can't run (no pandas, unreadable
        parquet, or a fold's train/backtest failed) so the caller can decide how
        to fail closed.
        """
        from . import validation
        try:
            import pandas as pd
        except Exception:
            return None
        try:
            df = pd.read_parquet(features_path)
        except Exception:
            return None
        n = len(df)
        # keep chronological order so folds respect time (prepare usually already
        # emits time-ordered rows; sort if an obvious time column exists)
        for tcol in ("date", "timestamp", "ts", "datetime", "time"):
            if tcol in df.columns:
                try:
                    df = df.sort_values(tcol).reset_index(drop=True)
                except Exception:
                    pass
                break
        folds = validation.walk_forward_folds(n, self._val_folds(), self._val_embargo())
        if not folds:
            return None
        valdir = os.path.join(staging, "val")
        shutil.rmtree(valdir, ignore_errors=True)
        os.makedirs(valdir, exist_ok=True)
        per_fold = []
        for i, (tr0, tr1, te0, te1) in enumerate(folds):
            fdir = os.path.join(valdir, f"fold{i}")
            os.makedirs(fdir, exist_ok=True)
            train_pq = os.path.join(fdir, "train.parquet")
            test_pq = os.path.join(fdir, "test.parquet")
            try:
                df.iloc[tr0:tr1].to_parquet(train_pq)
                df.iloc[te0:te1].to_parquet(test_pq)
            except Exception:
                return None
            run_fold = os.path.join(fdir, "run")
            rc, _out, _err = self._run_cli(
                ["train", "--input", train_pq, "--output", run_fold], cwd=staging)
            if rc != 0:
                return None
            mpath = os.path.join(fdir, self._metrics_file())
            rc, out, _err = self._run_cli(
                ["backtest", "--input", test_pq,
                 "--model", os.path.join(run_fold, "model.joblib"),
                 "--output", mpath], cwd=staging)
            if rc != 0:
                return None
            m = self._read_metrics(fdir) or self._parse_stdout_metrics(out)
            if not m:
                return None
            per_fold.append({
                "fold": i,
                "train_rows": tr1 - tr0,
                "test_rows": te1 - te0,
                "total_return": self._first(m, _RETURN_KEYS),
                "max_drawdown": self._first(m, _DRAWDOWN_KEYS),
            })
        return validation.aggregate_oos(per_fold)

    # ---------------- metrics + gate ----------------
    @staticmethod
    def _first(d, keys):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] is not None:
                return d[k]
        return None

    @staticmethod
    def _parse_stdout_metrics(text):
        """Fallback: extract a JSON metrics object from the backtest's stdout
        (some builds only print, they don't write a file)."""
        if not text:
            return None
        try:
            m = json.loads(text)
            if isinstance(m, dict):
                return m
        except Exception:
            pass
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                m = json.loads(text[start:end + 1])
                if isinstance(m, dict):
                    return m
            except Exception:
                return None
        return None

    def _read_metrics(self, run_dir):
        """Read the backtest metrics JSON. Tries the configured filename in the
        run dir, then a couple of common fallbacks. Returns dict or None."""
        candidates = [
            os.path.join(run_dir, self._metrics_file()),
            os.path.join(run_dir, "metrics.json"),
            os.path.join(run_dir, "backtest.json"),
            os.path.join(run_dir, "report.json"),
        ]
        for path in candidates:
            try:
                with open(path) as f:
                    m = json.load(f)
                if isinstance(m, dict):
                    m["_metrics_path"] = path
                    return m
            except Exception:
                continue
        return None

    def _evaluate_gates(self, metrics):
        """Return (promote: bool, breakdown: dict). Fail-closed: anything
        missing or unreadable blocks promotion.

        PRIMARY gate = return + drawdown (the metrics crypto_ml's backtest
        emits). Both are REQUIRED — if either is absent we do not promote.
        Sharpe / trade-count / beats-baseline are OPTIONAL and only enforced
        when the backtest actually reports them, so enhancing the backtest later
        automatically tightens the gate without a code change."""
        g = self._gate()
        if not metrics:
            return False, {"pass": False, "reason": "no_metrics",
                           "detail": "backtest produced no readable metrics"}
        ret = self._first(metrics, _RETURN_KEYS)
        dd = self._first(metrics, _DRAWDOWN_KEYS)
        sharpe = self._first(metrics, _SHARPE_KEYS)
        trades = self._first(metrics, _TRADES_KEYS)
        beats = self._first(metrics, _BEATS_KEYS)
        checks = {}

        # --- required: profitability + drawdown control ---
        checks["return"] = {
            "value": ret, "min": g["min_return"],
            "pass": ret is not None and float(ret) >= g["min_return"]}
        checks["max_drawdown"] = {
            "value": dd, "limit": g["max_drawdown"],
            "pass": dd is not None and abs(float(dd)) <= g["max_drawdown"]}

        # --- walk-forward robustness: consistency across OOS folds ---
        frac = metrics.get("oos_frac_folds_positive")
        if frac is not None:
            checks["frac_folds_positive"] = {
                "value": frac, "min": g["min_frac_folds_positive"],
                "pass": float(frac) >= g["min_frac_folds_positive"]}

        # --- optional: only enforced when the metric is present ---
        if sharpe is not None:
            checks["sharpe"] = {"value": sharpe, "min": g["min_oos_sharpe"],
                                "pass": float(sharpe) >= g["min_oos_sharpe"]}
        if trades is not None and g["min_oos_trades"] > 0:
            checks["trades"] = {"value": trades, "min": g["min_oos_trades"],
                                "pass": int(trades) >= g["min_oos_trades"]}
        if g["require_beats_baseline"]:
            checks["beats_baseline"] = {"value": beats,
                                        "pass": beats is not None and bool(beats) is True}

        failed = [k for k, v in checks.items() if not v["pass"]]
        promote = not failed
        return promote, {
            "pass": promote,
            "reason": "gates_passed" if promote else ("failed:" + ",".join(failed)),
            "checks": checks,
            "metrics_path": metrics.get("_metrics_path"),
        }

    # ---------------- promotion (atomic) ----------------
    def _promote(self, run_dir):
        """Copy model.joblib + metadata.json from the staging run into the live
        model dir atomically, then reset the advisor so it reloads."""
        model_dir = self._model_dir()
        os.makedirs(model_dir, exist_ok=True)
        src_model = os.path.join(run_dir, "model.joblib")
        # metadata may sit in the run dir or beside the model
        src_meta = None
        for cand in (os.path.join(run_dir, "metadata.json"),
                     os.path.join(run_dir, "model.metadata.json")):
            if os.path.exists(cand):
                src_meta = cand
                break
        if not os.path.exists(src_model):
            return False, f"no model.joblib in {run_dir}"
        try:
            for src, name in ((src_model, "model.joblib"), (src_meta, "metadata.json")):
                if not src:
                    continue
                dst = os.path.join(model_dir, name)
                tmp = dst + ".tmp"
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
            try:
                from .model_advisor import advisor
                advisor.reload()
            except Exception:
                pass
            return True, "promoted"
        except Exception as e:
            return False, f"promotion copy failed: {e}"

    # ---------------- the pipeline ----------------
    def run_pipeline(self, force=False):
        """Run the whole export->train->backtest->gate->promote sequence
        SYNCHRONOUSLY. Returns the run report dict (also stored as self.last_run).
        Safe to call directly in tests with _run_cli stubbed."""
        with self._lock:
            if self._running:
                return {"started": False, "reason": "already_running"}
            if not force:
                ok, reason = self.should_run()
                if not ok:
                    return {"started": False, "reason": reason}
            self._running = True

        started = time.time()
        report = {
            "started": True,
            "forced": bool(force),
            "started_at": started,
            "steps": [],
            "promoted": False,
        }
        labeled_at_start = self._labeled_count()
        staging = self._staging_dir()
        run_dir = os.path.join(staging, "run")
        try:
            # 0) fresh staging
            shutil.rmtree(staging, ignore_errors=True)
            os.makedirs(staging, exist_ok=True)

            # 1) export our labeled dataset (in-process)
            ev, lb, meta = self._export_dataset(staging)
            report["dataset"] = {"n_events": meta.get("n_events"),
                                 "n_labeled": meta.get("n_labeled_signals")}
            if not meta.get("n_events"):
                report["reason"] = "no_events_to_train_on"
                return report

            feats = os.path.join(staging, "features.parquet")
            # 2..4) lab CLI steps. We train the FINAL (to-be-promoted) model on
            # ALL data; the honest gate metric comes from purged walk-forward
            # validation below, NOT from an in-sample backtest of this model.
            steps = [
                ("validate", ["validate", "--input", ev]),
                ("prepare", ["prepare", "--input", ev, "--output", feats]),
                ("train", ["train", "--input", feats, "--output", run_dir]),
            ]
            step_out = {}
            for name, args in steps:
                rc, out, err = self._run_cli(args, cwd=staging)
                step_out[name] = out or ""
                report["steps"].append({"step": name, "rc": rc,
                                        "out_tail": (out or "")[-400:],
                                        "err_tail": err[-400:] if err else ""})
                if rc != 0:
                    not_installed = (
                        rc == 127
                        or "no module named crypto_ml" in (err or "").lower()
                        or "no module named 'crypto_ml'" in (err or "").lower())
                    report["reason"] = (
                        "lab_not_installed" if not_installed else f"{name}_failed_rc{rc}")
                    self.last_error = f"{name} rc={rc}: {err[-200:]}"
                    return report

            # 5) TRUSTWORTHY GATE: purged, embargoed walk-forward OOS validation.
            metrics = self._run_purged_validation(staging, feats)
            if metrics is not None:
                report["validation"] = {
                    "source": "purged_walkforward",
                    "n_folds": metrics.get("oos_n_folds"),
                    "frac_folds_positive": metrics.get("oos_frac_folds_positive"),
                    "median_return": metrics.get("total_return"),
                    "mean_return": metrics.get("mean_return"),
                    "worst_drawdown": metrics.get("max_drawdown"),
                    "per_fold": metrics.get("per_fold"),
                }
            elif self._allow_backtest_fallback():
                # explicit opt-in only: gate on the (in-sample!) backtest instead
                metrics_path = os.path.join(run_dir, self._metrics_file())
                rc, out, err = self._run_cli(
                    ["backtest", "--input", feats,
                     "--model", os.path.join(run_dir, "model.joblib"),
                     "--output", metrics_path], cwd=staging)
                report["steps"].append({"step": "backtest", "rc": rc,
                                        "out_tail": (out or "")[-400:],
                                        "err_tail": err[-400:] if err else ""})
                metrics = self._read_metrics(run_dir) or self._parse_stdout_metrics(out or "")
                report["validation"] = {"source": "raw_backtest_fallback"}
            else:
                report["reason"] = "validation_unavailable_not_promoted"
                report["validation"] = {"source": "none",
                                        "detail": "purged walk-forward could not "
                                        "run (need pandas + a readable features "
                                        "parquet); refusing to promote on an "
                                        "in-sample backtest. Set "
                                        "ml_gate_allow_backtest_fallback=1 to override."}
                return report
            report["metrics"] = {k: v for k, v in (metrics or {}).items()
                                 if k != "_metrics_path"}
            promote, gate = self._evaluate_gates(metrics)
            report["gate"] = gate

            # 7) promote iff gates pass
            if promote:
                ok, msg = self._promote(run_dir)
                report["promoted"] = ok
                report["promote_detail"] = msg
                if ok:
                    self._save_state({"last_trained_label_count": labeled_at_start,
                                      "last_trained_ts": time.time()})
                    report["reason"] = "promoted"
                else:
                    report["reason"] = "gate_passed_but_promote_failed"
            else:
                report["reason"] = "gate_failed_kept_current_model"
                # advance the trigger baseline anyway so we don't retrain the
                # identical dataset on the very next tick; a fresh attempt waits
                # for genuinely new labels.
                self._save_state({"last_trained_label_count": labeled_at_start,
                                  "last_trained_ts": time.time(),
                                  "last_gate_fail": gate.get("reason")})
            return report
        except Exception as e:  # pragma: no cover - defensive
            report["reason"] = f"exception:{type(e).__name__}:{e}"
            self.last_error = f"{type(e).__name__}: {e}"
            return report
        finally:
            report["finished_at"] = time.time()
            report["duration_sec"] = round(report["finished_at"] - started, 1)
            self.last_run = report
            self._running = False

    def start_async(self, force=False):
        """Kick the pipeline in a daemon thread. Returns immediately with a
        started/blocked flag (mirrors the evolve endpoints' guard)."""
        with self._lock:
            if self._running:
                return {"started": False, "reason": "a training run is already in progress"}
        if not force:
            ok, reason = self.should_run()
            if not ok:
                return {"started": False, "reason": reason}
        threading.Thread(target=self.run_pipeline, kwargs={"force": force},
                         daemon=True).start()
        return {"started": True}

    # ---------------- introspection ----------------
    def status(self):
        return {
            "enabled": self.enabled(),
            "running": self._running,
            "lab_cmd": self._lab_cmd(),
            "min_new_labels": self._min_new_labels(),
            "gate": self._gate(),
            "trigger": self.trigger_status(),
            "last_run": self.last_run,
            "last_error": self.last_error,
        }


trainer = MLTrainer()
