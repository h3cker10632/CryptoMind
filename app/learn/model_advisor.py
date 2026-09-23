"""Optional ML-model advisor — a trained crypto_ml_lab model as ONE more vote.

Phase 2 of the crypto_ml_lab integration (see docs/crypto_ml_lab_integration.md).
Phase 1 emits a training-grade dataset; the operator trains + validates a model
OFFLINE in crypto_ml_lab; this advisor round-trips that validated artifact back
into CryptoMind as a single directional lean that flows into the ensemble as the
`model` strategy vote — weighted by the Thompson bandit exactly like every other
sleeve. It is NEVER the driver.

Design (deliberately identical governance to the LLM advisor):
  * OFF by default (opt-in via the `model_advisor_enabled` setting).
  * Inert unless a model artifact exists AND its deps are importable — never
    blocks, never raises into the decision loop. `lean()` returns a CACHED value;
    a slow background task does the (heavy) inference out of band.
  * Leans are clamped to [-1, 1] and expire after a TTL so a stale prediction
    can't dominate; on any error the lean decays to 0.
  * NO TRAINING/SERVING SKEW: features are built with crypto_ml_lab's OWN
    `features.make_features` + `model_matrix` when the package is importable, so
    inference matches training exactly. Columns are aligned to the artifact's
    saved feature list. If crypto_ml isn't importable the advisor stays inert
    (and says so in stats) rather than guessing with a divergent feature set.

Artifact layout (produced by `crypto_ml.cli train`):
  <model_dir>/model.joblib     — the trained MultiTaskBaseline (or compatible)
  <model_dir>/metadata.json    — {"features": [...], "horizons": [...]}
Default <model_dir> is `model_artifact/` at the repo root; override with the
CRYPTOMIND_MODEL_DIR env var.
"""
import json
import math
import os
import time


class ModelAdvisor:
    def __init__(self):
        self._leans = {}          # product -> {"lean": float, "ts": float, "why": str}
        self.last_refresh = 0.0
        self.last_error = ""
        self.calls = 0
        self._model = None        # lazily loaded artifact
        self._meta = None
        self._loaded_from = None
        self._load_failed = False

    # ---------------- config ----------------
    @staticmethod
    def enabled():
        try:
            from .. import settings
            return bool(settings.get("model_advisor_enabled"))
        except Exception:
            return False

    @classmethod
    def model_dir(cls):
        env = os.environ.get("CRYPTOMIND_MODEL_DIR")
        if env:
            return env
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "model_artifact")

    @classmethod
    def _model_path(cls):
        return os.path.join(cls.model_dir(), "model.joblib")

    @classmethod
    def _meta_path(cls):
        return os.path.join(cls.model_dir(), "metadata.json")

    @staticmethod
    def _ttl():
        from ..tunables import tv
        return float(tv("model_lean_ttl_sec"))

    @staticmethod
    def _cadence():
        from ..tunables import tv
        return float(tv("model_refresh_sec"))

    @staticmethod
    def _rug_veto():
        from ..tunables import tv
        return float(tv("model_rug_veto"))

    @staticmethod
    def _return_scale():
        from ..tunables import tv
        return float(tv("model_return_scale"))

    def artifact_present(self):
        return os.path.exists(self._model_path())

    # ---------------- model loading (lazy, cached) ----------------
    def _load(self):
        """Load the joblib artifact + metadata once. Returns the model or None.
        Never raises; records the reason in last_error."""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        if not self.artifact_present():
            self.last_error = f"no model artifact at {self._model_path()}"
            return None
        try:
            import joblib
        except Exception as e:
            self._load_failed = True
            self.last_error = f"joblib not importable: {e}"
            return None
        try:
            self._model = joblib.load(self._model_path())
        except Exception as e:
            self._load_failed = True
            self.last_error = f"failed to load model: {e}"
            return None
        try:
            if os.path.exists(self._meta_path()):
                with open(self._meta_path()) as f:
                    self._meta = json.load(f)
        except Exception:
            self._meta = None
        self._loaded_from = self._model_path()
        return self._model

    def set_model(self, model, meta=None):
        """Inject a model directly (used by tests / programmatic loading)."""
        self._model = model
        self._meta = meta
        self._loaded_from = "injected"
        self._load_failed = False

    def configured(self):
        """Enabled AND a usable model is loadable."""
        return self.enabled() and self._load() is not None

    # ---------------- read path (hot, never blocks) ----------------
    def lean(self, product):
        """Cached directional lean in [-1, 1] for the signal engine. Returns 0.0
        when disabled, unconfigured, missing, or expired — so `strat_model` stays
        silent (and is excluded by the engine's active-strategy renorm) until a
        fresh prediction exists."""
        if not self.enabled():
            return 0.0
        e = self._leans.get(product)
        if not e:
            return 0.0
        if time.time() - e["ts"] > self._ttl():
            return 0.0
        return max(-1.0, min(1.0, e["lean"]))

    def set_lean(self, product, lean, why=""):
        self._leans[product] = {"lean": max(-1.0, min(1.0, float(lean))),
                                "ts": time.time(), "why": why}

    # ---------------- feature building (no skew) ----------------
    def _build_matrix(self, product, candles, book=None, sentiment=None):
        """Build the model input row for `product` from recent candle history,
        using crypto_ml_lab's OWN feature pipeline for exact train/serve parity.

        Columns mirror the Phase-1 export event schema (price/volume plus the
        optional liquidity/imbalance/sentiment the lab's make_features expects);
        the live book snapshot is stamped only on the last bar (no back-fill).

        Returns a 1-row matrix (last bar) aligned to the artifact's saved feature
        order, or None if crypto_ml is unavailable or the history is too short.
        NEVER raises.
        """
        try:
            import pandas as pd
            from crypto_ml.features import make_features, model_matrix
        except Exception as e:
            self.last_error = (f"crypto_ml/pandas not importable ({e}); install "
                               "crypto_ml_lab for exact-parity inference")
            return None
        if not candles or len(candles) < 20:
            return None
        book = book or {}
        liq = (book.get("bid_depth", 0.0) + book.get("ask_depth", 0.0)) or None
        n = len(candles)
        # candle = [ts, low, high, open, close, volume]
        rows = []
        for i, c in enumerate(candles):
            try:
                is_last = (i == n - 1)
                rows.append({
                    "ts": pd.to_datetime(float(c[0]), unit="s", utc=True),
                    "symbol": product,
                    "price": float(c[4]),
                    "volume": float(c[5]),
                    # live-snapshot fields only meaningful on the last bar
                    "liquidity": float(liq) if (is_last and liq) else None,
                    "buy_volume": None,
                    "sell_volume": None,
                    "bot_signal": float(sentiment) if (is_last and sentiment is not None) else 0.0,
                })
            except (IndexError, TypeError, ValueError):
                continue
        if len(rows) < 20:
            return None
        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        feats = make_features(df)
        X, cols = model_matrix(feats)
        # Align to the artifact's trained feature order when we have it.
        want = (self._meta or {}).get("features")
        if want:
            for c in want:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[want]
        return X.iloc[[-1]]

    def _predict_lean(self, product, candles, book=None, sentiment=None):
        """Run the model on the latest bar and derive a lean in [-1, 1].

        The lean is tanh(predicted_shortest_horizon_return * scale), vetoed to 0
        when the model's rug/risk probability exceeds the veto threshold — the
        same signal shaping crypto_ml_lab's own backtest uses. Returns
        (lean, why) or (None, reason)."""
        model = self._load()
        if model is None:
            return None, self.last_error or "no model"
        X = self._build_matrix(product, candles, book=book, sentiment=sentiment)
        if X is None:
            return None, self.last_error or "insufficient history / no features"
        try:
            self.calls += 1
            pred = model.predict(X)
        except Exception as e:
            return None, f"predict failed: {e}"
        ret0, risk = self._interpret(pred)
        if ret0 is None:
            return None, "unrecognised model output shape"
        if risk is not None and risk >= self._rug_veto():
            return 0.0, f"rug/risk veto ({risk:.2f}>={self._rug_veto():.2f})"
        lean = math.tanh(ret0 * self._return_scale())
        return max(-1.0, min(1.0, lean)), f"E[r0]={ret0:+.4f} risk={risk}"

    @staticmethod
    def _interpret(pred):
        """Normalise a model's output into (shortest_horizon_return, risk_prob).

        Supports crypto_ml_lab's MultiTaskBaseline ({"returns": 2d, "risk": 1d})
        and a plain regressor/array. Returns (None, None) if unusable."""
        try:
            if isinstance(pred, dict):
                rets = pred.get("returns")
                risk = pred.get("risk")
                r0 = float(_first_scalar(rets))
                rk = None if risk is None else float(_first_scalar(risk))
                return r0, rk
            # plain array-like of predicted returns
            return float(_first_scalar(pred)), None
        except Exception:
            return None, None

    # ---------------- refresh path (cold, out of band) ----------------
    async def refresh(self, market, products):
        """Slow, best-effort refresh of the lean cache. Only runs when enabled +
        configured and past the cadence. Any per-product failure is swallowed and
        leaves the previous (possibly-expiring) lean in place."""
        if not self.configured():
            return 0
        if time.time() - self.last_refresh < self._cadence():
            return 0
        self.last_refresh = time.time()
        updated = 0
        for p in list(products):
            candles = list(getattr(market, "candles", {}).get(p, []))
            book = getattr(market, "books", {}).get(p)
            sent = None
            try:
                from ..nlp.sentiment import nlp
                sent = nlp.asset_sentiment.get(p, {}).get("score")
            except Exception:
                pass
            try:
                lean, why = self._predict_lean(p, candles, book=book, sentiment=sent)
                if lean is not None:
                    self.set_lean(p, lean, why)
                    updated += 1
            except Exception as e:
                self.last_error = str(e)[:200]
        if updated:
            from .. import db
            db.log_event("learn", f"Model advisor refreshed {updated} leans "
                                  f"(fed to the bandit as the 'model' arm)")
        return updated

    # ---------------- introspection ----------------
    def stats(self):
        return {
            "enabled": self.enabled(),
            "configured": self.configured(),
            "artifact_present": self.artifact_present(),
            "model_dir": self.model_dir(),
            "loaded_from": self._loaded_from,
            "features_expected": len((self._meta or {}).get("features", []) or []),
            "horizons": (self._meta or {}).get("horizons"),
            "cached_leans": {p: round(e["lean"], 3)
                             for p, e in self._leans.items()},
            "last_refresh": self.last_refresh,
            "calls": self.calls,
            "last_error": self.last_error,
        }


def _first_scalar(x):
    """Return the first scalar of a possibly-nested array-like (row 0, col 0)."""
    cur = x
    for _ in range(4):
        try:
            cur = cur[0]
        except (TypeError, IndexError, KeyError):
            break
    return cur


advisor = ModelAdvisor()
