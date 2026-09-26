"""Optional LLM advisor — ONE more opinion, weighted like any other strategy.

Design philosophy (deliberately the OPPOSITE of NOFX, whose *entire* strategy
IS the LLM): CryptoMind's edge is its learning loop, so we must NOT hand the
wheel to a language model. Instead the advisor contributes a single directional
lean per coin that flows into the ensemble as the `llm` strategy vote. The
Thompson bandit then weights it from realized trade PnL exactly like every
other sleeve — if the LLM adds edge it earns weight; if it doesn't, the
forgetting bandit decays it toward zero. It rides all our existing validation
instead of bypassing it, and it is fully reversible.

Safety properties:
  * OFF by default (opt-in via the `llm_advisor_enabled` setting).
  * Inert with no API key configured — never blocks, never raises into the
    decision loop. `lean()` returns a CACHED value; a slow background task does
    the (latency/cost-heavy) refresh out of band.
  * Leans are clamped to [-1, 1] and expire after a TTL so a stale opinion
    can't dominate; on any error the lean decays to 0.
"""
import os, time, json


class LLMAdvisor:
    def __init__(self):
        self._leans = {}          # product -> {"lean": float, "ts": float, "why": str}
        self.last_refresh = 0.0
        self.last_error = ""
        self.calls = 0

    # ---------------- config ----------------
    @staticmethod
    def enabled():
        try:
            from .. import settings
            return bool(settings.get("llm_advisor_enabled"))
        except Exception:
            return False

    KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "llm_key.txt")

    DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
    DEFAULT_MODEL = "gemini-2.5-flash"

    @staticmethod
    def _setting(key):
        """Read one setting, returning None (never raising) if unavailable."""
        try:
            from .. import settings
            v = settings.get(key)
            return v if v not in (None, "") else None
        except Exception:
            return None

    @classmethod
    def _api_key(cls):
        """Resolution order: env vars → dashboard Settings (.secrets.json) →
        llm_key.txt (gitignored, same pattern as api_token.txt)."""
        env = (os.environ.get("CRYPTOMIND_LLM_KEY")
               or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
        if env:
            return env.strip()
        setting = cls._setting("llm_api_key")
        if setting:
            return str(setting).strip()
        try:
            if os.path.exists(cls.KEY_PATH):
                with open(cls.KEY_PATH) as f:
                    t = f.read().strip()
                if t:
                    return t
        except OSError:
            pass
        return None

    @classmethod
    def _key_source(cls):
        """Report WHERE the active key comes from — env vars win over the
        Settings key, so this makes a 'wrong key' due to env shadowing obvious."""
        for name in ("CRYPTOMIND_LLM_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
            if os.environ.get(name):
                return f"env:{name}"
        if cls._setting("llm_api_key"):
            return "Settings"
        try:
            if os.path.exists(cls.KEY_PATH) and open(cls.KEY_PATH).read().strip():
                return "llm_key.txt"
        except OSError:
            pass
        return None

    @staticmethod
    def _ttl():
        from ..tunables import tv
        return float(tv("llm_lean_ttl_sec"))

    @staticmethod
    def _cadence():
        from ..tunables import tv
        return float(tv("llm_refresh_sec"))

    def configured(self):
        return self.enabled() and bool(self._api_key())

    # ---------------- read path (hot, never blocks) ----------------
    def lean(self, product):
        """Cached directional lean in [-1, 1] for the signal engine. Returns 0.0
        when disabled, unconfigured, missing, or expired — so `strat_llm` stays
        silent (and is excluded by the engine's active-strategy renorm) until a
        fresh opinion exists."""
        if not self.enabled():
            return 0.0
        e = self._leans.get(product)
        if not e:
            return 0.0
        if time.time() - e["ts"] > self._ttl():
            return 0.0
        return max(-1.0, min(1.0, e["lean"]))

    def set_lean(self, product, lean, why=""):
        """Inject a lean (used by the refresh path and by tests)."""
        self._leans[product] = {"lean": max(-1.0, min(1.0, float(lean))),
                                "ts": time.time(), "why": why}

    # ---------------- refresh path (cold, out of band) ----------------
    def build_context(self, product, market, nlp):
        """Compact, model-agnostic snapshot handed to the LLM. Kept small and
        numeric so it's cheap and reproducible."""
        f = market.features(product)
        if not f:
            return None
        sent = nlp.asset_score(product)
        return {
            "product": product,
            "price": round(f["price"], 6),
            "rsi": round(f["rsi"], 1),
            "macd_delta": round(f["macd_delta"], 6),
            "mom_1h": round(f["mom_1h"], 4),
            "mom_4h": round(f["mom_4h"], 4),
            "mtf_align": round(f.get("mtf_align", 0.0), 3),
            "vol_ratio": round(f["vol_ratio"], 2),
            "asset_sentiment": round(sent[0], 3),
            "regime": market.regime().get("label"),
        }

    async def refresh(self, market, nlp, products):
        """Slow, best-effort refresh of the lean cache. Only runs when enabled +
        configured and past the cadence. Any failure is swallowed (logged) and
        leaves the previous (possibly-expiring) leans in place."""
        if not self.configured():
            return 0
        if time.time() - self.last_refresh < self._cadence():
            return 0
        self.last_refresh = time.time()
        updated = 0
        for p in list(products):
            ctx = self.build_context(p, market, nlp)
            if not ctx:
                continue
            try:
                lean, why = await self._ask_model(ctx)
                if lean is not None:
                    self.set_lean(p, lean, why)
                    updated += 1
            except Exception as e:
                self.last_error = str(e)[:200]
        if updated:
            from .. import db
            db.log_event("learn", f"LLM advisor refreshed {updated} leans "
                                  f"(fed to the bandit as the 'llm' arm)")
        return updated

    @classmethod
    def _base_url(cls):
        base = (os.environ.get("CRYPTOMIND_LLM_BASE")
                or cls._setting("llm_api_base")
                or cls.DEFAULT_BASE).strip().rstrip("/")
        # Be forgiving about a base that already includes the call path: a user
        # pasting the FULL endpoint (…/chat/completions) is a classic cause of a
        # bare 404 once we append /chat/completions again. Strip it back off.
        for suffix in ("/chat/completions", "/completions", "/v1/chat/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        base = base.rstrip("/")
        # Anthropic's OpenAI-compatible API lives under /v1; a bare host 404s.
        if base.endswith("api.anthropic.com"):
            base += "/v1"
        return base

    @classmethod
    def _model(cls):
        return (os.environ.get("CRYPTOMIND_LLM_MODEL")
                or cls._setting("llm_model")
                or cls.DEFAULT_MODEL)

    async def _ask_model(self, ctx):
        """Query Gemini (OpenAI-compatible chat) for a directional lean.

        Defaults to Google's Gemini OpenAI-compat endpoint. Override with
        CRYPTOMIND_LLM_BASE / CRYPTOMIND_LLM_MODEL. Returns (None, "") on any
        non-parseable response so the caller keeps the previous cached lean.
        """
        import httpx
        self.calls += 1
        base = self._base_url()
        model = self._model()
        sys_prompt = (
            "You are a cautious crypto trading advisor. Given compact market "
            "features for one asset, respond with ONLY a JSON object "
            '{"lean": <number -1..1>, "why": "<short reason>"} where lean is '
            "your directional bias (-1 strong short … +1 strong long, 0 = no "
            "edge). Be conservative; prefer 0 when signals conflict."
        )
        payload_base = {
            "model": model,
            "temperature": 0.2,
            "max_tokens": 512,          # required by Anthropic's compat layer; harmless elsewhere
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(ctx)},
            ],
        }
        # Gemini's OpenAI-compat endpoint often 400s on response_format/json_object.
        # Try WITH it first (stricter JSON where supported); on a 400 retry WITHOUT
        # it — _parse tolerates fenced/loose JSON either way. Any HTTP error's real
        # body is captured into last_error so the operator sees the actual reason.
        async with httpx.AsyncClient(timeout=20) as client:
            for with_json_mode in (True, False):
                payload = dict(payload_base)
                if with_json_mode:
                    payload["response_format"] = {"type": "json_object"}
                url = f"{base}/chat/completions"
                try:
                    r = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {self._api_key()}"},
                        json=payload)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    self.last_error = ""              # success clears prior error
                    return self._parse(content)
                except httpx.HTTPStatusError as e:
                    body = ""
                    try:
                        body = e.response.text[:300]
                    except Exception:
                        body = ""
                    code = e.response.status_code
                    # include model + URL so a 404 (wrong path/model) is diagnosable
                    # at a glance instead of a bare status line with no body.
                    detail = body or "(empty body)"
                    hint = ""
                    if code == 404:
                        hint = (f" — model '{model}' not found, or wrong base URL "
                                f"({base}). Check Settings → LLM advisor.")
                    elif code in (401, 403):
                        src = self._key_source()
                        hint = (f" — key rejected. Active key came from {src}; note "
                                "env vars OVERRIDE the Settings key, so unset "
                                "CRYPTOMIND_LLM_KEY/GEMINI_API_KEY/OPENAI_API_KEY if "
                                "one is shadowing the key you entered.")
                    self.last_error = f"HTTP {code} at {url}: {detail}{hint}"
                    # a 400 while json-mode was on → retry once without it
                    if code == 400 and with_json_mode:
                        continue
                    return None, ""
                except Exception as e:               # network/timeout/parse
                    self.last_error = str(e)[:200]
                    return None, ""
        return None, ""

    @staticmethod
    def _parse(content):
        """Extract {lean, why} from a model response, tolerating code fences and
        surrounding prose (NOFX-style defensive parsing)."""
        if not content:
            return None, ""
        s = content.strip()
        # find the first {...} block
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j == -1 or j <= i:
            return None, ""
        try:
            obj = json.loads(s[i:j + 1])
        except Exception:
            return None, ""
        if "lean" not in obj:
            return None, ""
        try:
            lean = max(-1.0, min(1.0, float(obj["lean"])))
        except (TypeError, ValueError):
            return None, ""
        return lean, str(obj.get("why", ""))[:200]

    @classmethod
    def _provider(cls):
        b = cls._base_url()
        if "anthropic.com" in b:
            return "anthropic"
        if "openai.com" in b:
            return "openai"
        if "googleapis.com" in b or "generativelanguage" in b:
            return "gemini"
        return "custom"

    def stats(self):
        return {
            "enabled": self.enabled(),
            "configured": self.configured(),
            "provider": self._provider(),
            "key_source": self._key_source(),
            "model": self._model(),
            "cached_leans": {p: round(e["lean"], 3)
                             for p, e in self._leans.items()},
            "last_refresh": self.last_refresh,
            "calls": self.calls,
            "last_error": self.last_error,
        }


advisor = LLMAdvisor()
