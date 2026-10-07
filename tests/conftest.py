"""Shared test setup: point the DB at a temp file and initialise schema so
modules that log order/trade events during tests don't hit missing tables.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# use an isolated DB before importing anything that touches config.DB_PATH
_tmp = tempfile.mkdtemp(prefix="cryptomind-test-")
import app.config as _config
_config.DB_PATH = os.path.join(_tmp, "test.db")

import app.db as _db
_db.DB_PATH = _config.DB_PATH
_db.init()

# isolate the tunables override file so tests that call tunables.update() never
# clobber the operator's real tunables.json (it lives at the repo root).
import app.tunables as _tunables
_tunables.PATH = os.path.join(_tmp, "tunables.json")
_tunables._overrides = None

# likewise isolate operator settings so tests that flip settings (e.g. the LLM
# advisor toggle) never touch the real settings.json at the repo root.
import app.settings as _settings
_settings.SETTINGS_PATH = os.path.join(_tmp, "settings.json")
_settings._settings = None

# the replay's on-disk per-bar cache (app/backtest/bar_cache.py) goes to a
# temp dir too, so tests never read or write the operator's .cache/
import app.backtest.bar_cache as _bar_cache
_bar_cache.DIR = os.path.join(_tmp, "replay_bars")
