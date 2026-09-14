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
