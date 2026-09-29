"""Standalone Polymarket prediction-market sleeve (paper now, live-ready seam).

Kept entirely separate from the crypto trading core: its own broker, sizer, and
learner instance, wired to Polymarket's public Gamma/CLOB data. It REUSES the
crypto book's learning machinery (RegimeBandit + an online-model-style learner)
without sharing its books, so the two never cross-contaminate. Off by default.
"""
from .engine import engine
from .broker import broker
from .learner import learner

__all__ = ["engine", "broker", "learner"]
