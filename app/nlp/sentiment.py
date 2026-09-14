"""NLP module — lexicon-based crypto sentiment + narrative detection.

Lightweight (no model downloads) so it runs anywhere; the interface is the
same one a FinBERT/CryptoBERT upgrade would slot into.
"""
import re, time, math
from collections import Counter

BULL = {
    "surge": 2, "soar": 2, "rally": 2, "breakout": 2, "bullish": 2, "ath": 2,
    "all-time": 1.5, "record": 1.5, "adoption": 1.5, "etf": 1, "approval": 1.5,
    "gain": 1, "gains": 1, "up": .5, "rise": 1, "rises": 1, "rising": 1, "jump": 1.5,
    "pump": 1, "moon": 1.5, "buy": .5, "buying": 1, "accumulate": 1.5, "inflow": 1.5,
    "institutional": 1, "upgrade": 1, "partnership": 1, "launch": .5, "growth": 1,
    "bull": 1.5, "optimism": 1.5, "recovery": 1, "rebound": 1.5, "halving": .5,
}
BEAR = {
    "crash": 2, "plunge": 2, "dump": 2, "bearish": 2, "selloff": 2, "sell-off": 2,
    "hack": 2, "exploit": 2, "scam": 2, "fraud": 2, "lawsuit": 1.5, "sec": .5,
    "ban": 1.5, "fear": 1, "fud": 1, "drop": 1, "drops": 1, "fall": 1, "falls": 1,
    "down": .5, "decline": 1, "liquidation": 1.5, "liquidations": 1.5, "outflow": 1.5,
    "bankrupt": 2, "bankruptcy": 2, "collapse": 2, "warning": 1, "risk": .5,
    "sell": .5, "selling": 1, "panic": 2, "bear": 1.5, "correction": 1, "tumble": 1.5,
}
WORD_RE = re.compile(r"[a-z][a-z\-']+")

# macro lexicon: risk-on (+) vs risk-off (−) for crypto as a risk asset
MACRO_POS = {
    "cut": 1.5, "cuts": 1.5, "cutting": 1.5, "dovish": 2, "easing": 1.5,
    "stimulus": 1.5, "cooling": 1, "cooled": 1, "slowing": .5, "soft": .5,
    "approval": 1.5, "approved": 1.5, "adoption": 1.5, "clarity": 1,
    "friendly": 1, "legalize": 1.5, "etf": .5, "growth": .5, "rally": 1,
}
MACRO_NEG = {
    "hike": 1.5, "hikes": 1.5, "hawkish": 2, "tightening": 1.5, "inflation": .5,
    "recession": 2, "crackdown": 2, "lawsuit": 1.5, "ban": 2, "sues": 1.5,
    "enforcement": 1.5, "restrictions": 1.5, "hot": .5, "surge": .3,
    "tariff": 1, "tariffs": 1, "war": 1, "crisis": 2, "default": 1.5,
}


def score_macro(text: str) -> float:
    """Risk-on/risk-off score in [-1, 1] for macro headlines."""
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    s = sum(MACRO_POS.get(w, 0) for w in words) - sum(MACRO_NEG.get(w, 0) for w in words)
    return max(-1.0, min(1.0, s / max(3.0, len(words) * 0.35)))


def score_text(text: str) -> float:
    """Sentiment in [-1, 1]."""
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    s = sum(BULL.get(w, 0) for w in words) - sum(BEAR.get(w, 0) for w in words)
    return max(-1.0, min(1.0, s / max(3.0, len(words) * 0.35)))


class NLPEngine:
    def __init__(self):
        self.asset_sentiment = {}   # product -> {score, n_docs}
        self.market_sentiment = 0.0
        self.macro_sentiment = 0.0  # risk-on/risk-off from macro headlines
        self.macro_docs = 0
        self.narratives = []        # top emerging topics
        self.last_update = 0.0

    def process(self, documents, fear_greed):
        per_asset, all_scores, macro_scores, words = {}, [], [], Counter()
        stop = {"the", "and", "for", "with", "this", "that", "from", "will",
                "has", "have", "are", "its", "was", "new", "how", "why", "what",
                "you", "your", "after", "amid", "over", "into", "more", "than",
                "says", "could", "here", "week", "just", "not", "but", "all"}
        for d in documents:
            if d.get("kind") == "macro":
                macro_scores.append(score_macro(d["title"]))
                continue                      # macro docs don't tag coins
            s = score_text(d["title"])
            all_scores.append(s)
            for a in d["assets"]:
                per_asset.setdefault(a, []).append(s)
            for w in WORD_RE.findall(d["title"].lower()):
                if len(w) > 3 and w not in stop:
                    words[w] += 1

        self.asset_sentiment = {
            a: {"score": sum(v) / len(v), "n_docs": len(v)}
            for a, v in per_asset.items()
        }
        base = sum(all_scores) / len(all_scores) if all_scores else 0.0
        self.macro_sentiment = (sum(macro_scores) / len(macro_scores)
                                if macro_scores else 0.0)
        self.macro_docs = len(macro_scores)
        # blend headline sentiment with Fear & Greed and macro risk backdrop
        fg = ((fear_greed["value"] - 50) / 50) if fear_greed else 0.0
        self.market_sentiment = 0.5 * base + 0.3 * fg + 0.2 * self.macro_sentiment
        self.narratives = [{"topic": w, "count": n} for w, n in words.most_common(10)]
        self.last_update = time.time()

    def asset_score(self, product):
        d = self.asset_sentiment.get(product)
        if not d:
            return self.market_sentiment * 0.5, 0
        # shrink toward market sentiment when few docs
        w = min(1.0, d["n_docs"] / 5)
        return w * d["score"] + (1 - w) * self.market_sentiment, d["n_docs"]


nlp = NLPEngine()
