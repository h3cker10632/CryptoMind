"""Market-scoped, bounded research evidence for Polymarket forecasts."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import re
import threading
import time
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import httpx


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_UA = {"User-Agent": "CryptoMind/1.0 (+paper-research)"}
_STOP_WORDS = {
	"about", "after", "against", "among", "before", "being", "championship",
	"could", "does", "from", "have", "into", "market", "over", "should",
	"team", "than", "that", "their", "there", "these", "they", "this",
	"under", "will", "with", "would", "win", "yes", "no",
}
_GENERIC_SINGLE_TERMS = {
	"city", "county", "country", "council", "election", "government",
	"league", "national", "president", "price", "state", "united",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{2,}", re.I)
_TAG_RE = re.compile(r"<[^>]*>")
_MAX_DOCS = 12
_MAX_QUERY_CACHE = 300
_MAX_AGE_SEC = 72 * 60 * 60
_QUERY_REFRESH_SEC = 15 * 60
_BATCH_REFRESH_SEC = 5 * 60
_MAX_BATCH_QUERIES = 20


def _terms(text: str) -> list[str]:
	return list(dict.fromkeys(
		token.lower().strip("'-") for token in _TOKEN_RE.findall(text)
		if token.lower() not in _STOP_WORDS and len(token.strip("'-")) >= 4))


def market_query(market: dict) -> str:
	"""Build a compact, normalized query from the question and category."""
	question = str(market.get("question") or "")
	words = _terms(question)
	proper_names = re.findall(
		r"\b(?:[A-Z][a-z0-9&'-]+\s+){0,2}[A-Z][a-z0-9&'-]+\b", question)
	selected = []
	for phrase in proper_names + words[:6] + _terms(str(market.get("category") or ""))[:1]:
		normalized = " ".join(_terms(phrase))
		if normalized and normalized not in selected:
			selected.append(normalized)
	return " ".join(selected)[:180]


def _canonical_url(url: str) -> str:
	try:
		parts = urlsplit(url.strip())
		if parts.scheme not in ("http", "https") or not parts.netloc:
			return ""
		query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
				 if not key.lower().startswith("utm_")
				 and key.lower() not in {"gclid", "fbclid", "oc", "hl", "gl"}]
		return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
						   parts.path.rstrip("/"), urlencode(sorted(query)), ""))
	except (TypeError, ValueError):
		return ""


def _local_name(tag: str) -> str:
	return tag.rsplit("}", 1)[-1].lower()


def _child_text(node, name: str) -> str:
	for child in node:
		if _local_name(child.tag) == name:
			return (child.text or child.attrib.get("href") or "").strip()
	return ""


def _published_ts(value: str) -> float:
	if not value:
		return 0.0
	try:
		parsed = parsedate_to_datetime(value)
		if parsed.tzinfo is None:
			parsed = parsed.replace(tzinfo=timezone.utc)
		return parsed.timestamp()
	except (TypeError, ValueError, OverflowError):
		try:
			parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
			if parsed.tzinfo is None:
				parsed = parsed.replace(tzinfo=timezone.utc)
			return parsed.timestamp()
		except (TypeError, ValueError, OverflowError):
			return 0.0


def _parse_feed(xml_text: str, query_terms: list[str], now: float) -> list[dict]:
	try:
		root = ET.fromstring(xml_text)
	except (ET.ParseError, TypeError, ValueError):
		return []
	seen, docs = set(), []
	for item in root.iter():
		if _local_name(item.tag) not in {"item", "entry"}:
			continue
		title = html.unescape(_child_text(item, "title"))[:300]
		link = _child_text(item, "link")
		published = (_child_text(item, "pubdate") or _child_text(item, "published")
					 or _child_text(item, "updated"))
		published_ts = _published_ts(published)
		if not title or not link or not published_ts:
			continue
		if published_ts > now + 300 or now - published_ts > _MAX_AGE_SEC:
			continue
		canonical = _canonical_url(link)
		if not canonical or canonical in seen:
			continue
		excerpt = _child_text(item, "description") or _child_text(item, "summary")
		excerpt = html.unescape(_TAG_RE.sub(" ", excerpt))
		excerpt = " ".join(excerpt.split())[:320]
		searchable_terms = set(_terms(f"{title} {excerpt}"))
		matched = [term for term in query_terms if term in searchable_terms]
		if (not matched or (len(set(matched)) == 1
				and matched[0] in _GENERIC_SINGLE_TERMS)):
			continue
		source = ""
		for child in item:
			if _local_name(child.tag) == "source":
				source = (child.text or "").strip()
				break
		seen.add(canonical)
		digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
		docs.append({
			"id": digest,
			"publisher": source[:100] or (urlsplit(canonical).hostname or "unknown"),
			"title": title,
			"url": canonical,
			"published_at": datetime.fromtimestamp(
				published_ts, timezone.utc).isoformat(),
			"published_ts": published_ts,
			"retrieved_at": now,
			"matched_terms": matched[:12],
			"excerpt": excerpt,
		})
		if len(docs) >= _MAX_DOCS:
			break
	return docs


class PMResearchCache:
	"""Fetch and retain bounded market-specific RSS evidence off the quote path."""

	def __init__(self, fetcher=None, max_workers=3):
		self._executor = ThreadPoolExecutor(
			max_workers=max(1, min(int(max_workers), 3)),
			thread_name_prefix="pm-research")
		self._fetcher = fetcher or self._fetch_rss
		self._lock = threading.RLock()
		self._cache: dict[str, dict] = {}
		self._pending: dict[str, Future] = {}
		self._condition_queries: dict[str, tuple[str, list[str]]] = {}
		self._last_batch_refresh = 0.0
		self._last_error = ""
		self._request_count = 0
		self._cursor = 0

	def _fetch_rss(self, query: str) -> str:
		url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
		with httpx.Client(timeout=8.0, headers=_UA, follow_redirects=True) as client:
			response = client.get(url)
			response.raise_for_status()
			return response.text

	def _collect_done(self):
		now = time.time()
		with self._lock:
			for query, future in list(self._pending.items()):
				if not future.done():
					continue
				del self._pending[query]
				old = self._cache.get(query, {"docs": []})
				try:
					xml_text = future.result()
					docs = _parse_feed(xml_text, old.get("query_terms", []), now)
					self._cache[query] = {
						"docs": docs or old.get("docs", []),
						"query_terms": old.get("query_terms", []),
						"fetched_at": now,
					}
					self._last_error = ""
				except Exception as exc:  # noqa: BLE001
					old["fetched_at"] = now
					self._cache[query] = old
					self._last_error = f"{type(exc).__name__}: {exc}"[:200]
			if len(self._cache) > _MAX_QUERY_CACHE:
				oldest = sorted(self._cache, key=lambda key: self._cache[key]["fetched_at"])
				for query in oldest[:len(self._cache) - _MAX_QUERY_CACHE]:
					self._cache.pop(query, None)

	def refresh(self, markets: list[dict]) -> int:
		"""Schedule up to 20 unique queries per five-minute batch; never block on RSS."""
		self._collect_done()
		now = time.time()
		unique = {}
		condition_queries = {}
		for market in markets:
			condition_id = str(market.get("condition_id") or "")
			query = market_query(market)
			terms = _terms(str(market.get("question") or ""))
			if not condition_id or not query or not terms:
				continue
			unique.setdefault(query, terms)
			condition_queries[condition_id] = (query, terms)
		with self._lock:
			self._condition_queries = condition_queries
		if now - self._last_batch_refresh < _BATCH_REFRESH_SEC or not unique:
			return 0
		keys = list(unique)
		start = self._cursor % len(keys)
		ordered = keys[start:] + keys[:start]
		scheduled = 0
		with self._lock:
			for query in ordered:
				cached = self._cache.get(query)
				if query in self._pending:
					continue
				if cached and now - cached["fetched_at"] < _QUERY_REFRESH_SEC:
					continue
				if scheduled >= _MAX_BATCH_QUERIES:
					break
				self._cache.setdefault(query, {"docs": [], "fetched_at": 0.0})
				self._cache[query]["query_terms"] = unique[query]
				self._pending[query] = self._executor.submit(self._fetcher, query)
				scheduled += 1
				self._request_count += 1
		self._cursor = (start + max(1, scheduled)) % len(keys)
		self._last_batch_refresh = now
		return scheduled

	def evidence(self, condition_id: str) -> list[dict]:
		self._collect_done()
		with self._lock:
			query_info = self._condition_queries.get(str(condition_id))
			if not query_info:
				return []
			query, terms = query_info
			cached = dict(self._cache.get(query) or {})
		now = time.time()
		return [dict(doc) for doc in cached.get("docs", [])
				if 0 <= now - doc.get("published_ts", 0) <= _MAX_AGE_SEC
				and any(term in (doc.get("matched_terms") or []) for term in terms)]

	def stats(self) -> dict:
		self._collect_done()
		covered = sum(bool(self.evidence(cid)) for cid in self._condition_queries)
		total = len(self._condition_queries)
		fresh_docs = [doc for cached in self._cache.values()
					  for doc in cached.get("docs", [])
					  if 0 <= time.time() - doc.get("published_ts", 0) <= _MAX_AGE_SEC]
		return {
			"markets_seen": total,
			"markets_covered": covered,
			"coverage": round(covered / total, 4) if total else 0.0,
			"fresh_documents": len(fresh_docs),
			"pending_queries": len(self._pending),
			"requests": self._request_count,
			"last_error": self._last_error,
		}

	def close(self):
		self._executor.shutdown(wait=False, cancel_futures=True)


research_cache = PMResearchCache()