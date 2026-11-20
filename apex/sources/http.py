"""Shared HTTP client: disk cache, per-host rate limiting, retries and request logging.

Every network request is written to the `fetch_log` table so failed requests are
visible on the Data Management page.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import threading
import time
from collections import deque
from urllib.parse import urlencode, urlparse

import requests

from .. import config, db

log = logging.getLogger("apex.http")

# (max requests, per seconds) windows for each host. Values sit below the
# published free-tier limits so a full backfill never trips them.
RATE_LIMITS = {
    "api.jolpi.ca": [(4, 1.0), (450, 3600.0)],
    "api.openf1.org": [(3, 1.0), (28, 60.0)],
    "api.open-meteo.com": [(5, 1.0), (500, 60.0)],
    "archive-api.open-meteo.com": [(5, 1.0), (500, 60.0)],
}
SOURCE_NAMES = {
    "api.jolpi.ca": "Jolpica F1",
    "api.openf1.org": "OpenF1",
    "api.open-meteo.com": "Open-Meteo forecast",
    "archive-api.open-meteo.com": "Open-Meteo archive",
}


class FetchError(RuntimeError):
    pass


class _Limiter:
    def __init__(self):
        self._hist: dict[str, deque] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        windows = RATE_LIMITS.get(host, [(5, 1.0)])
        with self._lock:
            hist = self._hist.setdefault(host, deque())
            while True:
                now = time.monotonic()
                longest = max(w for _, w in windows)
                while hist and now - hist[0] > longest:
                    hist.popleft()
                sleep_for = 0.0
                for limit, window in windows:
                    recent = [t for t in hist if now - t <= window]
                    if len(recent) >= limit:
                        sleep_for = max(sleep_for, window - (now - recent[0]) + 0.05)
                if sleep_for <= 0:
                    hist.append(now)
                    return
                if sleep_for > 5:
                    log.info("Rate limit for %s: waiting %.0fs", host, sleep_for)
                time.sleep(sleep_for)


_limiter = _Limiter()
_session = requests.Session()
_session.headers["User-Agent"] = "apex-f1-analytics/1.0 (open-source portfolio project)"


def _cache_path(url: str):
    h = hashlib.sha1(url.encode()).hexdigest()
    host = urlparse(url).netloc.replace(".", "_")
    return config.CACHE_DIR / host / f"{h}.json.gz"


def _log(url: str, status: int | None, ok: bool, from_cache: bool, message: str = "") -> None:
    try:
        with db.session() as conn:
            conn.execute(
                "INSERT INTO fetch_log (ts, source, url, http_status, ok, from_cache, message)"
                " VALUES (?,?,?,?,?,?,?)",
                (db.utcnow(), SOURCE_NAMES.get(urlparse(url).netloc, urlparse(url).netloc),
                 url, status, int(ok), int(from_cache), message[:500]))
    except Exception:  # logging must never break a fetch
        log.debug("could not write fetch_log", exc_info=True)


def get_json(url: str, params: dict | None = None, max_age: float | None = None,
             retries: int = 4, not_found_ok: bool = True):
    """GET a JSON document.

    max_age: seconds a cached copy stays valid. None = cache forever (used for
    historical data that will not change); 0 = always refetch.
    Returns None for 404 when not_found_ok (OpenF1 uses 404 for "no rows").
    """
    if params:
        url = f"{url}?{urlencode(params)}"
    path = _cache_path(url)
    if path.exists() and max_age != 0:
        age = time.time() - path.stat().st_mtime
        if max_age is None or age < max_age:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)

    host = urlparse(url).netloc
    last_err = ""
    status = None
    for attempt in range(retries + 1):
        _limiter.wait(host)
        try:
            resp = _session.get(url, timeout=60)
            status = resp.status_code
        except requests.RequestException as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(60, 2 ** attempt * 2))
            continue
        if status == 200:
            data = resp.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
            _log(url, status, True, False)
            return data
        if status == 404 and not_found_ok:
            _log(url, status, True, False, "no data (404)")
            return None
        last_err = resp.text[:300]
        if status in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else min(120, 5 * 2 ** attempt)
            if status == 429 and "Daily" in resp.text:
                break  # Open-Meteo daily quota: retrying today is pointless
            time.sleep(wait)
            continue
        break
    # Fall back to a stale cached copy rather than failing outright.
    if path.exists():
        _log(url, status, False, True, f"network failed, used stale cache: {last_err}")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    _log(url, status, False, False, last_err)
    raise FetchError(f"{url} -> {status}: {last_err}")
