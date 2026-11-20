import gzip
import json

from apex.sources import http, jolpica


def test_lap_time_parsing():
    assert jolpica.lap_str_to_s("1:30.031") == 90.031
    assert jolpica.lap_str_to_s("59.5") == 59.5
    assert jolpica.lap_str_to_s("") is None
    assert jolpica.lap_str_to_s(None) is None


def test_http_uses_cache_and_never_refetches(synthetic_db, monkeypatch):
    calls = []

    class Resp:
        status_code = 200
        headers = {}
        text = ""

        def json(self):
            return {"ok": 1}

    def fake_get(url, timeout):
        calls.append(url)
        return Resp()

    monkeypatch.setattr(http._session, "get", fake_get)
    url = "https://api.jolpi.ca/ergast/f1/2020.json"
    assert http.get_json(url) == {"ok": 1}
    assert http.get_json(url) == {"ok": 1}
    assert len(calls) == 1


def test_http_falls_back_to_stale_cache(synthetic_db, monkeypatch):
    url = "https://api.openf1.org/v1/sessions?year=2024"
    path = http._cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump([{"cached": True}], f)

    class Resp:
        status_code = 503
        headers = {}
        text = "down"

    monkeypatch.setattr(http._session, "get", lambda url, timeout: Resp())
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.get_json(url, max_age=0) == [{"cached": True}]
