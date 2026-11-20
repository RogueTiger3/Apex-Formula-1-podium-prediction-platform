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


def test_numpy_values_are_stored_as_numbers(synthetic_db):
    import numpy as np
    from apex import db
    with db.session() as conn:
        db.upsert(conn, "race_weather", [dict(race_id="x", kind="forecast", temp_c=np.float64(20.5),
                                              precip_mm=np.float64("nan"), precip_prob_pct=np.int64(35))])
        conn.execute("INSERT INTO race_weather (race_id, kind, precip_prob_pct) VALUES ('y', 'forecast', ?)",
                     (np.int64(7).tobytes(),))  # what an older version wrote
    db.init_db()  # repairs old blobs
    rows = db.query("SELECT race_id, typeof(precip_prob_pct) t, precip_prob_pct, precip_mm FROM race_weather ORDER BY race_id")
    assert "blob" not in set(rows["t"])
    assert list(rows["precip_prob_pct"]) == [35, 7]
    assert rows["precip_mm"].isna().all()
