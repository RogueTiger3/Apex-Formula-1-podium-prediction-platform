"""Open-Meteo weather (free, no key). https://open-meteo.com/

* Forecasts (up to 16 days ahead) for upcoming races.
* The ERA5-based archive for past races (used for training before OpenF1 track
  weather exists, i.e. before 2023).

Values are averaged over the two-hour race window at the circuit's coordinates.
Open-Meteo enforces a daily request quota per IP address; when it is exhausted
the functions return None and the rest of the pipeline carries on.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .. import config, db
from .http import FetchError, get_json

HOURLY = "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"


def _window(data: dict, start: datetime) -> dict | None:
    h = pd.DataFrame(data.get("hourly", {}))
    if h.empty:
        return None
    h["time"] = pd.to_datetime(h["time"], utc=True)
    s = pd.Timestamp(start).floor("h")
    w = h[(h["time"] >= s) & (h["time"] <= s + pd.Timedelta(hours=2))]
    if w.empty or w["temperature_2m"].isna().all():
        return None
    out = dict(temp_c=w["temperature_2m"].mean(), precip_mm=w["precipitation"].sum(),
               wind_kmh=w["wind_speed_10m"].mean(), humidity_pct=w["relative_humidity_2m"].mean(),
               precip_prob_pct=None)
    if "precipitation_probability" in w:
        out["precip_prob_pct"] = w["precipitation_probability"].max()
    return out


def race_window(lat: float, lon: float, start_iso: str, forecast: bool) -> tuple[dict | None, str]:
    start = datetime.fromisoformat(start_iso)
    day = start.date().isoformat()
    end_day = (start + timedelta(hours=3)).date().isoformat()
    params = dict(latitude=lat, longitude=lon, start_date=day, end_date=end_day,
                  hourly=HOURLY, timezone="UTC")
    try:
        if forecast:
            params["hourly"] = HOURLY + ",precipitation_probability"
            data = get_json(config.OPEN_METEO_FORECAST, params, max_age=3 * 3600)
        else:
            data = get_json(config.OPEN_METEO_ARCHIVE, params, max_age=None)
    except FetchError as exc:
        return None, str(exc)
    if not data:
        return None, "empty response"
    return _window(data, start), ""
