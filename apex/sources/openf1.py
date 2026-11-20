"""OpenF1 API: lap timing, stints, weather and race control for 2023 onwards.

Docs: https://openf1.org/  Historical data is free and needs no key. (Live data
during a session is restricted to paid accounts, so Apex only reads sessions
after they have finished.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, db
from .http import get_json

SOURCE = "openf1"
SESSION_ALIASES = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Qualifying": "Q", "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ",
    "Sprint": "S", "Race": "R",
}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def sessions(year: int, max_age: float | None) -> list[dict]:
    return get_json(f"{config.OPENF1_BASE}/sessions", {"year": year}, max_age=max_age) or []


def _get(endpoint: str, session_key: int):
    return get_json(f"{config.OPENF1_BASE}/{endpoint}", {"session_key": session_key}, max_age=None) or []


def driver_info(session_key: int) -> list[dict]:
    return _get("drivers", session_key)


def lap_stats(session_key: int, is_race: bool) -> pd.DataFrame:
    """Per-driver pace, consistency, long-run pace and tyre degradation for one session."""
    laps = pd.DataFrame(_get("laps", session_key))
    if laps.empty or "lap_duration" not in laps:
        return pd.DataFrame()
    stints = pd.DataFrame(_get("stints", session_key))
    laps = laps[["driver_number", "lap_number", "lap_duration", "is_pit_out_lap"]].copy()
    laps = laps.sort_values(["driver_number", "lap_number"])
    laps["is_pit_out_lap"] = laps["is_pit_out_lap"].astype("boolean").fillna(False).astype(bool)
    # An in-lap is the lap before a pit-out lap.
    nxt = laps.groupby("driver_number")["is_pit_out_lap"].shift(-1)
    laps["is_in_lap"] = nxt.astype("boolean").fillna(False).astype(bool)
    base = laps["lap_duration"].notna() & ~laps["is_pit_out_lap"] & ~laps["is_in_lap"]
    if is_race:
        # Compare each lap with the field's median for the same lap number. Laps where
        # the whole field is slow (safety car, VSC, red flag) are dropped entirely, and
        # a driver's lap must be within 5% of the field median (removes incidents).
        lap_med = laps[base].groupby("lap_number")["lap_duration"].median()
        neutral = lap_med > lap_med.median() * 1.10
        med = laps["lap_number"].map(lap_med)
        ok = (laps["lap_number"] > 1) & ~laps["lap_number"].map(neutral).astype("boolean").fillna(True).astype(bool) \
            & (laps["lap_duration"] <= med * 1.05)
    else:
        # Practice/qualifying: drop cool-down and traffic laps (over 107% of own best).
        own_best = laps.groupby("driver_number")["lap_duration"].transform("min")
        ok = laps["lap_duration"] <= own_best * 1.07
    clean = laps[base & ok].copy()

    # Attach stint information (compound and tyre age) to each lap.
    clean["stint"] = np.nan
    clean["tyre_age"] = np.nan
    wet = {}
    n_stints = {}
    if not stints.empty:
        for _, s in stints.iterrows():
            if pd.isna(s.get("lap_start")) or pd.isna(s.get("lap_end")):
                continue
            m = (clean["driver_number"] == s["driver_number"]) & clean["lap_number"].between(s["lap_start"], s["lap_end"])
            clean.loc[m, "stint"] = s["stint_number"]
            clean.loc[m, "tyre_age"] = clean.loc[m, "lap_number"] - s["lap_start"] + (s.get("tyre_age_at_start") or 0)
            if str(s.get("compound", "")).upper() in WET_COMPOUNDS:
                wet[s["driver_number"]] = 1
        n_stints = stints.groupby("driver_number")["stint_number"].max().to_dict()

    out = []
    for drv, g in laps.groupby("driver_number"):
        c = clean[clean["driver_number"] == drv]
        long_laps, slopes, weights = [], [], []
        for _, sg in c.dropna(subset=["stint"]).groupby("stint"):
            if len(sg) >= 5:
                long_laps.append(sg["lap_duration"])
                if sg["tyre_age"].nunique() >= 5:
                    slope = np.polyfit(sg["tyre_age"], sg["lap_duration"], 1)[0]
                    slopes.append(slope)
                    weights.append(len(sg))
        long = pd.concat(long_laps) if long_laps else pd.Series(dtype=float)
        out.append(dict(
            car_number=int(drv), n_laps=int(g["lap_duration"].notna().sum()),
            best_lap_s=g["lap_duration"].min(),
            median_clean_lap_s=c["lap_duration"].median() if len(c) else None,
            clean_lap_std_s=c["lap_duration"].std() if len(c) >= 5 else None,
            long_run_pace_s=long.median() if len(long) else None,
            long_run_laps=int(len(long)),
            deg_slope_s_per_lap=float(np.average(slopes, weights=weights)) if slopes else None,
            n_stints=int(n_stints.get(drv, 0)) or None,
            used_wet_tyres=int(wet.get(drv, 0)),
        ))
    return pd.DataFrame(out)


def weather(session_key: int) -> dict | None:
    w = pd.DataFrame(_get("weather", session_key))
    if w.empty:
        return None
    return dict(air_temp_c=w["air_temperature"].mean(), track_temp_c=w["track_temperature"].mean(),
                humidity_pct=w["humidity"].mean(), wind_speed_ms=w["wind_speed"].mean(),
                rainfall_fraction=float((w["rainfall"].fillna(0) > 0).mean()), n_samples=len(w))


def race_events(session_key: int) -> dict | None:
    rc = _get("race_control", session_key)
    if not rc:
        return None
    sc = vsc = red = 0
    for m in rc:
        msg = (m.get("message") or "").upper()
        if m.get("category") == "SafetyCar" and "DEPLOYED" in msg:
            if "VIRTUAL" in msg:
                vsc += 1
            else:
                sc += 1
        if (m.get("flag") or "").upper() == "RED":
            red += 1
    return dict(safety_cars=sc, virtual_safety_cars=vsc, red_flags=red)


def now() -> str:
    return db.utcnow()
