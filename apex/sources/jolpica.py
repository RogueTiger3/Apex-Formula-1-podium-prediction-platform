"""Jolpica F1 API (the community successor to the Ergast API). No key required.

Docs: https://github.com/jolpica/jolpica-f1
"""
from __future__ import annotations

from .. import config, db
from .http import get_json

SOURCE = "jolpica"


def _paged(path: str, table_key: str, max_age: float | None) -> list[dict]:
    """Fetch every page of a Jolpica endpoint and merge races split across pages."""
    races: dict[tuple, dict] = {}
    offset, limit = 0, 100
    while True:
        data = get_json(f"{config.JOLPICA_BASE}/{path}.json",
                        {"limit": limit, "offset": offset}, max_age=max_age)
        mr = data["MRData"]
        for race in mr["RaceTable"]["Races"]:
            key = (race["season"], race["round"])
            if key not in races:
                races[key] = {**race, table_key: []}
            races[key][table_key].extend(race.get(table_key, []))
        total = int(mr["total"])
        offset += limit
        if offset >= total:
            break
    return list(races.values())


def lap_str_to_s(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if ":" in value:
            m, s = value.split(":")
            return int(m) * 60 + float(s)
        return float(value)
    except ValueError:
        return None


def _iso(date: str | None, time: str | None) -> str | None:
    if not date:
        return None
    return f"{date}T{time or '12:00:00Z'}".replace("Z", "+00:00")


def race_id(season, rnd) -> str:
    return f"{int(season)}_{int(rnd):02d}"


def schedule(season: int, max_age: float | None) -> tuple[list[dict], list[dict]]:
    data = get_json(f"{config.JOLPICA_BASE}/{season}.json", {"limit": 100}, max_age=max_age)
    now = db.utcnow()
    races, circuits = [], []
    for r in data["MRData"]["RaceTable"]["Races"]:
        c = r["Circuit"]
        circuits.append(dict(circuit_id=c["circuitId"], name=c["circuitName"],
                             locality=c["Location"].get("locality"),
                             country=c["Location"].get("country"),
                             lat=float(c["Location"]["lat"]), lon=float(c["Location"]["long"]),
                             source=SOURCE, retrieved_at=now))
        races.append(dict(
            race_id=race_id(r["season"], r["round"]), season=int(r["season"]),
            round=int(r["round"]), race_name=r["raceName"], circuit_id=c["circuitId"],
            race_start_utc=_iso(r["date"], r.get("time")),
            quali_start_utc=_iso(r.get("Qualifying", {}).get("date"), r.get("Qualifying", {}).get("time")),
            sprint_start_utc=_iso(r.get("Sprint", {}).get("date"), r.get("Sprint", {}).get("time")),
            fp1_start_utc=_iso(r.get("FirstPractice", {}).get("date"), r.get("FirstPractice", {}).get("time")),
            has_sprint=int("Sprint" in r), source=SOURCE, retrieved_at=now))
    return races, circuits


def _people(entry: dict, now: str) -> tuple[dict, dict]:
    d, c = entry["Driver"], entry["Constructor"]
    driver = dict(driver_id=d["driverId"], code=d.get("code"), permanent_number=d.get("permanentNumber"),
                  given_name=d.get("givenName"), family_name=d.get("familyName"),
                  nationality=d.get("nationality"), date_of_birth=d.get("dateOfBirth"),
                  source=SOURCE, retrieved_at=now)
    cons = dict(constructor_id=c["constructorId"], name=c["name"], nationality=c.get("nationality"),
                source=SOURCE, retrieved_at=now)
    return driver, cons


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def results(season: int, max_age: float | None):
    now = db.utcnow()
    rows, drivers, cons = [], {}, {}
    for race in _paged(f"{season}/results", "Results", max_age):
        rid = race_id(race["season"], race["round"])
        for e in race["Results"]:
            d, c = _people(e, now)
            drivers[d["driver_id"]], cons[c["constructor_id"]] = d, c
            pos_text = e.get("positionText", "")
            rows.append(dict(
                race_id=rid, driver_id=d["driver_id"], constructor_id=c["constructor_id"],
                car_number=_int(e.get("number")), grid=_int(e.get("grid")),
                # Classified position only; R/D/E/W/F/N mean not classified.
                position=_int(e["position"]) if pos_text.isdigit() else None,
                position_text=pos_text, points=float(e.get("points", 0)), laps=_int(e.get("laps")),
                status=e.get("status"), time_ms=float(e["Time"]["millis"]) if e.get("Time", {}).get("millis") else None,
                fastest_lap_rank=_int(e.get("FastestLap", {}).get("rank")),
                source=SOURCE, retrieved_at=now))
    return rows, list(drivers.values()), list(cons.values())


def qualifying(season: int, max_age: float | None):
    now = db.utcnow()
    rows, drivers, cons = [], {}, {}
    for race in _paged(f"{season}/qualifying", "QualifyingResults", max_age):
        rid = race_id(race["season"], race["round"])
        for e in race["QualifyingResults"]:
            d, c = _people(e, now)
            drivers[d["driver_id"]], cons[c["constructor_id"]] = d, c
            rows.append(dict(race_id=rid, driver_id=d["driver_id"], constructor_id=c["constructor_id"],
                             car_number=_int(e.get("number")), position=_int(e.get("position")),
                             q1_s=lap_str_to_s(e.get("Q1")), q2_s=lap_str_to_s(e.get("Q2")),
                             q3_s=lap_str_to_s(e.get("Q3")), source=SOURCE, retrieved_at=now))
    return rows, list(drivers.values()), list(cons.values())


def sprints(season: int, max_age: float | None):
    now = db.utcnow()
    rows = []
    for race in _paged(f"{season}/sprint", "SprintResults", max_age):
        rid = race_id(race["season"], race["round"])
        for e in race["SprintResults"]:
            pos_text = e.get("positionText", "")
            rows.append(dict(race_id=rid, driver_id=e["Driver"]["driverId"],
                             constructor_id=e["Constructor"]["constructorId"], grid=_int(e.get("grid")),
                             position=_int(e["position"]) if pos_text.isdigit() else None,
                             position_text=pos_text, points=float(e.get("points", 0)),
                             status=e.get("status"), source=SOURCE, retrieved_at=now))
    return rows


def pit_stops(season: int, rnd: int, max_age: float | None):
    now = db.utcnow()
    rows = []
    for race in _paged(f"{season}/{rnd}/pitstops", "PitStops", max_age):
        rid = race_id(race["season"], race["round"])
        for e in race["PitStops"]:
            rows.append(dict(race_id=rid, driver_id=e["driverId"], stop=_int(e["stop"]),
                             lap=_int(e["lap"]), duration_s=lap_str_to_s(e.get("duration")),
                             source=SOURCE, retrieved_at=now))
    return rows


def current_standings(max_age: float | None = 3600):
    now = db.utcnow()
    rows = []
    for kind, key, list_key in (("driver", "driverStandings", "DriverStandings"),
                                ("constructor", "constructorStandings", "ConstructorStandings")):
        data = get_json(f"{config.JOLPICA_BASE}/current/{key}.json", {"limit": 100}, max_age=max_age)
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            continue
        sl = lists[0]
        for s in sl[list_key]:
            if kind == "driver":
                ent = s["Driver"]
                eid, name = ent["driverId"], f'{ent.get("givenName")} {ent.get("familyName")}'
            else:
                ent = s["Constructor"]
                eid, name = ent["constructorId"], ent["name"]
            rows.append(dict(kind=kind, season=int(sl["season"]), after_round=int(sl["round"]),
                             position=_int(s.get("position")), entity_id=eid, entity_name=name,
                             points=float(s["points"]), wins=_int(s["wins"]),
                             source=SOURCE, retrieved_at=now))
    return rows
