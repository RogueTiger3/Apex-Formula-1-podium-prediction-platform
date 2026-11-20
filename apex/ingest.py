"""Data ingestion: brings the SQLite database up to date from every source.

Safe to run repeatedly. Seasons that are complete in the database are skipped,
raw responses are cached on disk, and every write is an upsert keyed on natural
keys, so re-running never duplicates records.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, db
from .sources import jolpica, openf1, weather
from .sources.http import FetchError

log = logging.getLogger("apex.ingest")

# Forecasts are fetched for races starting within this many days. Further out,
# Open-Meteo forecasts for a specific two-hour window add little skill.
FORECAST_DAYS = 4


@dataclass
class IngestSummary:
    newly_final_races: list[str] = field(default_factory=list)
    provisional_races: list[str] = field(default_factory=list)
    sessions_processed: int = 0
    weather_rows: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def has_new_final_data(self) -> bool:
        return bool(self.newly_final_races)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def _season_complete(conn, season: int) -> bool:
    row = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN result_status='final' THEN 1 ELSE 0 END) FROM races WHERE season=?",
        (season,)).fetchone()
    return bool(row[0]) and row[0] == row[1]


def _store_people(conn, drivers, cons):
    db.upsert(conn, "drivers", drivers)
    db.upsert(conn, "constructors", cons)


def sync_jolpica(summary: IngestSummary, seasons: list[int], current: int) -> None:
    now = _now()
    for season in seasons:
        with db.session() as conn:
            if season < current and _season_complete(conn, season):
                continue
            before = dict(conn.execute(
                "SELECT race_id, result_status FROM races WHERE season=?", (season,)).fetchall())
        past = season < current
        max_age = None if past else 0
        try:
            races, circuits = jolpica.schedule(season, max_age=None if past else 6 * 3600)
            res, d1, c1 = jolpica.results(season, max_age)
            quali, d2, c2 = jolpica.qualifying(season, max_age)
            spr = jolpica.sprints(season, max_age)
        except FetchError as exc:
            summary.errors.append(f"Jolpica {season}: {exc}")
            log.warning("Jolpica %s failed: %s", season, exc)
            continue
        have_results = {r["race_id"] for r in res}
        for r in races:
            start = _parse(r["race_start_utc"])
            if r["race_id"] in have_results:
                confirmed = start and now > start + timedelta(hours=config.RESULT_CONFIRMATION_HOURS)
                r["result_status"] = "final" if confirmed else "provisional"
            else:
                r["result_status"] = "scheduled"
            if r["result_status"] == "final" and before.get(r["race_id"]) != "final":
                summary.newly_final_races.append(r["race_id"])
            if r["result_status"] == "provisional":
                summary.provisional_races.append(r["race_id"])
        with db.session() as conn:
            db.upsert(conn, "circuits", circuits)
            db.upsert(conn, "races", races)
            _store_people(conn, d1 + d2, c1 + c2)
            # Replace a season's rows wholesale so corrected results (penalties, DSQs) win.
            for table, rows in (("results", res), ("qualifying", quali), ("sprint_results", spr)):
                ids = sorted({r["race_id"] for r in rows})
                if ids:
                    conn.execute(f"DELETE FROM {table} WHERE race_id IN ({','.join('?'*len(ids))})", ids)
                    db.upsert(conn, table, rows)
        log.info("Jolpica %s: %d races, %d results, %d quali rows", season, len(races), len(res), len(quali))

    # Pit stops: one request per race; final races are cached forever.
    with db.session() as conn:
        todo = conn.execute(
            "SELECT season, round, race_id, result_status FROM races WHERE season>=? AND result_status!='scheduled'"
            " AND race_id NOT IN (SELECT DISTINCT race_id FROM pit_stops) ORDER BY race_id",
            (config.FIRST_PITSTOP_SEASON,)).fetchall()
    for season, rnd, rid, status in todo:
        try:
            rows = jolpica.pit_stops(season, rnd, max_age=None if status == "final" else 0)
        except FetchError as exc:
            summary.errors.append(f"Jolpica pit stops {rid}: {exc}")
            continue
        with db.session() as conn:
            db.upsert(conn, "pit_stops", rows)

    try:
        st = jolpica.current_standings(max_age=0)
        with db.session() as conn:
            conn.execute("DELETE FROM standings")
            db.upsert(conn, "standings", st)
    except FetchError as exc:
        summary.errors.append(f"Jolpica standings: {exc}")


def _number_map(conn, race_id: str, season: int) -> dict[int, str]:
    """Car number -> driver_id for a race weekend, falling back to the latest earlier race."""
    for table in ("results", "qualifying"):
        rows = conn.execute(f"SELECT car_number, driver_id FROM {table} WHERE race_id=?", (race_id,)).fetchall()
        if rows:
            return {int(n): d for n, d in rows if n is not None}
    row = conn.execute(
        "SELECT r.race_id FROM races r JOIN results x ON x.race_id=r.race_id WHERE r.season=? AND r.race_id<?"
        " ORDER BY r.race_id DESC LIMIT 1", (season, race_id)).fetchone()
    if row:
        rows = conn.execute("SELECT car_number, driver_id FROM results WHERE race_id=?", (row[0],)).fetchall()
        return {int(n): d for n, d in rows if n is not None}
    return {}


def _match_meetings(sessions: list[dict], races: pd.DataFrame) -> dict[int, str]:
    """OpenF1 meeting_key -> Jolpica race_id, matched on the race (or last session) date."""
    out = {}
    by_meeting: dict[int, list[dict]] = {}
    for s in sessions:
        by_meeting.setdefault(s["meeting_key"], []).append(s)
    races = races.copy()
    races["date"] = pd.to_datetime(races["race_start_utc"], utc=True).dt.date
    for mk, ss in by_meeting.items():
        if any(s.get("session_type") == "Practice" and "Day" in s.get("session_name", "") for s in ss):
            continue  # pre-season testing
        last = max(pd.Timestamp(s["date_start"]) for s in ss)
        diffs = (pd.to_datetime(races["date"]) - pd.Timestamp(last.date())).abs()
        if len(diffs) and diffs.min() <= pd.Timedelta(days=1):
            out[mk] = races.loc[diffs.idxmin(), "race_id"]
    return out


def sync_openf1(summary: IngestSummary, seasons: list[int], current: int) -> None:
    now = _now()
    for season in [s for s in seasons if s >= config.FIRST_OPENF1_SEASON]:
        try:
            sess = openf1.sessions(season, max_age=None if season < current else 3 * 3600)
        except FetchError as exc:
            summary.errors.append(f"OpenF1 sessions {season}: {exc}")
            continue
        races = db.query("SELECT race_id, race_start_utc FROM races WHERE season=?", (season,))
        meeting_to_race = _match_meetings(sess, races)
        with db.session() as conn:
            processed = {r[0] for r in conn.execute("SELECT session_key FROM openf1_sessions WHERE processed=1")}
        for s in sess:
            rid = meeting_to_race.get(s["meeting_key"])
            name = openf1.SESSION_ALIASES.get(s["session_name"])
            if not rid or not name or s.get("is_cancelled"):
                continue
            end = pd.Timestamp(s["date_end"]).to_pydatetime()
            if s["session_key"] in processed or now < end + timedelta(minutes=45):
                continue
            try:
                stats = openf1.lap_stats(s["session_key"], is_race=name in ("R", "S"))
                wx = openf1.weather(s["session_key"])
                ev = openf1.race_events(s["session_key"]) if name == "R" else None
                # Driver list (acronyms, team colours): only needed for the race, or when
                # car numbers cannot be mapped from Jolpica yet. Saves a fifth of the requests.
                with db.session() as conn:
                    have_map = bool(_number_map(conn, rid, season))
                info = openf1.driver_info(s["session_key"]) if (name == "R" or not have_map) else []
            except FetchError as exc:
                summary.errors.append(f"OpenF1 {rid} {name}: {exc}")
                continue
            ts = db.utcnow()
            with db.session() as conn:
                nmap = _number_map(conn, rid, season)
                code_map = dict(conn.execute("SELECT code, driver_id FROM drivers WHERE code IS NOT NULL").fetchall())
                acr = {int(d["driver_number"]): d.get("name_acronym") for d in info}
                rows = []
                for _, r in stats.iterrows():
                    drv = nmap.get(int(r["car_number"])) or code_map.get(acr.get(int(r["car_number"])))
                    if not drv:
                        continue
                    rows.append(dict(race_id=rid, session_name=name, driver_id=drv,
                                     **{k: (None if pd.isna(v) else v) for k, v in r.items()},
                                     source="openf1", retrieved_at=ts))
                db.upsert(conn, "session_driver_stats", rows)
                if wx:
                    db.upsert(conn, "session_weather", [dict(race_id=rid, session_name=name, **wx,
                                                             source="openf1", retrieved_at=ts)])
                if ev:
                    db.upsert(conn, "race_events", [dict(race_id=rid, **ev, source="openf1", retrieved_at=ts)])
                # Team colours for the dashboard.
                cmap = dict(conn.execute("SELECT driver_id, constructor_id FROM results WHERE race_id=?"
                                         " UNION SELECT driver_id, constructor_id FROM qualifying WHERE race_id=?",
                                         (rid, rid)).fetchall())
                colours = {}
                for d in info:
                    drv = nmap.get(int(d["driver_number"]))
                    if drv in cmap and d.get("team_colour"):
                        colours[cmap[drv]] = dict(constructor_id=cmap[drv], colour="#" + d["team_colour"],
                                                  season=season, source="openf1", retrieved_at=ts)
                db.upsert(conn, "team_colours", list(colours.values()))
                db.upsert(conn, "openf1_sessions", [dict(
                    session_key=s["session_key"], meeting_key=s["meeting_key"], race_id=rid,
                    session_name=name, session_type=s.get("session_type"), date_start=s["date_start"],
                    date_end=s["date_end"], year=season, location=s.get("location"),
                    processed=int(bool(rows)), source="openf1", retrieved_at=ts)])
            summary.sessions_processed += 1


def sync_weather(summary: IngestSummary, max_archive_requests: int = 400) -> None:
    now = _now()
    races = db.query(
        "SELECT r.race_id, r.race_start_utc, r.result_status, c.lat, c.lon FROM races r"
        " JOIN circuits c ON c.circuit_id=r.circuit_id"
        " WHERE r.race_id NOT IN (SELECT race_id FROM race_weather WHERE kind='observed')"
        " ORDER BY r.race_id DESC")
    done = 0
    ts = db.utcnow()
    for _, r in races.iterrows():
        start = _parse(r["race_start_utc"])
        if start is None:
            continue
        days_ahead = (start - now).total_seconds() / 86400
        if r["result_status"] == "scheduled" and -0.1 <= days_ahead <= FORECAST_DAYS:
            wx, err = weather.race_window(r["lat"], r["lon"], r["race_start_utc"], forecast=True)
            kind = "forecast"
        elif start < now - timedelta(days=6) and done < max_archive_requests:
            # The ERA5 archive lags real time by about five days.
            wx, err = weather.race_window(r["lat"], r["lon"], r["race_start_utc"], forecast=False)
            kind = "observed"
            done += 1
        else:
            continue
        if wx is None:
            if err:
                summary.errors.append(f"Open-Meteo {r['race_id']}: {err[:120]}")
                if "limit" in err.lower():
                    break  # daily quota hit: stop for today
            continue
        with db.session() as conn:
            db.upsert(conn, "race_weather", [dict(race_id=r["race_id"], kind=kind, **wx,
                                                  source="open-meteo", retrieved_at=ts)])
        summary.weather_rows += 1


def current_season() -> int:
    try:
        races, _ = jolpica.schedule("current", max_age=6 * 3600)  # type: ignore[arg-type]
        if races:
            return int(races[0]["season"])
    except FetchError:
        pass
    return _now().year


def sync_all(include_weather: bool = True) -> IngestSummary:
    db.init_db()
    summary = IngestSummary()
    current = current_season()
    seasons = list(range(config.FIRST_SEASON, current + 1))
    sync_jolpica(summary, seasons, current)
    sync_openf1(summary, seasons, current)
    if include_weather:
        sync_weather(summary)
    return summary
