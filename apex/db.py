"""SQLite storage layer.

All SQL here is deliberately plain ANSI SQL (no SQLite-only features apart from
`INSERT OR REPLACE`, wrapped in `upsert`) so the schema can move to PostgreSQL by
swapping `connect()` and the upsert statement.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS circuits (
    circuit_id TEXT PRIMARY KEY, name TEXT, locality TEXT, country TEXT,
    lat REAL, lon REAL, source TEXT, retrieved_at TEXT);

CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY, season INTEGER, round INTEGER, race_name TEXT,
    circuit_id TEXT, race_start_utc TEXT, quali_start_utc TEXT, sprint_start_utc TEXT,
    fp1_start_utc TEXT, has_sprint INTEGER, result_status TEXT,
    source TEXT, retrieved_at TEXT);

CREATE TABLE IF NOT EXISTS drivers (
    driver_id TEXT PRIMARY KEY, code TEXT, permanent_number TEXT, given_name TEXT,
    family_name TEXT, nationality TEXT, date_of_birth TEXT, source TEXT, retrieved_at TEXT);

CREATE TABLE IF NOT EXISTS constructors (
    constructor_id TEXT PRIMARY KEY, name TEXT, nationality TEXT, source TEXT, retrieved_at TEXT);

CREATE TABLE IF NOT EXISTS team_colours (
    constructor_id TEXT PRIMARY KEY, colour TEXT, season INTEGER, source TEXT, retrieved_at TEXT);

CREATE TABLE IF NOT EXISTS results (
    race_id TEXT, driver_id TEXT, constructor_id TEXT, car_number INTEGER, grid INTEGER,
    position INTEGER, position_text TEXT, points REAL, laps INTEGER, status TEXT,
    time_ms REAL, fastest_lap_rank INTEGER, source TEXT, retrieved_at TEXT,
    PRIMARY KEY (race_id, driver_id));

CREATE TABLE IF NOT EXISTS qualifying (
    race_id TEXT, driver_id TEXT, constructor_id TEXT, car_number INTEGER, position INTEGER,
    q1_s REAL, q2_s REAL, q3_s REAL, source TEXT, retrieved_at TEXT,
    PRIMARY KEY (race_id, driver_id));

CREATE TABLE IF NOT EXISTS sprint_results (
    race_id TEXT, driver_id TEXT, constructor_id TEXT, grid INTEGER, position INTEGER,
    position_text TEXT, points REAL, status TEXT, source TEXT, retrieved_at TEXT,
    PRIMARY KEY (race_id, driver_id));

CREATE TABLE IF NOT EXISTS pit_stops (
    race_id TEXT, driver_id TEXT, stop INTEGER, lap INTEGER, duration_s REAL,
    source TEXT, retrieved_at TEXT, PRIMARY KEY (race_id, driver_id, stop));

CREATE TABLE IF NOT EXISTS openf1_sessions (
    session_key INTEGER PRIMARY KEY, meeting_key INTEGER, race_id TEXT, session_name TEXT,
    session_type TEXT, date_start TEXT, date_end TEXT, year INTEGER, location TEXT,
    processed INTEGER DEFAULT 0, source TEXT, retrieved_at TEXT);

CREATE TABLE IF NOT EXISTS session_driver_stats (
    race_id TEXT, session_name TEXT, driver_id TEXT, car_number INTEGER,
    n_laps INTEGER, best_lap_s REAL, median_clean_lap_s REAL, clean_lap_std_s REAL,
    long_run_pace_s REAL, long_run_laps INTEGER, deg_slope_s_per_lap REAL,
    n_stints INTEGER, used_wet_tyres INTEGER, source TEXT, retrieved_at TEXT,
    PRIMARY KEY (race_id, session_name, driver_id));

CREATE TABLE IF NOT EXISTS session_weather (
    race_id TEXT, session_name TEXT, air_temp_c REAL, track_temp_c REAL, humidity_pct REAL,
    wind_speed_ms REAL, rainfall_fraction REAL, n_samples INTEGER, source TEXT, retrieved_at TEXT,
    PRIMARY KEY (race_id, session_name));

CREATE TABLE IF NOT EXISTS race_events (
    race_id TEXT PRIMARY KEY, safety_cars INTEGER, virtual_safety_cars INTEGER,
    red_flags INTEGER, source TEXT, retrieved_at TEXT);

CREATE TABLE IF NOT EXISTS race_weather (
    race_id TEXT, kind TEXT, temp_c REAL, precip_mm REAL, precip_prob_pct REAL,
    wind_kmh REAL, humidity_pct REAL, source TEXT, retrieved_at TEXT,
    PRIMARY KEY (race_id, kind));

CREATE TABLE IF NOT EXISTS standings (
    kind TEXT, season INTEGER, after_round INTEGER, position INTEGER, entity_id TEXT,
    entity_name TEXT, points REAL, wins INTEGER, source TEXT, retrieved_at TEXT,
    PRIMARY KEY (kind, season, entity_id));

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT, url TEXT,
    http_status INTEGER, ok INTEGER, from_cache INTEGER, message TEXT);

CREATE TABLE IF NOT EXISTS update_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, finished_at TEXT, trigger TEXT,
    status TEXT, new_races INTEGER, retrained INTEGER, summary TEXT);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT, stage INTEGER, algorithm TEXT, trained_at TEXT,
    data_cutoff_race TEXT, n_train INTEGER, n_features INTEGER, metrics_json TEXT,
    params_json TEXT, path TEXT, is_active INTEGER, PRIMARY KEY (model_version, stage));

CREATE TABLE IF NOT EXISTS backtest_metrics (
    model_version TEXT, stage INTEGER, algorithm TEXT, season TEXT, n_races INTEGER,
    n_rows INTEGER, accuracy REAL, precision_ REAL, recall REAL, f1 REAL, brier REAL,
    log_loss REAL, roc_auc REAL, top3_hit_rate REAL,
    PRIMARY KEY (model_version, stage, algorithm, season));

CREATE TABLE IF NOT EXISTS feature_importance (
    model_version TEXT, stage INTEGER, feature TEXT, label TEXT, category TEXT,
    mean_abs_shap REAL, PRIMARY KEY (model_version, stage, feature));

CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id TEXT PRIMARY KEY, created_at TEXT, race_id TEXT, stage INTEGER,
    model_version TEXT, algorithm TEXT, predicted_podium TEXT, inputs_json TEXT,
    trigger TEXT);

CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT, race_id TEXT, driver_id TEXT, constructor_id TEXT, rank INTEGER,
    p_win REAL, p_top2 REAL, p_podium REAL, p_top5 REAL, p_top10 REAL,
    p_podium_model REAL, explanation TEXT, PRIMARY KEY (run_id, driver_id));
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or config.DB_PATH), timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path | str | None = None):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | str | None = None) -> None:
    with session(path) as conn:
        conn.executescript(SCHEMA)


def upsert(conn: sqlite3.Connection, table: str, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = (f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' for _ in cols)})")
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    return len(rows)


def insert(conn: sqlite3.Connection, table: str, rows: Iterable[dict]) -> int:
    """Plain insert: fails on duplicates. Used for append-only history tables."""
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    return len(rows)


def query(sql: str, params: tuple | dict = (), path: Path | str | None = None) -> pd.DataFrame:
    with session(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def scalar(sql: str, params: tuple = (), path: Path | str | None = None):
    with session(path) as conn:
        row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def to_json(obj) -> str:
    return json.dumps(obj, default=str, sort_keys=True)
