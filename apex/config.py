"""Central configuration. Paths and constants live here so every module agrees."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("APEX_DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("APEX_CACHE_DIR", DATA_DIR / "cache"))
MODEL_DIR = Path(os.environ.get("APEX_MODEL_DIR", ROOT / "models"))
DB_PATH = Path(os.environ.get("APEX_DB_PATH", DATA_DIR / "apex.db"))

# First season stored. Earlier seasons add little for the hybrid-era cars and cost API calls.
FIRST_SEASON = int(os.environ.get("APEX_FIRST_SEASON", 2014))
# First season used as a backtest test season (needs a few seasons of history before it).
FIRST_TEST_SEASON = 2018
# Pit stop data is fetched one request per race, so we limit it to recent seasons.
FIRST_PITSTOP_SEASON = 2018
# OpenF1 has lap, stint, weather and race-control data from 2023 onwards.
FIRST_OPENF1_SEASON = 2023

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE = "https://api.openf1.org/v1"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# A race result is treated as provisional until this many hours after the race start.
# Stewards' decisions and appeals normally settle well within this window.
RESULT_CONFIRMATION_HOURS = 72

STAGES = {
    1: "Pre-weekend",
    2: "Post-practice",
    3: "Post-qualifying",
    4: "Pre-race",
}

TARGETS = {"win": 1, "top2": 2, "podium": 3, "top5": 5, "top10": 10}

for _d in (DATA_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
