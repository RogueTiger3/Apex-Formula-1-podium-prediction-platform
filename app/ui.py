"""Shared dashboard helpers: styling, cached data access and chart defaults."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apex import config, db, features, models  # noqa: E402

ACCENT = "#E4002B"
MUTED = "#8A93A6"
GRID = "#232A38"
SERIES = ["#E4002B", "#3B82F6", "#F5B700", "#10B981", "#A855F7", "#F97316", "#06B6D4", "#EC4899"]

pio.templates["apex"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6E8EE", size=13),
    colorway=SERIES,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="#141924", bordercolor=GRID),
))
pio.templates.default = "plotly_dark+apex"

CSS = """
<style>
[data-testid="stAppViewContainer"] .main .block-container {padding-top: 1.6rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.apex-kicker {color:#8A93A6; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:600;}
.apex-card {background:#141924; border:1px solid #232A38; border-radius:10px; padding:16px 18px; height:100%;}
.apex-card h4 {margin:0 0 4px 0; font-size:1.05rem;}
.apex-big {font-size:1.9rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.1;}
.apex-sub {color:#8A93A6; font-size:.85rem;}
.apex-pos {display:inline-block; min-width:2.1rem; font-weight:700; color:#E4002B;}
.apex-pill {display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; border:1px solid #232A38; color:#C9CFDB;}
.apex-pill.est {border-color:#10B981; color:#10B981;}
.apex-pill.hyp {border-color:#F5B700; color:#F5B700;}
.apex-bar {height:4px; border-radius:2px; margin-top:8px;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
"""


def setup(title: str, kicker: str = "Apex · F1 predictive analytics"):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="apex-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if not config.DB_PATH.exists() or not db.scalar("SELECT COUNT(*) FROM sqlite_master WHERE name='races'"):
        st.warning("No database found yet. Open **Data Management** and press *Run update now* to download the data.")
        st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return db.query(sql, params)


@st.cache_resource(ttl=300, show_spinner=False)
def active_models():
    return models.load_active()


@st.cache_data(ttl=300, show_spinner="Building features…")
def feature_table(target_race: str | None = None) -> pd.DataFrame:
    return features.build(target_race)


def clear_caches():
    q.clear()
    feature_table.clear()
    active_models.clear()


@st.cache_data(ttl=300)
def names() -> dict[str, str]:
    d = db.query("SELECT driver_id, given_name, family_name FROM drivers")
    return dict(zip(d["driver_id"], d["given_name"] + " " + d["family_name"]))


@st.cache_data(ttl=300)
def codes() -> dict[str, str]:
    d = db.query("SELECT driver_id, code, family_name FROM drivers")
    return {r.driver_id: (r.code or r.family_name[:3].upper()) for r in d.itertuples()}


@st.cache_data(ttl=300)
def team_names() -> dict[str, str]:
    d = db.query("SELECT constructor_id, name FROM constructors")
    return dict(zip(d["constructor_id"], d["name"]))


@st.cache_data(ttl=300)
def team_colours() -> dict[str, str]:
    d = db.query("SELECT constructor_id, colour FROM team_colours ORDER BY season")
    return dict(zip(d["constructor_id"], d["colour"]))


def colour(constructor_id: str, i: int = 0) -> str:
    return team_colours().get(constructor_id, SERIES[i % len(SERIES)])


@st.cache_data(ttl=300)
def races() -> pd.DataFrame:
    r = db.query("SELECT r.*, c.name AS circuit_name, c.locality, c.country FROM races r"
                 " LEFT JOIN circuits c ON c.circuit_id=r.circuit_id ORDER BY r.race_id")
    r["start"] = pd.to_datetime(r["race_start_utc"], utc=True)
    return r


def race_label(row) -> str:
    return f"{row['season']} R{int(row['round'])} · {row['race_name']}"


def pct(x, digits=0) -> str:
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def fmt_ts(ts: str | None) -> str:
    if not ts:
        return "never"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.strftime("%d %b %Y, %H:%M UTC")


def ago(ts: str | None) -> str:
    if not ts:
        return "never"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    secs = (datetime.now(timezone.utc) - t.to_pydatetime()).total_seconds()
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} days ago"


def card(title: str, big: str, sub: str = "", bar: str | None = None):
    bar_html = f'<div class="apex-bar" style="background:{bar}"></div>' if bar else ""
    st.markdown(f'<div class="apex-card"><div class="apex-sub">{title}</div><div class="apex-big">{big}</div>'
                f'<div class="apex-sub">{sub}</div>{bar_html}</div>', unsafe_allow_html=True)


def latest_run(race_id: str, stage: int | None = None) -> pd.Series | None:
    sql = "SELECT * FROM prediction_runs WHERE race_id=?"
    params: tuple = (race_id,)
    if stage is not None:
        sql += " AND stage=?"
        params += (stage,)
    runs = q(sql + " ORDER BY stage DESC, created_at DESC", params)
    return runs.iloc[0] if len(runs) else None


def run_predictions(run_id: str) -> pd.DataFrame:
    p = q("SELECT * FROM predictions WHERE run_id=? ORDER BY rank", (run_id,))
    p["name"] = p["driver_id"].map(names())
    p["code"] = p["driver_id"].map(codes())
    p["team"] = p["constructor_id"].map(team_names())
    return p


def explanation(row) -> dict:
    try:
        return json.loads(row["explanation"])
    except (TypeError, ValueError):
        return {}


PROB_LABELS = {"p_win": "Win", "p_top2": "Top 2", "p_p2": "P2", "p_p3": "P3", "p_podium": "Podium",
               "p_top5": "Top 5", "p_top10": "Top 10", "p_dnf": "DNF"}


def as_pct(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Copy of df with probability columns converted to 0-100 for display."""
    out = df.copy()
    for c in cols or [c for c in out.columns if c in PROB_LABELS or c in ("Probability", "delta")]:
        out[c] = out[c] * 100
    return out


def prob_columns(bars=("p_win", "p_podium")) -> dict:
    cfg = {}
    for c, label in PROB_LABELS.items():
        if c in bars:
            cfg[c] = st.column_config.ProgressColumn(label, format="%.1f%%", min_value=0.0, max_value=100.0)
        else:
            cfg[c] = st.column_config.NumberColumn(label, format="%.1f%%")
    return cfg


def combo_column(max_pct: float):
    return st.column_config.ProgressColumn("Probability", format="%.2f%%", min_value=0.0, max_value=max(max_pct, 0.01))
