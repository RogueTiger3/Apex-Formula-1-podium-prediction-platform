import json

import pandas as pd
import streamlit as st

from app import ui
from apex import config, db

ui.setup("Data management")

cnt = lambda sql: int(ui.q(sql)["n"].iloc[0])
reg = ui.q("SELECT * FROM model_registry WHERE is_active=1 ORDER BY stage")
logs = ui.q("SELECT * FROM update_log ORDER BY id DESC LIMIT 50")
ok = logs[logs["status"].isin(["success", "partial"])]

k = st.columns(4)
with k[0]:
    ui.card("Last successful update", ui.ago(ok["finished_at"].iloc[0]) if len(ok) else "never",
            ui.fmt_ts(ok["finished_at"].iloc[0]) if len(ok) else "")
n_races = cnt("SELECT COUNT(DISTINCT race_id) AS n FROM results")
n_prov = cnt("SELECT COUNT(*) AS n FROM races WHERE result_status='provisional'")
n_sched = cnt("SELECT COUNT(*) AS n FROM races WHERE result_status='scheduled'")
n_drv = cnt("SELECT COUNT(*) AS n FROM drivers")
n_con = cnt("SELECT COUNT(*) AS n FROM constructors")
with k[1]:
    ui.card("Races stored", f"{n_races:,}", f"{n_prov} provisional · {n_sched} scheduled")
with k[2]:
    ui.card("Drivers · Constructors", f"{n_drv} · {n_con}",
            f"{config.FIRST_SEASON} onwards")
with k[3]:
    ui.card("Current model", reg["model_version"].iloc[0].split("-")[0] if len(reg) else "none",
            f"{reg['model_version'].iloc[0]} · trained {ui.fmt_ts(reg['trained_at'].iloc[0])}" if len(reg) else "")

st.subheader("Refresh data")
st.markdown("Updates normally run automatically through GitHub Actions every six hours. This button runs the same pipeline here: "
            "it downloads anything new, retrains only if a race result has been confirmed since the last training run, "
            "and refreshes predictions for the next race.")
c1, c2 = st.columns([1, 3])
with c1:
    force = st.checkbox("Force a full retrain", help="Re-runs the backtest for every stage. Takes a few minutes.")
    go = st.button("Run update now", type="primary")
if go:
    from apex import pipeline
    with st.status("Running update…", expanded=True) as box:
        st.write("Checking Jolpica, OpenF1 and Open-Meteo for new data…")
        summary = pipeline.update(trigger="dashboard", force_retrain=force)
        ui.clear_caches()
        st.write(f"Newly confirmed races: {', '.join(summary.get('newly_final', [])) or 'none'}")
        st.write(f"Predictions: {len(summary.get('predictions', []))} stage(s) refreshed")
        if summary.get("errors"):
            st.write("Some requests failed (details in the log below); existing data was kept.")
        box.update(label=f"Update finished: {summary['status']}", state="complete" if summary["status"] != "failed" else "error")

st.subheader("Data sources")
src = ui.q("""SELECT source, COUNT(*) AS requests, SUM(ok) AS ok, MAX(CASE WHEN ok=1 THEN ts END) AS last_ok,
              MAX(CASE WHEN ok=0 THEN ts END) AS last_fail FROM fetch_log GROUP BY source""").set_index("source")
sources = [
    ("Jolpica F1", "Results, qualifying, sprints, calendars, circuits, pit stops, standings", f"{config.FIRST_SEASON}+ (pit stops {config.FIRST_PITSTOP_SEASON}+)", "No key; ~500 requests/hour"),
    ("OpenF1", "Lap times, stints/tyres, trackside weather, race control (safety cars), team colours", f"{config.FIRST_OPENF1_SEASON}+", "No key for historical data; live session data is paid-only, so Apex reads sessions once finished"),
    ("Open-Meteo archive", "Race-window rain, temperature, wind for past races", "1940+", "No key; daily quota per IP"),
    ("Open-Meteo forecast", "Race-window forecast for races in the next 15 days", "Upcoming", "No key; daily quota per IP"),
    ("FastF1", "Not used", "–", "Its timing source (F1 live timing) now requires an F1 account login, so it cannot run keyless; OpenF1 supplies the same lap and tyre data"),
]
rows = []
for name, what, cover, access in sources:
    s = src.loc[name] if name in src.index else None
    rows.append(dict(Source=name, Provides=what, Coverage=cover, Access=access,
                     Requests=int(s["requests"]) if s is not None else 0,
                     **{"Last success": ui.fmt_ts(s["last_ok"]) if s is not None and s["last_ok"] else "–",
                        "Last failure": ui.fmt_ts(s["last_fail"]) if s is not None and s["last_fail"] else "–"}))
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Every stored row keeps its source and retrieval time (columns `source` and `retrieved_at`). Raw API responses are cached on disk so nothing is downloaded twice.")

cov = ui.q("""SELECT r.season, COUNT(DISTINCT r.race_id) AS races,
              COUNT(DISTINCT q.race_id) AS with_qualifying, COUNT(DISTINCT p.race_id) AS with_pit_stops,
              COUNT(DISTINCT s.race_id) AS with_lap_data, COUNT(DISTINCT w.race_id) AS with_weather
              FROM races r LEFT JOIN qualifying q ON q.race_id=r.race_id LEFT JOIN pit_stops p ON p.race_id=r.race_id
              LEFT JOIN session_driver_stats s ON s.race_id=r.race_id
              LEFT JOIN race_weather w ON w.race_id=r.race_id AND w.kind='observed'
              WHERE r.result_status!='scheduled' GROUP BY r.season ORDER BY r.season DESC""")
st.markdown("**Coverage by season** (completed races)")
st.dataframe(cov, hide_index=True, width="stretch", column_config={"season": st.column_config.NumberColumn("Season", format="%d")})

st.subheader("Update log")
if logs.empty:
    st.info("No updates have run yet.")
else:
    l = logs.copy()
    def short(s):
        if not s:
            return ""
        d = json.loads(s)
        parts = [f"confirmed: {', '.join(d['newly_final'])}" if d.get("newly_final") else "no newly confirmed results",
                 f"{d.get('sessions', 0)} OpenF1 sessions", f"{len(d.get('predictions', []))} predictions"]
        if d.get("errors"):
            parts.append(f"{len(d['errors'])} errors")
        return "; ".join(parts)
    l["summary_short"] = l["summary"].map(short)
    st.dataframe(l[["started_at", "finished_at", "trigger", "status", "new_races", "retrained", "summary_short"]], hide_index=True,
                 width="stretch", column_config={"summary_short": "Summary", "retrained": st.column_config.CheckboxColumn("Retrained")})

st.subheader("Failed data requests")
fails = ui.q("SELECT ts, source, http_status, url, message FROM fetch_log WHERE ok=0 ORDER BY id DESC LIMIT 100")
if fails.empty:
    st.success("No failed requests in the log.")
else:
    st.dataframe(fails, hide_index=True, width="stretch")
    st.caption("Failures are retried automatically on the next run. Where a cached copy existed it was used instead.")

st.subheader("Model registry")
st.dataframe(ui.q("SELECT model_version, stage, algorithm, trained_at, data_cutoff_race, n_train, n_features, is_active FROM model_registry ORDER BY trained_at DESC, stage"),
             hide_index=True, width="stretch", column_config={"is_active": st.column_config.CheckboxColumn("Active")})
