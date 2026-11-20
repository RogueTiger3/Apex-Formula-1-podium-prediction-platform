import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui
from apex import config
from apex.features import INCIDENT_STATUSES, NON_START

ui.setup("Constructor analytics")
names, tnames = ui.names(), ui.team_names()
res = ui.q("SELECT x.*, r.season, r.round, r.race_name, r.race_start_utc FROM results x JOIN races r ON r.race_id=x.race_id ORDER BY r.race_start_utc")
quali = ui.q("SELECT race_id, driver_id, position AS quali_pos FROM qualifying")
res = res.merge(quali, on=["race_id", "driver_id"], how="left")
status = res["status"].fillna("")
res["finished"] = status.eq("Finished") | status.str.startswith("+") | status.str.contains("Lap") | res["position"].notna()
res["mech_dnf"] = ~res["finished"] & ~status.isin(INCIDENT_STATUSES) & ~status.isin(NON_START)
latest = int(res["season"].max())
current = res[res["season"] == latest].groupby("constructor_id")["points"].sum().sort_values(ascending=False).index.tolist()
others = sorted(set(res["constructor_id"]) - set(current))

c1, c2 = st.columns([1.2, 1])
with c1:
    team = st.selectbox("Constructor", current + others, format_func=lambda c: tnames.get(c, c) + ("" if c in current else " (former entrant)"))
t = res[res["constructor_id"] == team]
seasons = sorted(t["season"].unique())
with c2:
    span = st.select_slider("Seasons", options=seasons, value=(max(seasons[0], seasons[-1] - 4), seasons[-1])) if len(seasons) > 1 else (seasons[0], seasons[0])
t = t[t["season"].between(*span)]
col = ui.colour(team)

per_race = t.groupby(["race_id", "season", "round", "race_name"]).agg(points=("points", "sum"), best=("position", "min"),
                                                                     avg_q=("quali_pos", "mean"), mech=("mech_dnf", "sum")).reset_index()
k = st.columns(5)
for c, (lab, v, s) in zip(k, [
    ("Race entries", f"{len(per_race)}", f"{t['driver_id'].nunique()} drivers"),
    ("Wins", f"{int((t['position'] == 1).sum())}", ""),
    ("Podiums", f"{int((t['position'] <= 3).sum())}", ""),
    ("Points per race", f"{per_race['points'].mean():.1f}", "both cars, incl. fastest-lap bonuses"),
    ("Mechanical DNF rate", ui.pct(t["mech_dnf"].mean(), 1), "per car start"),
]):
    with c:
        ui.card(lab, v, s, bar=col)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Performance trend", "Driver comparison", "Reliability", "Qualifying and race pace"])
with tab1:
    per_race["label"] = per_race["season"].astype(str) + " R" + per_race["round"].astype(str)
    per_race["roll"] = per_race["points"].rolling(5, min_periods=1).mean()
    fig = go.Figure()
    fig.add_bar(x=per_race["label"], y=per_race["points"], name="Points", marker_color=col, opacity=0.55,
                customdata=per_race["race_name"], hovertemplate="%{customdata}<br>%{y:.0f} pts<extra></extra>")
    fig.add_scatter(x=per_race["label"], y=per_race["roll"], name="5-race average", line=dict(color="#F5B700", width=2))
    fig.update_layout(height=380, title="Points per race (development trend)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    season = res[res["constructor_id"] == team].groupby("season").agg(Points=("points", "sum"), Wins=("position", lambda s: (s == 1).sum()),
                                                                     Podiums=("position", lambda s: (s <= 3).sum()), Races=("race_id", "nunique")).reset_index()
    st.dataframe(season, hide_index=True, width="stretch", column_config={"season": st.column_config.NumberColumn("Season", format="%d"),
                                                                         "Points": st.column_config.NumberColumn(format="%.0f")})
with tab2:
    g = t.groupby("driver_id").agg(Starts=("race_id", "count"), Points=("points", "sum"), Wins=("position", lambda s: (s == 1).sum()),
                                   Podiums=("position", lambda s: (s <= 3).sum()), AvgFinish=("position", "mean"),
                                   AvgQuali=("quali_pos", "mean"), DNF=("finished", lambda s: 1 - s.mean())).reset_index()
    g["Driver"] = g["driver_id"].map(names)
    st.dataframe(g[["Driver", "Starts", "Points", "Wins", "Podiums", "AvgFinish", "AvgQuali", "DNF"]].sort_values("Points", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"Points": st.column_config.NumberColumn(format="%.0f"), "AvgFinish": st.column_config.NumberColumn("Avg finish", format="%.1f"),
                                "AvgQuali": st.column_config.NumberColumn("Avg qualifying", format="%.1f"),
                                "DNF": st.column_config.NumberColumn("DNF rate", format="percent")})
    fig = go.Figure()
    for i, (dv, gg) in enumerate(t.groupby("driver_id")):
        gg = gg.sort_values("race_start_utc")
        fig.add_scatter(x=gg["season"].astype(str) + " R" + gg["round"].astype(str), y=gg["position"].rolling(5, min_periods=1).mean(),
                        mode="lines", name=names.get(dv, dv), line=dict(color=ui.SERIES[i % len(ui.SERIES)]))
    fig.update_layout(height=360, title="Average finish, rolling 5 races (classified finishes)", yaxis=dict(autorange="reversed"),
                      xaxis=dict(showticklabels=False))
    st.plotly_chart(fig, width="stretch")
with tab3:
    rel = res[res["constructor_id"] == team].groupby("season").agg(starts=("race_id", "count"), mech=("mech_dnf", "sum"),
                                                                    other=("finished", lambda s: (~s).sum())).reset_index()
    rel["Mechanical DNF rate"] = rel["mech"] / rel["starts"]
    field = res.groupby("season")["mech_dnf"].mean()
    fig = go.Figure()
    fig.add_bar(x=rel["season"], y=rel["Mechanical DNF rate"], name=tnames.get(team, team), marker_color=col)
    fig.add_scatter(x=field.index, y=field.values, name="Field average", mode="lines+markers", line=dict(color=ui.MUTED, dash="dot"))
    fig.update_layout(height=340, yaxis=dict(tickformat=".0%"), title="Mechanical retirement rate by season")
    st.plotly_chart(fig, width="stretch")
    reasons = t[~t["finished"]]["status"].value_counts().reset_index()
    reasons.columns = ["Reason", "Count"]
    st.dataframe(reasons, hide_index=True, width="stretch")
    st.caption("Mechanical = any non-finish whose recorded reason is not an accident, collision, damage, disqualification or non-start.")
with tab4:
    stats = ui.q("SELECT s.race_id, s.driver_id, s.median_clean_lap_s, s.best_lap_s, r.season, r.round, r.race_name FROM session_driver_stats s"
                 " JOIN races r ON r.race_id=s.race_id WHERE s.session_name='R'")
    if stats.empty:
        st.info("Race pace needs OpenF1 lap data (2023 onwards), which has not been downloaded yet.")
    else:
        stats["pace_gap"] = (stats["median_clean_lap_s"] / stats.groupby("race_id")["median_clean_lap_s"].transform("min") - 1) * 100
        stats = stats.merge(res[["race_id", "driver_id", "constructor_id"]], on=["race_id", "driver_id"])
        tp = stats[stats["constructor_id"] == team].groupby(["race_id", "season", "round", "race_name"])["pace_gap"].min().reset_index()
        tp = tp[tp["season"].between(*span)]
        if tp.empty:
            st.info("No race-pace data in the selected seasons (lap data starts in 2023).")
        else:
            fig = go.Figure(go.Scatter(x=tp["season"].astype(str) + " R" + tp["round"].astype(str), y=tp["pace_gap"], mode="lines+markers",
                                       line=dict(color=col), customdata=tp["race_name"], hovertemplate="%{customdata}: +%{y:.2f}%<extra></extra>"))
            fig.update_layout(height=340, title="Race pace: team's best median clean lap vs fastest car (%)", yaxis_title="% slower")
            st.plotly_chart(fig, width="stretch")
    qq = per_race.dropna(subset=["avg_q"])
    if len(qq):
        fig = go.Figure(go.Scatter(x=qq["season"].astype(str) + " R" + qq["round"].astype(str), y=qq["avg_q"].rolling(5, min_periods=1).mean(),
                                   mode="lines", line=dict(color="#3B82F6")))
        fig.update_layout(height=300, title="Average qualifying position (rolling 5 races)", yaxis=dict(autorange="reversed"), xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, width="stretch")
