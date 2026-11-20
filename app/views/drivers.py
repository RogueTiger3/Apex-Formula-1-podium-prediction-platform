import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui
from apex import config, predict

ui.setup("Driver analytics")
names, tnames = ui.names(), ui.team_names()

res = ui.q("SELECT x.*, r.season, r.round, r.race_name, r.circuit_id, r.race_start_utc FROM results x"
           " JOIN races r ON r.race_id=x.race_id ORDER BY r.race_start_utc")
quali = ui.q("SELECT race_id, driver_id, position AS quali_pos, q1_s, q2_s, q3_s FROM qualifying")
res = res.merge(quali, on=["race_id", "driver_id"], how="left")
latest_season = int(res["season"].max())
current = res[res["season"] == latest_season]["driver_id"].unique().tolist()
order = ui.q("SELECT entity_id, position FROM standings WHERE kind='driver'").set_index("entity_id")["position"].to_dict()
current = sorted(current, key=lambda d: (order.get(d, 99), names.get(d, d)))
others = sorted(set(res["driver_id"]) - set(current), key=lambda d: names.get(d, d))

c1, c2 = st.columns([1.2, 1])
with c1:
    drv = st.selectbox("Driver", current + others,
                       format_func=lambda d: names.get(d, d) + ("" if d in current else " (not on the current grid)"))
d = res[res["driver_id"] == drv].copy()
seasons = sorted(d["season"].unique())
with c2:
    span = st.select_slider("Seasons", options=seasons, value=(seasons[0], seasons[-1])) if len(seasons) > 1 else (seasons[0], seasons[0])
d = d[d["season"].between(*span)]
st.caption(f"Data covers {config.FIRST_SEASON} onwards (Jolpica F1). Positions count classified finishes; retirements count as DNF.")

starts = len(d)
wins = int((d["position"] == 1).sum())
pods = int((d["position"] <= 3).sum())
finished = d["position"].notna()
k = st.columns(6)
for col, (t, v, s) in zip(k, [
    ("Starts", f"{starts}", f"{int(d['season'].nunique())} seasons"),
    ("Wins", f"{wins}", ui.pct(wins / starts if starts else None, 1)),
    ("Podiums", f"{pods}", f"{ui.pct(pods / starts if starts else None, 1)} of starts"),
    ("Average finish", f"{d.loc[finished, 'position'].mean():.1f}" if finished.any() else "–", "classified finishes"),
    ("Average qualifying", f"{d['quali_pos'].mean():.1f}" if d["quali_pos"].notna().any() else "–", ""),
    ("DNF rate", ui.pct((~finished).mean() if starts else None, 1), f"{int((~finished).sum())} retirements"),
]):
    with col:
        ui.card(t, v, s)

st.write("")
t1, t2, t3, t4, t5 = st.tabs(["Results and form", "Qualifying", "Circuits", "Team-mate comparison", "Next race"])
with t1:
    d["label"] = d["season"].astype(str) + " " + d["race_name"].str.replace(" Grand Prix", " GP")
    field = res.groupby("race_id")["driver_id"].size()
    d["finish_plot"] = d["position"].fillna(d["race_id"].map(field))
    d["podium"] = (d["position"] <= 3).astype(float)
    d["roll_pod"] = d["podium"].rolling(10, min_periods=3).mean()
    d["ewm_finish"] = d["finish_plot"].ewm(halflife=4).mean()
    fig = go.Figure()
    fig.add_scatter(x=d["label"], y=d["finish_plot"], mode="markers", name="Finish (DNF plotted at back)",
                    marker=dict(color=np.where(d["position"].isna(), "#5B6B8C", "#E4002B"), size=7),
                    customdata=np.stack([d["status"], d["grid"]], axis=1),
                    hovertemplate="%{x}<br>Finish %{y} · grid %{customdata[1]}<br>%{customdata[0]}<extra></extra>")
    fig.add_scatter(x=d["label"], y=d["ewm_finish"], mode="lines", name="Weighted form (half-life 4 races)",
                    line=dict(color="#F5B700", width=2))
    fig.update_layout(height=420, yaxis=dict(autorange="reversed", title="Position"), xaxis=dict(showticklabels=False),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    by = d.groupby("season").agg(Starts=("race_id", "count"), Wins=("position", lambda s: (s == 1).sum()),
                                 Podiums=("position", lambda s: (s <= 3).sum()), Points=("points", "sum"),
                                 AvgFinish=("position", "mean"), Teams=("constructor_id", lambda s: ", ".join(sorted({tnames.get(x, x) for x in s}))))
    by["Podium rate"] = by["Podiums"] / by["Starts"]
    st.dataframe(by.reset_index(), hide_index=True, width="stretch",
                 column_config={"season": st.column_config.NumberColumn("Season", format="%d"),
                                "AvgFinish": st.column_config.NumberColumn("Avg finish", format="%.1f"),
                                "Points": st.column_config.NumberColumn(format="%.0f"),
                                "Podium rate": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)})
    fig = go.Figure(go.Scatter(x=d["label"], y=d["roll_pod"], mode="lines", line=dict(color="#10B981"),
                               hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
    fig.update_layout(height=260, title="Podium rate, rolling 10 races", yaxis=dict(tickformat=".0%"), xaxis=dict(showticklabels=False))
    st.plotly_chart(fig, width="stretch")
with t2:
    q = d.dropna(subset=["quali_pos"]).copy()
    if q.empty:
        st.info("No qualifying data for this selection.")
    else:
        allq = ui.q("SELECT race_id, MIN(MIN(COALESCE(q1_s,9e9), COALESCE(q2_s,9e9), COALESCE(q3_s,9e9))) AS pole FROM qualifying GROUP BY race_id")
        q = q.merge(allq, on="race_id", how="left")
        q["best"] = q[["q1_s", "q2_s", "q3_s"]].min(axis=1)
        q["gap"] = (q["best"] / q["pole"] - 1) * 100
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Histogram(x=q["quali_pos"], xbins=dict(start=0.5, end=24.5, size=1), marker_color="#3B82F6"))
            fig.update_layout(height=320, title="Qualifying positions", xaxis_title="Grid slot", yaxis_title="Sessions")
            st.plotly_chart(fig, width="stretch")
        with b:
            per = q.groupby("season").agg(avg_q=("quali_pos", "mean"), gap=("gap", "median"), poles=("quali_pos", lambda s: (s == 1).sum()))
            fig = go.Figure()
            fig.add_bar(x=per.index, y=per["gap"], marker_color="#E4002B", hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
            fig.update_layout(height=320, title="Median gap to pole (%) by season", yaxis_title="%")
            st.plotly_chart(fig, width="stretch")
        st.dataframe(per.reset_index(), hide_index=True, width="stretch",
                     column_config={"season": st.column_config.NumberColumn("Season", format="%d"),
                                    "avg_q": st.column_config.NumberColumn("Avg qualifying", format="%.1f"),
                                    "gap": st.column_config.NumberColumn("Median gap to pole", format="%.2f%%"), "poles": "Poles"})
with t3:
    circ = ui.q("SELECT circuit_id, name FROM circuits")
    cn = dict(zip(circ["circuit_id"], circ["name"]))
    g = d.groupby("circuit_id").agg(Starts=("race_id", "count"), Wins=("position", lambda s: (s == 1).sum()),
                                    Podiums=("position", lambda s: (s <= 3).sum()), AvgFinish=("position", "mean"),
                                    AvgQuali=("quali_pos", "mean"), Best=("position", "min")).reset_index()
    g["Circuit"] = g["circuit_id"].map(cn)
    g["Podium rate"] = g["Podiums"] / g["Starts"]
    g = g.sort_values(["Podium rate", "AvgFinish"], ascending=[False, True])
    st.dataframe(g[["Circuit", "Starts", "Wins", "Podiums", "Podium rate", "AvgFinish", "AvgQuali", "Best"]],
                 hide_index=True, width="stretch",
                 column_config={"Podium rate": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
                                "AvgFinish": st.column_config.NumberColumn("Avg finish", format="%.1f"),
                                "AvgQuali": st.column_config.NumberColumn("Avg qualifying", format="%.1f"),
                                "Best": st.column_config.NumberColumn("Best", format="%d")})
with t4:
    mates = res.merge(d[["race_id", "constructor_id"]], on=["race_id", "constructor_id"])
    mates = mates[mates["driver_id"] != drv]
    pair = d.merge(mates[["race_id", "driver_id", "position", "quali_pos", "points"]], on="race_id", suffixes=("", "_tm"))
    if pair.empty:
        st.info("No team-mate data.")
    else:
        pair["tm"] = pair["driver_id_tm"].map(names)
        h2h = pair.groupby(["season", "tm"]).apply(lambda g: pd.Series(dict(
            Races=len(g),
            Quali_ahead=int((g["quali_pos"] < g["quali_pos_tm"]).sum()),
            Race_ahead=int((g["position"].fillna(99) < g["position_tm"].fillna(99)).sum()),
            Points=g["points"].sum(), Points_tm=g["points_tm"].sum())), include_groups=False).reset_index()
        h2h["Qualifying H2H"] = h2h["Quali_ahead"].astype(int).astype(str) + "–" + (h2h["Races"] - h2h["Quali_ahead"]).astype(int).astype(str)
        h2h["Race H2H"] = h2h["Race_ahead"].astype(int).astype(str) + "–" + (h2h["Races"] - h2h["Race_ahead"]).astype(int).astype(str)
        st.dataframe(h2h[["season", "tm", "Races", "Qualifying H2H", "Race H2H", "Points", "Points_tm"]], hide_index=True, width="stretch",
                     column_config={"season": st.column_config.NumberColumn("Season", format="%d"), "tm": "Team-mate",
                                    "Points": st.column_config.NumberColumn(f"{names.get(drv, drv).split()[-1]} pts", format="%.0f"),
                                    "Points_tm": st.column_config.NumberColumn("Team-mate pts", format="%.0f")})
        st.caption("Head-to-head counts races where both drivers were entered; a retirement counts as finishing behind.")
with t5:
    nr = predict.next_race()
    if nr is None:
        st.info("No upcoming race.")
    else:
        run = ui.latest_run(nr["race_id"])
        if run is None:
            st.info("No prediction stored yet for the next race. Open **Race Predictions** to generate one.")
        else:
            p = ui.run_predictions(run["run_id"]).set_index("driver_id")
            if drv not in p.index:
                st.info(f"{names.get(drv, drv)} is not on the entry list for the {nr['race_name']}.")
            else:
                r = p.loc[drv]
                st.markdown(f"**{nr['race_name']}** · stage {int(run['stage'])} ({config.STAGES[int(run['stage'])]}) · "
                            f"ranked {int(r['rank'])} of {len(p)} on podium probability")
                k = st.columns(5)
                for col, (lab, c) in zip(k, [("Win", "p_win"), ("Top 2", "p_top2"), ("Podium", "p_podium"), ("Top 5", "p_top5"), ("Top 10", "p_top10")]):
                    with col:
                        ui.card(lab, ui.pct(r[c], 1))
                st.markdown(f"> {ui.explanation(r).get('text', '')}")
