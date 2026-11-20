import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui

ui.setup("Circuit analytics")
names, tnames = ui.names(), ui.team_names()
races = ui.races()
circ = ui.q("SELECT * FROM circuits")
counts = races[races["result_status"] != "scheduled"].groupby("circuit_id").size()
upcoming_ids = races[races["result_status"] == "scheduled"].sort_values("race_start_utc")["circuit_id"].tolist()
opts = upcoming_ids + sorted(set(circ["circuit_id"]) - set(upcoming_ids), key=lambda c: -counts.get(c, 0))
cname = dict(zip(circ["circuit_id"], circ["name"] + " · " + circ["locality"].fillna("")))
cid = st.selectbox("Circuit", opts, format_func=lambda c: cname.get(c, c) + (" (on this season's remaining calendar)" if c in upcoming_ids else ""))

rr = races[(races["circuit_id"] == cid) & (races["result_status"] != "scheduled")]
if rr.empty:
    st.info("No completed races at this circuit in the database yet (new venue). Circuit features fall back to all-circuit averages.")
    st.stop()
ids = tuple(rr["race_id"])
ph = ",".join("?" * len(ids))
res = ui.q(f"SELECT * FROM results WHERE race_id IN ({ph})", ids)
pits = ui.q(f"SELECT race_id, driver_id, COUNT(*) AS stops FROM pit_stops WHERE race_id IN ({ph}) GROUP BY race_id, driver_id", ids)
ev = ui.q(f"SELECT * FROM race_events WHERE race_id IN ({ph})", ids)
wx = ui.q(f"SELECT * FROM race_weather WHERE kind='observed' AND race_id IN ({ph})", ids)
sw = ui.q(f"SELECT * FROM session_weather WHERE session_name='R' AND race_id IN ({ph})", ids)
deg = ui.q(f"SELECT race_id, deg_slope_s_per_lap, n_stints FROM session_driver_stats WHERE session_name='R' AND race_id IN ({ph})", ids)

ok = res[(res["grid"] > 0) & res["position"].notna()]
corr = ok.groupby("race_id").apply(lambda g: g["grid"].corr(g["position"], method="spearman"), include_groups=False)
pole = res[res["grid"] == 1]
k = st.columns(6)
items = [
    ("Races stored", f"{len(rr)}", f"{rr['season'].min()}–{rr['season'].max()}"),
    ("Grid–finish correlation", f"{corr.mean():.2f}", "higher = harder to overtake"),
    ("Pole-to-win rate", ui.pct((pole["position"] == 1).mean()), f"{int((pole['position'] == 1).sum())} of {len(pole)}"),
    ("Retirement rate", ui.pct(res["position"].isna().mean(), 1), "per car start"),
    ("Safety car or red flag", ui.pct(((ev["safety_cars"] + ev["red_flags"]) > 0).mean()) if len(ev) else "–",
     f"{len(ev)} races with race-control data (2023+)" if len(ev) else "race-control data starts 2023"),
    ("Pit stops per driver", f"{pits['stops'].mean():.2f}" if len(pits) else "–", "2018 onwards" if len(pits) else "no pit data"),
]
for c, it in zip(k, items):
    with c:
        ui.card(*it)
st.write("")

t1, t2, t3, t4 = st.tabs(["Winners and results", "Driver and team suitability", "Weather", "Tyres and strategy"])
with t1:
    w = res[res["position"] <= 3].merge(rr[["race_id", "season", "race_name"]], on="race_id")
    w["Driver"] = w["driver_id"].map(names)
    w["Team"] = w["constructor_id"].map(tnames)
    piv = w.pivot_table(index=["race_id", "season", "race_name"], columns="position", values="Driver", aggfunc="first").reset_index()
    piv.columns = ["race_id", "Season", "Race", "Winner", "Second", "Third"][: len(piv.columns)]
    pole_by_race = dict(zip(pole["race_id"], pole["driver_id"].map(names)))
    piv["From pole?"] = piv["race_id"].map(pole_by_race) == piv["Winner"]
    piv = piv.drop(columns="race_id")
    st.dataframe(piv.sort_values("Season", ascending=False), hide_index=True, width="stretch",
                 column_config={"Season": st.column_config.NumberColumn(format="%d")})
    fig = go.Figure(go.Bar(x=rr.set_index("race_id").loc[corr.index, "season"], y=corr.values, marker_color="#3B82F6"))
    fig.update_layout(height=280, title="Grid vs finish rank correlation by year", yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, width="stretch")
with t2:
    min_starts = st.slider("Minimum starts here", 1, 8, 2)
    g = res.groupby("driver_id").agg(Starts=("race_id", "count"), Wins=("position", lambda s: (s == 1).sum()),
                                     Podiums=("position", lambda s: (s <= 3).sum()), AvgFinish=("position", "mean"),
                                     Gain=("grid", "mean")).reset_index()
    fin = res.assign(gain=res["grid"] - res["position"]).groupby("driver_id")["gain"].mean()
    g["Places gained"] = g["driver_id"].map(fin)
    g = g[g["Starts"] >= min_starts]
    g["Podium rate"] = g["Podiums"] / g["Starts"]
    g["Driver"] = g["driver_id"].map(names)
    a, b = st.columns(2)
    with a:
        st.markdown("**Drivers**")
        st.dataframe(g.sort_values(["Podium rate", "AvgFinish"], ascending=[False, True])[["Driver", "Starts", "Wins", "Podiums", "Podium rate", "AvgFinish", "Places gained"]],
                     hide_index=True, width="stretch",
                     column_config={"Podium rate": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
                                    "AvgFinish": st.column_config.NumberColumn("Avg finish", format="%.1f"),
                                    "Places gained": st.column_config.NumberColumn(format="%+.1f")})
    with b:
        st.markdown("**Constructors**")
        c = res.groupby("constructor_id").agg(Entries=("race_id", "count"), Wins=("position", lambda s: (s == 1).sum()),
                                              Podiums=("position", lambda s: (s <= 3).sum()), AvgFinish=("position", "mean"),
                                              Points=("points", "sum")).reset_index()
        c["Team"] = c["constructor_id"].map(tnames)
        c["Points per entry"] = c["Points"] / c["Entries"]
        st.dataframe(c.sort_values("Points per entry", ascending=False)[["Team", "Entries", "Wins", "Podiums", "AvgFinish", "Points per entry"]],
                     hide_index=True, width="stretch",
                     column_config={"AvgFinish": st.column_config.NumberColumn("Avg finish", format="%.1f"),
                                    "Points per entry": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Suitability is historical performance at this venue only. Small samples are noisy; the model shrinks them towards each driver's overall record.")
with t3:
    if wx.empty and sw.empty:
        st.info("No weather records for this circuit yet. Open-Meteo archive data fills in as the daily request quota allows; OpenF1 track weather starts in 2023.")
    else:
        if len(wx):
            w = wx.merge(rr[["race_id", "season"]], on="race_id").sort_values("season")
            st.markdown("**Race-window weather (Open-Meteo archive, two hours from lights out)**")
            st.dataframe(w[["season", "temp_c", "precip_mm", "wind_kmh", "humidity_pct"]], hide_index=True, width="stretch",
                         column_config={"season": st.column_config.NumberColumn("Season", format="%d"),
                                        "temp_c": st.column_config.NumberColumn("Air temp °C", format="%.1f"),
                                        "precip_mm": st.column_config.NumberColumn("Rain mm", format="%.1f"),
                                        "wind_kmh": st.column_config.NumberColumn("Wind km/h", format="%.0f"),
                                        "humidity_pct": st.column_config.NumberColumn("Humidity %", format="%.0f")})
        if len(sw):
            s = sw.merge(rr[["race_id", "season"]], on="race_id").sort_values("season")
            st.markdown("**Trackside weather during the race (OpenF1)**")
            st.dataframe(s[["season", "air_temp_c", "track_temp_c", "humidity_pct", "wind_speed_ms", "rainfall_fraction"]],
                         hide_index=True, width="stretch",
                         column_config={"season": st.column_config.NumberColumn("Season", format="%d"),
                                        "air_temp_c": st.column_config.NumberColumn("Air °C", format="%.1f"),
                                        "track_temp_c": st.column_config.NumberColumn("Track °C", format="%.1f"),
                                        "humidity_pct": st.column_config.NumberColumn("Humidity %", format="%.0f"),
                                        "wind_speed_ms": st.column_config.NumberColumn("Wind m/s", format="%.1f"),
                                        "rainfall_fraction": st.column_config.NumberColumn("Time with rain", format="percent")})
with t4:
    if len(pits):
        pp = pits.merge(rr[["race_id", "season"]], on="race_id").groupby("season")["stops"].mean()
        fig = go.Figure(go.Bar(x=pp.index, y=pp.values, marker_color="#F5B700"))
        fig.update_layout(height=280, title="Average pit stops per driver", yaxis_title="Stops")
        st.plotly_chart(fig, width="stretch")
    if len(deg.dropna(subset=["deg_slope_s_per_lap"])):
        dd = deg.merge(rr[["race_id", "season"]], on="race_id").groupby("season")["deg_slope_s_per_lap"].median()
        fig = go.Figure(go.Bar(x=dd.index, y=dd.values, marker_color="#E4002B"))
        fig.update_layout(height=280, title="Median lap-time change per lap of tyre age (s/lap, OpenF1 long stints)",
                          yaxis_title="s per lap")
        st.plotly_chart(fig, width="stretch")
        st.caption("Measured on clean laps within stints of five or more laps. Fuel burn-off makes cars faster as the race goes on, "
                   "so raw values can be negative; compare circuits and years rather than reading the number as pure tyre wear.")
    else:
        st.info("Tyre degradation needs OpenF1 lap data (2023 onwards).")
