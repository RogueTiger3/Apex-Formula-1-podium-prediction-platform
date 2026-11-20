import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui
from apex import config, predict
from apex.simulate import Scenario, simulate

ui.setup("Monte Carlo race simulator")
models = ui.active_models()
if not models:
    st.warning("No trained model yet. Run an update from **Data Management**.")
    st.stop()
upcoming = predict.upcoming_races()
if upcoming.empty:
    st.info("There are no upcoming races to simulate.")
    st.stop()

st.markdown('<span class="apex-pill est">Estimated</span> values come from the trained model and historical data. '
            '<span class="apex-pill hyp">Hypothetical</span> values are assumptions you set below.',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    race_id = st.selectbox("Grand Prix", upcoming["race_id"].tolist(),
                           format_func=lambda r: ui.race_label(upcoming.set_index("race_id").loc[r]))
avail = predict.available_stages(race_id)
with c2:
    stage = st.selectbox("Model stage", avail, index=len(avail) - 1, format_func=lambda s: f"{s} · {config.STAGES[s]}")
with c3:
    n_sims = st.select_slider("Simulated races", [1_000, 5_000, 10_000, 50_000], value=10_000)

model = models[stage]
table = ui.feature_table(race_id)


@st.cache_data(ttl=600, show_spinner="Preparing model inputs…")
def base_inputs(race_id: str, stage: int, version: str, grid_changes: tuple = ()):
    rows = predict.prepare(race_id, stage, model, table)
    for drv, places in grid_changes:
        m = rows["driver_id"] == drv
        if stage == 4:
            rows.loc[m, "grid"] = (rows.loc[m, "grid"] + places).clip(upper=len(rows))
            rows.loc[m, "grid_penalty"] = rows.loc[m, "grid_penalty"].fillna(0) + places
        else:
            rows.loc[m, "quali_pos"] = (rows.loc[m, "quali_pos"] + places).clip(upper=len(rows))
    p = model.predict_podium(rows, rows["race_id"].values)
    ent = predict.entries_frame(rows, p)
    ctx = predict.race_context(rows, race_id)
    return ent, ctx


ent0, ctx = base_inputs(race_id, stage, model.version)
names, codes, tnames = ui.names(), ui.codes(), ui.team_names()
sp = model.sim_params

# ------------------------------------------------------------------ scenario controls
with st.expander("Scenario assumptions", expanded=True):
    a, b, c = st.columns(3)
    with a:
        rain_opt = st.radio("Weather", ["Estimated", "Dry race", "Wet race"], horizontal=True,
                            help=f"Estimated chance of rain: {ctx['p_rain']:.0%} ({ctx['rain_source']}).")
        sc_opt = st.radio("Safety car", ["Estimated", "No safety car", "Safety car"], horizontal=True,
                          help=f"Estimated chance of a safety car or red flag: {ctx['p_sc']:.0%} ({ctx['sc_source']}).")
    with b:
        dnf_mult = st.slider("Reliability (DNF rate multiplier)", 0.0, 3.0, 1.0, 0.25,
                             help="1.0 = estimated DNF rates. 2.0 doubles every driver's retirement chance.")
        grid_driver = st.selectbox("Grid change: driver", ["(none)"] + ent0["driver_id"].tolist(),
                                   format_func=lambda d: d if d == "(none)" else names.get(d, d),
                                   disabled=stage < 3)
        grid_places = st.slider("Places further back", 1, 20, 5, disabled=stage < 3 or grid_driver == "(none)")
        if stage < 3:
            st.caption("Grid changes need a post-qualifying or pre-race model.")
    with c:
        teams = sorted(ent0["team"].unique())
        pace_team = st.selectbox("Team pace change", ["(none)"] + teams,
                                 format_func=lambda t: t if t == "(none)" else tnames.get(t, t.replace("_", " ").title()))
        pace_pct = st.slider("Race pace change (% per lap, positive = slower)", -1.0, 2.0, 0.5, 0.1,
                             disabled=pace_team == "(none)",
                             help="Hypothetical. For example, heavier tyre degradation that costs 0.5% per lap on average.")

scenario = Scenario(
    rain={"Estimated": None, "Dry race": False, "Wet race": True}[rain_opt],
    safety_car={"Estimated": None, "No safety car": False, "Safety car": True}[sc_opt],
    dnf_multiplier=dnf_mult,
    team_pace_penalty={pace_team: pace_pct} if pace_team != "(none)" else {},
)
grid_changes = ((grid_driver, grid_places),) if stage >= 3 and grid_driver != "(none)" else ()


@st.cache_data(ttl=600, show_spinner="Simulating races…")
def run(race_id, stage, version, n, scen_key, grid_changes):
    ent, ctx = base_inputs(race_id, stage, version, grid_changes)
    sc = Scenario(**scen_key)
    res, theta = simulate(ent, sp, sc, n_sims=n, p_rain=ctx["p_rain"], p_sc=ctx["p_sc"])
    return res


key = dict(rain=scenario.rain, safety_car=scenario.safety_car, dnf_multiplier=scenario.dnf_multiplier,
           team_pace_penalty=scenario.team_pace_penalty)
base = run(race_id, stage, model.version, n_sims, dict(rain=None, safety_car=None, dnf_multiplier=1.0, team_pace_penalty={}), ())
res = run(race_id, stage, model.version, n_sims, key, grid_changes)
is_base = scenario.is_baseline() and not grid_changes

bs, rs = base.summary(), res.summary()
rs["name"] = rs["driver_id"].map(names)
rs["code"] = rs["driver_id"].map(codes)
rs = rs.merge(bs[["driver_id", "p_podium", "p_win"]].rename(columns={"p_podium": "base_podium", "p_win": "base_win"}), on="driver_id")
rs["delta"] = rs["p_podium"] - rs["base_podium"]

lead = rs.iloc[0]
st.markdown(f"#### Across {n_sims:,} simulated races, {lead['name']} finished on the podium {int(lead['podium_count']):,} times "
            f"({lead['p_podium']:.1%}).")
st.caption(("Baseline: estimated conditions." if is_base else "Scenario applied: compared with the baseline below.")
           + f" Simulated rain share {res.rain_rate:.0%}, safety-car share {res.sc_rate:.0%}.")

if True:
    cols = ["code", "name", "p_win", "p_p2", "p_p3", "p_podium", "p_top5", "p_top10", "p_dnf"]
    if not is_base:
        cols.append("delta")
    cfg = ui.prob_columns()
    cfg.update({"code": "Code", "name": "Driver", "delta": st.column_config.NumberColumn("Podium change", format="%+.1f pts")})
    st.dataframe(ui.as_pct(rs[cols]), hide_index=True, width="stretch", height=min(36 * len(rs) + 40, 820), column_config=cfg)
l, r = st.columns([1, 1])
with l:
    st.markdown("**Most frequent podiums**")
    combos = pd.DataFrame([dict(Podium=" · ".join(codes.get(x, x) for x in c), Probability=pr,
                                Races=int(round(pr * n_sims))) for c, pr in res.podium_combos])
    st.dataframe(ui.as_pct(combos), hide_index=True, width="stretch",
                 column_config={"Probability": ui.combo_column(float(combos["Probability"].max()) * 100)})
with r:
    if is_base:
        st.markdown("**Change a scenario assumption above** to see how each driver's podium chance moves against this baseline.")
    else:
        top = rs.reindex(rs["delta"].abs().sort_values(ascending=False).index).head(8)
        fig = go.Figure(go.Bar(x=top["delta"], y=top["code"], orientation="h",
                               marker_color=["#10B981" if v > 0 else "#E4002B" for v in top["delta"]],
                               hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
        fig.update_layout(height=300, title="Biggest podium changes vs baseline", xaxis=dict(tickformat="+.0%"),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

st.subheader("Finishing position distribution")
order = [res.drivers.index(d) for d in rs["driver_id"]]
mat = res.position_probs[order]
fig = go.Figure(go.Heatmap(z=mat, x=[f"P{i}" for i in range(1, mat.shape[1] + 1)], y=rs["code"],
                           colorscale=[[0, "#0B0E14"], [0.15, "#3A0F1A"], [0.5, "#A5001F"], [1, "#FF4D6D"]],
                           hovertemplate="%{y} %{x}: %{z:.1%}<extra></extra>", colorbar=dict(tickformat=".0%")))
fig.update_layout(height=640, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("Each row sums to 100% (DNFs are placed at the back in the order they retire), so win, podium, top-five and "
           "top-ten figures are always consistent with each other.")

with st.expander("Where the simulation parameters come from"):
    st.markdown(f"""
* **Driver strengths** <span class="apex-pill est">Estimated</span> solved so that each driver's simulated podium rate matches
  the calibrated {ui.models.ALGO_LABELS.get(model.algo, model.algo)} podium probability for stage {stage}.
* **Overall randomness** <span class="apex-pill est">Estimated</span> Plackett-Luce scale {sp.get('scale', 1):.2f}, fitted on
  {sp.get('n_races', 0)} backtest races.
* **Rain** <span class="apex-pill est">Estimated</span> {"randomness ×%.2f in the wet vs ×%.2f in the dry, wet-skill weight %.2f, DNF rate ×%.2f, from %d wet races." % (sp['r_wet'], sp['r_dry'], sp['b_wet'], sp['dnf_mult_wet'], sp['n_wet_races']) if sp.get('n_wet_races', 0) >= 5 else "Too few wet races with reliable weather data (%d) to estimate a rain effect, so rain is simulated with the same parameters as dry races." % sp.get('n_wet_races', 0)}
* **Safety car** <span class="apex-pill est">Estimated</span> {"randomness ×%.2f with a safety car vs ×%.2f without, from %d safety-car races (2023 onwards, OpenF1 race control)." % (sp['r_sc'], sp['r_nosc'], sp['n_sc_races']) if sp.get('n_sc_races', 0) >= 5 else "Not enough races with race-control data yet; safety cars do not change the simulation."}
* **Race pace** <span class="apex-pill est">Estimated</span> 1% slower race pace lowers log-strength by {sp.get('a_pace', 0):.2f}
  (fitted on {sp.get('n_pace_races', 0)} races with lap data). Pace changes you enter are <span class="apex-pill hyp">Hypothetical</span>.
* **Retirements** <span class="apex-pill est">Estimated</span> average of each team's mechanical DNF rate and the driver's DNF rate, weighted to recent races.
* **Grid changes** <span class="apex-pill hyp">Hypothetical</span> input, but its effect is <span class="apex-pill est">Estimated</span>:
  the model is re-run with the new grid position.
""", unsafe_allow_html=True)
