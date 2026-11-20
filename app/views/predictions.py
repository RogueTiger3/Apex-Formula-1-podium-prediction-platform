import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui
from apex import config, explain, features, predict

ui.setup("Race predictions")
models = ui.active_models()
if not models:
    st.warning("No trained model yet. Run an update from **Data Management**.")
    st.stop()

upcoming = predict.upcoming_races()
if upcoming.empty:
    st.info("There are no upcoming races in the current calendar.")
    st.stop()

c1, c2 = st.columns([1.2, 1])
with c1:
    race_id = st.selectbox("Grand Prix", upcoming["race_id"].tolist(),
                           format_func=lambda r: ui.race_label(upcoming.set_index("race_id").loc[r]))
avail = predict.available_stages(race_id)
with c2:
    stage = st.radio("Prediction stage", list(config.STAGES), index=len(avail) - 1, horizontal=True,
                     format_func=lambda s: f"{s} · {config.STAGES[s]}")
if stage not in avail:
    need = {2: "practice lap data from OpenF1", 3: "the qualifying classification", 4: "the qualifying classification"}[stage]
    st.warning(f"Stage {stage} needs {need}, which is not available for this race yet. "
               f"Available now: {', '.join(config.STAGES[s] for s in avail)}.")
    st.stop()

model = models[stage]
run = ui.q("SELECT * FROM prediction_runs WHERE race_id=? AND stage=? AND model_version=? ORDER BY created_at DESC LIMIT 1",
           (race_id, stage, model.version))
if run.empty:
    with st.spinner("Generating prediction (model, 20,000 simulations and SHAP explanations)…"):
        predict.run_prediction(race_id, stage, models, trigger="dashboard", table=ui.feature_table(race_id))
    ui.q.clear()
    run = ui.q("SELECT * FROM prediction_runs WHERE race_id=? AND stage=? AND model_version=? ORDER BY created_at DESC LIMIT 1",
               (race_id, stage, model.version))
run = run.iloc[0]
inp = json.loads(run["inputs_json"])
p = ui.run_predictions(run["run_id"])

st.caption(f"{config.STAGES[stage]} prediction · model {run['model_version']} ({ui.models.ALGO_LABELS.get(run['algorithm'], run['algorithm'])}) · "
           f"generated {ui.fmt_ts(run['created_at'])} · inputs: "
           + ", ".join(filter(None, [
               "driver/team/circuit history",
               f"practice ({', '.join(inp['practice_sessions'])})" if inp.get("practice_sessions") and stage >= 2 else None,
               "qualifying" if stage >= 3 else None,
               "grid = qualifying order (official grid not yet published)" if stage == 4 else None,
               f"rain chance {inp['context']['p_rain']:.0%} ({inp['context']['rain_source']})",
           ])))

# ------------------------------------------------------------------ headline
top = p.iloc[0]
pw = p.sort_values("p_win", ascending=False)
mx = pw["p_win"].iloc[0]
openness = "Clear favourite" if mx >= 0.45 else "Two- or three-way fight" if mx >= 0.25 else "Open race"
h = st.columns(4)
with h[0]:
    ui.card("Predicted winner", pw.iloc[0]["name"], f"Win {ui.pct(mx)} · {pw.iloc[0]['team']}",
            bar=ui.colour(pw.iloc[0]["constructor_id"]))
for i in range(3):
    r = p.iloc[i]
    with h[i + 1]:
        ui.card(f"Podium pick {i + 1}", r["name"], f"Podium {ui.pct(r['p_podium'])} · {r['team']}",
                bar=ui.colour(r["constructor_id"], i))
st.write("")
st.markdown(f'<span class="apex-pill est">Race outlook: {openness}</span> '
            f'<span class="apex-pill">Favourite wins in {mx:.0%} of simulations</span>', unsafe_allow_html=True)

# ------------------------------------------------------------------ table and distribution
feats = inp.get("features", {})
def depth(d):
    f = feats.get(d, {})
    n = f.get("d_experience") or 0
    if n >= 40:
        return "High"
    return "Medium" if n >= 10 else "Low"
p["Data depth"] = p["driver_id"].map(depth)
st.subheader("All drivers")
if True:
    st.dataframe(ui.as_pct(p[["rank", "code", "name", "team", "p_win", "p_top2", "p_podium", "p_top5", "p_top10", "Data depth"]]),
                 hide_index=True, width="stretch", height=min(36 * len(p) + 40, 820),
                 column_config={"rank": "#", "code": "Code", "name": "Driver", "team": "Team", **ui.prob_columns()})
    st.caption("Data depth reflects how many previous starts the driver has in the database. "
               "Rookies' probabilities lean more on their team's form and carry more uncertainty.")
if True:
    st.subheader("Where each driver finishes")
    d = p.head(12).iloc[::-1]
    st.caption("Twelve most likely podium finishers. Bands are read from the same simulated races, so they always add up to 100%.")
    bands = [("Win", d["p_win"]), ("P2", d["p_top2"] - d["p_win"]), ("P3", d["p_podium"] - d["p_top2"]),
             ("P4–5", d["p_top5"] - d["p_podium"]), ("P6–10", d["p_top10"] - d["p_top5"]), ("Outside top 10", 1 - d["p_top10"])]
    colours = ["#E4002B", "#F0506E", "#F5A3B4", "#5B6B8C", "#39445A", "#232A38"]
    fig = go.Figure()
    for (name, vals), c in zip(bands, colours):
        fig.add_bar(y=d["code"], x=vals, name=name, orientation="h", marker_color=c,
                    hovertemplate="%{y} " + name + ": %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="stack", height=460, xaxis=dict(tickformat=".0%", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ explanations
st.divider()
st.subheader("Why the model thinks this")
st.caption("SHAP values: how much each input moved this driver's podium log-odds away from an average driver. "
           "Explanations are computed from the deployed model, not written by hand.")
ex1, ex2 = st.columns([1, 1.3])
with ex1:
    drv = st.selectbox("Driver", p["driver_id"].tolist(), format_func=lambda d: ui.names().get(d, d))
    row = p.set_index("driver_id").loc[drv]
    e = ui.explanation(row)
    st.markdown(f"> {e.get('text', 'No explanation stored.')}")
    pos, neg = e.get("positive", []), e.get("negative", [])
    if pos:
        st.markdown("**Raising the probability**\n" + "\n".join(
            f"- {i['category']} ({i['shap']:+.2f}): {i['label'].lower()} {i['value']}" for i in pos))
    if neg:
        st.markdown("**Lowering the probability**\n" + "\n".join(
            f"- {i['category']} ({i['shap']:+.2f}): {i['label'].lower()} {i['value']}" for i in neg))
    st.caption("Numbers in brackets are the net change in podium log-odds from that group of inputs.")
with ex2:
    shap_d = pd.Series(e.get("shap", {}), dtype=float)
    if len(shap_d):
        s = shap_d.reindex(shap_d.abs().sort_values(ascending=False).index).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=s.values, y=[features.label(f) for f in s.index], orientation="h",
                               marker_color=["#10B981" if v > 0 else "#E4002B" for v in s.values],
                               hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        fig.update_layout(height=430, title="Largest contributions (log-odds)", xaxis_title="← lowers podium chance · raises →")
        st.plotly_chart(fig, width="stretch")

# what matters for this race overall
all_shap = pd.DataFrame([ui.explanation(r).get("shap", {}) for _, r in p.iterrows()])
if not all_shap.empty:
    shares = explain.category_shares(all_shap)
    fig = go.Figure(go.Bar(x=shares["share"], y=shares["category"], orientation="h", marker_color="#3B82F6",
                           hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
    fig.update_layout(height=360, title="What drives the predictions for this race (share of total |SHAP|)",
                      xaxis=dict(tickformat=".0%"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ stage evolution
hist = ui.q("SELECT r.stage, r.created_at, p.driver_id, p.p_podium FROM prediction_runs r JOIN predictions p ON p.run_id=r.run_id"
            " WHERE r.race_id=? ORDER BY r.created_at", (race_id,))
if hist["stage"].nunique() > 1:
    st.subheader("How the prediction changed through the weekend")
    last_per_stage = hist.sort_values("created_at").groupby(["stage", "driver_id"]).tail(1)
    topd = p.head(8)["driver_id"].tolist()
    fig = go.Figure()
    for i, dv in enumerate(topd):
        g = last_per_stage[last_per_stage["driver_id"] == dv].sort_values("stage")
        fig.add_scatter(x=[config.STAGES[s] for s in g["stage"]], y=g["p_podium"], mode="lines+markers",
                        name=ui.codes().get(dv, dv), line=dict(color=ui.colour(p.set_index("driver_id").loc[dv, "constructor_id"], i)))
    fig.update_layout(height=380, yaxis=dict(tickformat=".0%", title="Podium probability"))
    st.plotly_chart(fig, width="stretch")
