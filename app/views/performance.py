import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

from app import ui
from apex import config, features
from apex.models import ALGO_LABELS

ui.setup("Model performance")
reg = ui.q("SELECT * FROM model_registry WHERE is_active=1 ORDER BY stage")
if reg.empty:
    st.warning("No trained model yet.")
    st.stop()
version = reg["model_version"].iloc[0]
stage = st.radio("Prediction stage", list(config.STAGES), index=3, horizontal=True, format_func=lambda s: f"{s} · {config.STAGES[s]}")
sel = reg.set_index("stage").loc[stage]
mets = ui.q("SELECT * FROM backtest_metrics WHERE model_version=? AND stage=?", (version, stage))
bt = ui.q("SELECT * FROM backtest_predictions WHERE model_version=? AND stage=?", (version, stage))
if mets.empty:
    st.info("No backtest results stored for this model version.")
    st.stop()
mets["Model"] = mets["algorithm"].map(ALGO_LABELS)
allm = mets[mets["season"] == "all"].sort_values("log_loss")
seasons = sorted(s for s in mets["season"].unique() if s != "all")

st.markdown(f"Expanding-window backtest: each season from **{seasons[0]}** to **{seasons[-1]}** is predicted by models trained only on "
            f"earlier seasons. Hyper-parameters and probability calibration use the last training season. "
            f"Selected for live use: **{ALGO_LABELS.get(sel['algorithm'], sel['algorithm'])}** (lowest average log loss).")

best_ml = allm[~allm["algorithm"].str.startswith("bench")].iloc[0]
best_bench = allm[allm["algorithm"].str.startswith("bench")].sort_values("log_loss").iloc[0]
skill = 1 - best_ml["brier"] / best_bench["brier"]
k = st.columns(4)
with k[0]:
    ui.card("Podium hit rate (top-3 picks)", ui.pct(best_ml["top3_hit_rate"], 1), f"{best_ml['Model']}")
with k[1]:
    ui.card("Brier score", f"{best_ml['brier']:.4f}", f"best benchmark {best_bench['brier']:.4f}")
with k[2]:
    ui.card("Improvement over best benchmark", f"{skill:+.1%}", f"Brier skill vs {best_bench['Model'].replace('Benchmark: ', '')}")
with k[3]:
    ui.card("ROC-AUC", f"{best_ml['roc_auc']:.3f}", f"{int(best_ml['n_races'])} races, {int(best_ml['n_rows']):,} driver entries")
if skill < 0.03:
    st.warning(f"Honest read-out: at this stage the best model is only {skill:.1%} better (Brier) than the simplest sensible benchmark. "
               "Qualifying position already carries most of the predictable signal.")

st.subheader("Model comparison (all backtest seasons)")
st.dataframe(allm[["Model", "log_loss", "brier", "roc_auc", "top3_hit_rate", "accuracy", "precision_", "recall", "f1"]],
             hide_index=True, width="stretch",
             column_config={"log_loss": st.column_config.NumberColumn("Log loss ↓", format="%.4f"),
                            "brier": st.column_config.NumberColumn("Brier ↓", format="%.4f"),
                            "roc_auc": st.column_config.NumberColumn("ROC-AUC ↑", format="%.3f"),
                            "top3_hit_rate": st.column_config.NumberColumn("Podium hit rate ↑", format="percent"),
                            "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                            "precision_": st.column_config.NumberColumn("Precision", format="percent"),
                            "recall": st.column_config.NumberColumn("Recall", format="percent"),
                            "f1": st.column_config.NumberColumn("F1", format="%.3f")})
st.caption("Podium hit rate: share of each race's three highest-probability drivers who actually finished on the podium. "
           "Accuracy, precision, recall and F1 use a 50% threshold. Accuracy looks high for every model because about 85% of entries "
           "are not on the podium; log loss and Brier score are the fairer comparison of probabilities.")

metric = st.radio("Season-by-season metric", ["log_loss", "brier", "top3_hit_rate", "roc_auc"], horizontal=True,
                  format_func={"log_loss": "Log loss", "brier": "Brier score", "top3_hit_rate": "Podium hit rate", "roc_auc": "ROC-AUC"}.get)
ps = mets[mets["season"] != "all"]
fig = go.Figure()
for i, (algo, g) in enumerate(ps.groupby("algorithm")):
    g = g.sort_values("season")
    dash = "dot" if algo.startswith("bench") else "solid"
    fig.add_scatter(x=g["season"], y=g[metric], mode="lines+markers", name=ALGO_LABELS.get(algo, algo),
                    line=dict(dash=dash, width=3 if algo == sel["algorithm"] else 1.5))
fig.update_layout(height=400, legend=dict(orientation="h", y=-0.2),
                  yaxis=dict(tickformat=".0%" if metric == "top3_hit_rate" else None))
st.plotly_chart(fig, width="stretch")

a, b = st.columns(2)
algos = [x for x in bt["algorithm"].unique()]
with a:
    st.subheader("Calibration")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.MUTED, dash="dash"), name="Perfect")
    for algo in [sel["algorithm"], best_bench["algorithm"]] + (["logreg"] if sel["algorithm"] != "logreg" else []):
        g = bt[bt["algorithm"] == algo]
        if g.empty:
            continue
        frac, mean = calibration_curve(g["actual_podium"], g["prob_podium"], n_bins=10, strategy="quantile")
        fig.add_scatter(x=mean, y=frac, mode="lines+markers", name=ALGO_LABELS.get(algo, algo))
    fig.update_layout(height=420, xaxis=dict(title="Predicted podium probability", tickformat=".0%"),
                      yaxis=dict(title="Observed podium rate", tickformat=".0%"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
with b:
    st.subheader("ROC curves")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.MUTED, dash="dash"), name="Chance")
    for algo in algos:
        g = bt[bt["algorithm"] == algo]
        fpr, tpr, _ = roc_curve(g["actual_podium"], g["prob_podium"])
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=ALGO_LABELS.get(algo, algo),
                        line=dict(dash="dot" if algo.startswith("bench") else "solid"))
    fig.update_layout(height=420, xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")

c, d = st.columns(2)
g = bt[bt["algorithm"] == sel["algorithm"]].copy()
with c:
    st.subheader("Confusion matrices")
    mode = st.radio("Rule", ["Top three per race", "Probability ≥ 50%"], horizontal=True)
    if mode.startswith("Top"):
        pred = (g.groupby("race_id")["prob_podium"].rank(ascending=False, method="first") <= 3).astype(int)
    else:
        pred = (g["prob_podium"] >= 0.5).astype(int)
    cm = confusion_matrix(g["actual_podium"], pred)
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: no podium", "Predicted: podium"], y=["Actual: no podium", "Actual: podium"],
                               text=cm, texttemplate="%{text:,}", colorscale=[[0, "#141924"], [1, "#E4002B"]], showscale=False))
    fig.update_layout(height=340, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
with d:
    st.subheader("Feature importance (SHAP)")
    imp = ui.q("SELECT * FROM feature_importance WHERE model_version=? AND stage=? ORDER BY mean_abs_shap DESC", (version, stage))
    if len(imp):
        top = imp.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["label"], orientation="h", marker_color="#3B82F6",
                               customdata=top["category"], hovertemplate="%{y} (%{customdata}): %{x:.3f}<extra></extra>"))
        fig.update_layout(height=340, xaxis_title="Mean |SHAP| (log-odds)")
        st.plotly_chart(fig, width="stretch")

if len(imp):
    cat = imp.groupby("category")["mean_abs_shap"].sum()
    cat = (cat / cat.sum()).sort_values()
    fig = go.Figure(go.Bar(x=cat.values, y=cat.index, orientation="h", marker_color="#10B981",
                           hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
    fig.update_layout(height=380, title="How much each kind of information matters (share of mean |SHAP|, most recent season)",
                      xaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ live prediction history
st.divider()
st.subheader("Prediction history (live predictions vs actual results)")
hist = ui.q("""SELECT r.run_id, r.created_at, r.race_id, r.stage, r.model_version, r.predicted_podium, ra.race_name, ra.result_status
               FROM prediction_runs r JOIN races ra ON ra.race_id=r.race_id ORDER BY r.created_at DESC""")
if hist.empty:
    st.info("No live predictions stored yet. Every prediction the app makes is kept here permanently and scored once the race result is in.")
else:
    res = ui.q("SELECT race_id, driver_id, position FROM results")
    rows = []
    for _, r in hist.iterrows():
        actual = res[(res["race_id"] == r["race_id"]) & (res["position"] <= 3)]["driver_id"].tolist()
        pred = json.loads(r["predicted_podium"])
        brier = None
        if actual:
            p = ui.q("SELECT driver_id, p_podium FROM predictions WHERE run_id=?", (r["run_id"],))
            y = p["driver_id"].isin(actual).astype(int)
            brier = float(((p["p_podium"] - y) ** 2).mean())
        rows.append(dict(Race=f"{r['race_id'][:4]} {r['race_name']}", Stage=config.STAGES[int(r["stage"])], Made=ui.fmt_ts(r["created_at"]),
                         Model=r["model_version"], Predicted=", ".join(ui.codes().get(x, x) for x in pred),
                         Actual=", ".join(ui.codes().get(x, x) for x in actual) if actual else "awaiting result",
                         Correct=len(set(pred) & set(actual)) if actual else None, Brier=brier))
    h = pd.DataFrame(rows)
    scored = h.dropna(subset=["Correct"])
    if len(scored):
        k = st.columns(3)
        with k[0]:
            ui.card("Scored live predictions", f"{len(scored)}", f"{scored['Race'].nunique()} races")
        with k[1]:
            ui.card("Average podium picks correct", f"{scored['Correct'].mean():.2f} / 3")
        with k[2]:
            ui.card("Average Brier score", f"{scored['Brier'].mean():.4f}")
    st.dataframe(h, hide_index=True, width="stretch",
                 column_config={"Brier": st.column_config.NumberColumn(format="%.4f"), "Correct": st.column_config.NumberColumn("Correct (of 3)", format="%d")})

st.subheader("Model version history")
vh = ui.q("""SELECT g.model_version, g.trained_at, g.data_cutoff_race, g.algorithm, g.n_train, m.log_loss, m.brier, m.top3_hit_rate
             FROM model_registry g JOIN backtest_metrics m ON m.model_version=g.model_version AND m.stage=g.stage AND m.algorithm=g.algorithm AND m.season='all'
             WHERE g.stage=? ORDER BY g.trained_at""", (stage,))
st.dataframe(vh, hide_index=True, width="stretch",
             column_config={"log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
                            "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
                            "top3_hit_rate": st.column_config.NumberColumn("Podium hit rate", format="percent")})
st.caption("Each retrain re-runs the full backtest, so versions are compared on the same seasons plus any newly completed races.")
