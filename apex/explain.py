"""Explainability: SHAP values for the deployed models.

SHAP values are computed on the calibrated log-odds of a podium finish using
`shap.PermutationExplainer`, which works for every model type including the
ensemble. Each value is the change in log-odds attributable to one feature
relative to an average training row, and they add up exactly to the
difference between that driver's log-odds and the baseline.

The text explanations are built from these numbers and the driver's actual
feature values. Nothing is written freehand or by a language model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import features
from .models import StageModel


def _explainer(model: StageModel):
    cols = model.features
    bg = model.background[cols]

    def f(arr):
        return model.logit(pd.DataFrame(arr, columns=cols))

    return shap.PermutationExplainer(f, shap.maskers.Independent(bg.values, max_samples=len(bg)), seed=0)


def shap_values(model: StageModel, X: pd.DataFrame, max_evals: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    cols = model.features
    exp = _explainer(model)
    max_evals = max_evals or 2 * len(cols) + 1
    sv = exp(X[cols].values.astype(float), max_evals=max_evals, silent=True)
    return pd.DataFrame(sv.values, columns=cols, index=X.index), np.asarray(sv.base_values)


def global_importance(model: StageModel, X: pd.DataFrame, n: int = 150) -> pd.DataFrame:
    sample = X.sample(min(n, len(X)), random_state=3)
    sv, _ = shap_values(model, sample)
    imp = sv.abs().mean().rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    imp["label"] = imp["feature"].map(features.label)
    imp["category"] = imp["feature"].map(features.category)
    return imp.sort_values("mean_abs_shap", ascending=False)


POSITION_FEATURES = {"quali_pos", "grid", "sprint_pos", "sprint_pos_preq", "d_champ_rank", "c_champ_rank"}
RATE_FEATURES = {"d_podium_ewm", "d_win_ewm", "d_podium_career", "d_dnf_ewm", "d_tm_quali_beat_ewm",
                 "c_podium_ewm", "c_mech_dnf_ewm", "dc_podium_rate", "circ_pole_win", "circ_dnf_rate",
                 "circ_sc_rate", "circ_wet_rate"}
PERCENT_GAP_FEATURES = {"d_quali_gap_ewm", "d_race_pace_ewm", "c_race_pace_ewm", "quali_gap", "fp_best_gap",
                        "fp_longrun_gap", "fp_team_best_gap", "d_consistency_ewm"}


def _fmt(feature: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "no data"
    v = float(v)
    if feature in POSITION_FEATURES:
        return f"P{int(round(v))}"
    if feature in RATE_FEATURES:
        return f"{v:.0%}"
    if feature in PERCENT_GAP_FEATURES:
        return f"{v:.2f}%"
    if feature in ("d_finish_ewm", "d_finish_last5", "d_quali_ewm", "c_finish_ewm", "c_quali_ewm",
                   "dc_finish_mean", "cc_finish_mean"):
        return f"avg P{v:.1f}"
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}"


def contributions(sv_row: pd.Series, x_row: pd.Series, top: int = 4) -> tuple[list[dict], list[dict]]:
    """Top positive and negative drivers of a prediction, grouped by feature category.

    Correlated inputs (for example a team's points and its qualifying record) can
    split credit between them with opposite signs, so the explanation nets SHAP
    values within each category first and then names the strongest feature in
    the category as the example.
    """
    items = [dict(feature=f, label=features.label(f), category=features.category(f),
                  value=_fmt(f, x_row.get(f)), shap=float(v)) for f, v in sv_row.items()]
    cats: dict[str, dict] = {}
    for it in items:
        c = cats.setdefault(it["category"], dict(category=it["category"], shap=0.0, members=[]))
        c["shap"] += it["shap"]
        c["members"].append(it)
    out = []
    for c in cats.values():
        same_sign = [m for m in c["members"] if np.sign(m["shap"]) == np.sign(c["shap"])] or c["members"]
        lead = max(same_sign, key=lambda m: abs(m["shap"]))
        out.append(dict(category=c["category"], shap=c["shap"], feature=lead["feature"],
                        label=lead["label"], value=lead["value"]))
    pos = sorted([i for i in out if i["shap"] > 0.02], key=lambda i: -i["shap"])[:top]
    neg = sorted([i for i in out if i["shap"] < -0.02], key=lambda i: i["shap"])[:top]
    return pos, neg


def narrative(name: str, p_podium: float, pos: list[dict], neg: list[dict]) -> str:
    level = ("a strong" if p_podium >= 0.5 else "a realistic" if p_podium >= 0.2
             else "an outside" if p_podium >= 0.05 else "a slim")
    parts = [f"{name} has {level} podium chance ({p_podium:.0%})."]
    if pos:
        parts.append("The main factors raising it are " +
                     _join([f"{i['category'].lower()} ({i['label'].lower()}: {i['value']})" for i in pos[:3]]) + ".")
    if neg:
        parts.append("It is reduced by " +
                     _join([f"{i['category'].lower()} ({i['label'].lower()}: {i['value']})" for i in neg[:2]]) + ".")
    return " ".join(parts)


def _join(xs: list[str]) -> str:
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def category_shares(sv: pd.DataFrame) -> pd.DataFrame:
    """Share of total |SHAP| by feature category (how much form vs qualifying vs circuit matters)."""
    a = sv.abs().sum()
    df = pd.DataFrame({"feature": a.index, "abs": a.values})
    df["category"] = df["feature"].map(features.category)
    out = df.groupby("category")["abs"].sum()
    return (out / out.sum()).sort_values(ascending=False).rename("share").reset_index()
