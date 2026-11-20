"""Live predictions for upcoming races, stored permanently in the prediction history."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, db, explain, features
from .models import StageModel
from .simulate import Scenario, simulate

log = logging.getLogger("apex.predict")


def upcoming_races(limit: int | None = None) -> pd.DataFrame:
    now = datetime.now(timezone.utc) - timedelta(hours=3)
    races = db.query("SELECT r.*, c.name AS circuit_name, c.locality, c.country, c.lat, c.lon FROM races r"
                     " LEFT JOIN circuits c ON c.circuit_id=r.circuit_id WHERE r.result_status='scheduled'"
                     " ORDER BY r.race_start_utc")
    races = races[pd.to_datetime(races["race_start_utc"], utc=True) >= now]
    return races.head(limit) if limit else races


def next_race() -> pd.Series | None:
    r = upcoming_races(1)
    return r.iloc[0] if len(r) else None


def available_stages(race_id: str) -> list[int]:
    """Stages whose inputs exist for this race weekend right now."""
    stages = [1]
    if db.scalar("SELECT COUNT(*) FROM session_driver_stats WHERE race_id=? AND session_name IN ('FP1','FP2','FP3')",
                 (race_id,)):
        stages.append(2)
    if db.scalar("SELECT COUNT(*) FROM qualifying WHERE race_id=?", (race_id,)):
        stages += [3, 4]
    return stages


def race_context(feat_rows: pd.DataFrame, race_id: str) -> dict:
    """Race-level probabilities used by the simulator (rain, safety car)."""
    fc = db.query("SELECT * FROM race_weather WHERE race_id=? AND kind='forecast'", (race_id,))
    circ_wet = feat_rows["circ_wet_rate"].dropna()
    circ_sc = feat_rows["circ_sc_rate"].dropna()
    ctx = dict(p_rain=float(circ_wet.iloc[0]) if len(circ_wet) else 0.1, rain_source="circuit history",
               p_sc=float(circ_sc.iloc[0]) if len(circ_sc) else 0.6, sc_source="circuit history")
    if len(circ_sc) == 0:
        ctx["sc_source"] = "long-run average (no circuit history yet)"
    if len(fc) and pd.notna(fc.iloc[0]["precip_prob_pct"]):
        ctx["p_rain"] = float(fc.iloc[0]["precip_prob_pct"]) / 100
        ctx["rain_source"] = "Open-Meteo forecast"
        ctx["forecast"] = fc.iloc[0][["temp_c", "precip_mm", "precip_prob_pct", "wind_kmh", "humidity_pct"]].to_dict()
    return ctx


def entries_frame(rows: pd.DataFrame, p_podium: np.ndarray) -> pd.DataFrame:
    field_dnf = rows["d_dnf_ewm"].mean()
    p_dnf = (0.5 * rows["c_mech_dnf_ewm"].fillna(field_dnf) + 0.5 * rows["d_dnf_ewm"].fillna(field_dnf))
    return pd.DataFrame(dict(driver_id=rows["driver_id"].values, team=rows["team"].values,
                             constructor_id=rows["constructor_id"].values, p_podium=p_podium,
                             p_dnf=p_dnf.fillna(0.08).values, wet_delta=rows["d_wet_delta"].fillna(0).values))


def prepare(race_id: str, stage: int, model: StageModel, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for an upcoming race, filled for the chosen stage."""
    df = table if table is not None else features.build(race_id)
    rows = df[df["race_id"] == race_id].copy().reset_index(drop=True)
    if stage == 4 and rows["grid"].isna().all():
        # The official starting grid is only published with the race result, so the
        # qualifying order (which already reflects most penalties) stands in for it.
        rows["grid"] = rows["quali_pos"]
        rows["grid_penalty"] = 0.0
    if stage >= 4:
        rows["sprint_pos"] = rows["sprint_pos"]
    return rows


def run_prediction(race_id: str, stage: int, models: dict[int, StageModel], trigger: str = "auto",
                   store: bool = True, table: pd.DataFrame | None = None, n_sims: int = 20_000) -> dict:
    model = models[stage]
    rows = prepare(race_id, stage, model, table)
    if rows.empty:
        raise ValueError(f"No entry list available for {race_id}")
    p = model.predict_podium(rows, rows["race_id"].values)
    ctx = race_context(rows, race_id)
    ent = entries_frame(rows, p)
    res, theta = simulate(ent, model.sim_params, Scenario(), n_sims=n_sims, p_rain=ctx["p_rain"], p_sc=ctx["p_sc"])
    summ = res.summary().merge(ent[["driver_id", "constructor_id", "p_podium"]].rename(columns={"p_podium": "p_podium_model"}),
                               on="driver_id")
    sv, base = explain.shap_values(model, rows)
    names = driver_names()
    expl = {}
    for i, r in rows.iterrows():
        pos, neg = explain.contributions(sv.loc[i], r[model.features])
        pm = float(p[i])
        expl[r["driver_id"]] = dict(text=explain.narrative(names.get(r["driver_id"], r["driver_id"]), pm, pos, neg),
                                    positive=pos, negative=neg,
                                    shap={k: round(float(v), 4) for k, v in sv.loc[i].items()},
                                    base=float(np.atleast_1d(base)[0]))
    summ["rank"] = np.arange(1, len(summ) + 1)
    inputs = dict(
        stage=stage, stage_name=config.STAGES[stage],
        practice_sessions=db.query("SELECT DISTINCT session_name FROM session_driver_stats WHERE race_id=?"
                                   " AND session_name LIKE 'FP%'", (race_id,))["session_name"].tolist(),
        qualifying_rows=int(db.scalar("SELECT COUNT(*) FROM qualifying WHERE race_id=?", (race_id,)) or 0),
        sprint_rows=int(db.scalar("SELECT COUNT(*) FROM sprint_results WHERE race_id=?", (race_id,)) or 0),
        grid_source="qualifying order" if stage == 4 else None,
        context=ctx, n_sims=n_sims,
        podium_combos=[dict(podium=list(c), probability=round(pr, 5)) for c, pr in res.podium_combos],
        theta=dict(zip(ent["driver_id"], np.round(theta, 4))),
        features={r["driver_id"]: {f: (None if pd.isna(r[f]) else round(float(r[f]), 4)) for f in model.features}
                  for _, r in rows.iterrows()},
    )
    digest = hashlib.sha1(json.dumps(inputs["features"], sort_keys=True).encode()).hexdigest()[:12]
    inputs["input_hash"] = digest
    out = dict(race_id=race_id, stage=stage, model_version=model.version, algorithm=model.algo,
               summary=summ, explanations=expl, inputs=inputs, rows=rows, theta=theta, sim=res)
    if store:
        out["run_id"] = store_prediction(out, trigger)
    return out


def store_prediction(out: dict, trigger: str) -> str | None:
    """Append to the prediction history unless an identical prediction already exists."""
    existing = db.query("SELECT run_id, inputs_json FROM prediction_runs WHERE race_id=? AND stage=? AND model_version=?",
                        (out["race_id"], out["stage"], out["model_version"]))
    for _, r in existing.iterrows():
        if json.loads(r["inputs_json"]).get("input_hash") == out["inputs"]["input_hash"]:
            return r["run_id"]
    run_id = uuid.uuid4().hex[:16]
    summ = out["summary"]
    podium = summ.sort_values("p_podium", ascending=False)["driver_id"].head(3).tolist()
    with db.session() as conn:
        db.insert(conn, "prediction_runs", [dict(
            run_id=run_id, created_at=db.utcnow(), race_id=out["race_id"], stage=out["stage"],
            model_version=out["model_version"], algorithm=out["algorithm"], predicted_podium=json.dumps(podium),
            inputs_json=db.to_json(out["inputs"]), trigger=trigger)])
        db.insert(conn, "predictions", [dict(
            run_id=run_id, race_id=out["race_id"], driver_id=r["driver_id"], constructor_id=r["constructor_id"],
            rank=int(r["rank"]), p_win=float(r["p_win"]), p_top2=float(r["p_top2"]), p_podium=float(r["p_podium"]),
            p_top5=float(r["p_top5"]), p_top10=float(r["p_top10"]), p_podium_model=float(r["p_podium_model"]),
            explanation=db.to_json(out["explanations"][r["driver_id"]]))
            for _, r in summ.iterrows()])
    return run_id


def driver_names() -> dict[str, str]:
    d = db.query("SELECT driver_id, given_name, family_name FROM drivers")
    return dict(zip(d["driver_id"], d["given_name"] + " " + d["family_name"]))
