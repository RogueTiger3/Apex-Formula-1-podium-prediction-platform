import json

import numpy as np
import pandas as pd

from apex import db, features, models, pipeline, predict
from apex.simulate import Scenario, simulate


def test_normalise_per_race_sums_to_three():
    p = np.array([0.9, 0.8, 0.7, 0.4, 0.1, 0.5, 0.5, 0.5, 0.5])
    races = np.array(["a"] * 5 + ["b"] * 4)
    out = models.normalise_per_race(p, races)
    assert abs(out[:5].sum() - 3) < 1e-6 and abs(out[5:].sum() - 3) < 1e-6
    assert (np.argsort(out[:5]) == np.argsort(p[:5])).all()
    assert ((out > 0) & (out < 1)).all()


def test_simulator_is_consistent_and_matches_targets():
    ent = pd.DataFrame(dict(driver_id=[f"d{i}" for i in range(10)], team=[f"t{i // 2}" for i in range(10)],
                            p_podium=[0.85, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02], p_dnf=0.05, wet_delta=0.0))
    ent["p_podium"] = ent["p_podium"] / ent["p_podium"].sum() * 3
    res, theta = simulate(ent, n_sims=20_000, p_rain=0.0, p_sc=0.0)
    s = res.summary().set_index("driver_id").loc[ent["driver_id"]]
    assert np.allclose(res.position_probs.sum(axis=1), 1)
    assert np.allclose(res.position_probs.sum(axis=0), 1)
    assert (s["p_win"] <= s["p_top2"] + 1e-12).all() and (s["p_top2"] <= s["p_podium"] + 1e-12).all()
    assert (s["p_podium"] <= s["p_top5"] + 1e-12).all() and (s["p_top5"] <= s["p_top10"] + 1e-12).all()
    assert np.abs(s["p_podium"].values - ent["p_podium"].values).max() < 0.03
    # A hypothetical pace penalty for the leading team must lower its drivers' chances.
    slow, _ = simulate(ent, scenario=Scenario(team_pace_penalty={"t0": 1.0}), n_sims=20_000, theta=theta, p_rain=0, p_sc=0)
    ss = slow.summary().set_index("driver_id")
    assert ss.loc["d0", "p_podium"] < s.loc["d0", "p_podium"]


def test_backtest_and_selection(synthetic_db):
    df = features.build()
    preds, mets = models.backtest(df, 3, "test", tune=False)
    assert set(preds["season"]) == {2022, 2023}
    assert {"logreg", "random_forest", "lightgbm", "ensemble", "bench_quali"} <= set(mets["algorithm"])
    # Training seasons always precede the test season.
    assert models.select_algorithm(mets) in {"logreg", "random_forest", "lightgbm", "ensemble"}
    ml = mets[(mets["season"] == "all") & (mets["algorithm"] == "logreg")].iloc[0]
    assert 0 < ml["brier"] < 0.25 and ml["roc_auc"] > 0.6


def test_end_to_end_prediction_is_stored_once(synthetic_db, monkeypatch):
    monkeypatch.setattr(models, "PARAM_GRID", {k: v[:1] for k, v in models.PARAM_GRID.items()})
    version = pipeline.retrain()
    stage_models = models.load_active()
    assert stage_models[1].version == version
    monkeypatch.setattr(predict, "upcoming_races", lambda limit=None: db.query(
        "SELECT * FROM races WHERE race_id='2024_01'"))
    out = predict.run_prediction("2024_01", 1, stage_models, trigger="test", n_sims=5000)
    s = out["summary"]
    assert len(s) == 10 and abs(s["p_podium"].sum() - 3) < 0.05 and abs(s["p_win"].sum() - 1) < 1e-6
    again = predict.run_prediction("2024_01", 1, stage_models, trigger="test", n_sims=5000)
    assert again["run_id"] == out["run_id"]  # identical inputs: no duplicate history row
    assert db.scalar("SELECT COUNT(*) FROM prediction_runs") == 1
    expl = json.loads(db.query("SELECT explanation FROM predictions LIMIT 1")["explanation"][0])
    assert expl["text"] and isinstance(expl["shap"], dict)
