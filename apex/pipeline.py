"""End-to-end update pipeline. This is what GitHub Actions and the dashboard's
refresh button run.

    python -m apex.pipeline update            # normal scheduled run
    python -m apex.pipeline update --retrain  # force a retrain
    python -m apex.pipeline predict           # just refresh predictions

Steps: ingest new data -> (if a race result became final, or no model exists)
rebuild features, backtest, select and retrain models -> predict the next race
at every stage whose inputs are available -> log the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback

import pandas as pd

from . import config, db, explain, features, ingest, models, predict
from .simulate import fit_scenario_params

log = logging.getLogger("apex.pipeline")


def retrain(tune: bool = True) -> str:
    """Backtest every stage, choose the best algorithm per stage and train the live models."""
    df = features.build()
    cutoff = df.loc[df["is_target"] == 0, "race_id"].max()
    version = models.new_version(cutoff)
    log.info("Retraining %s on %d rows", version, len(df))
    stage_models = {}
    for stage in config.STAGES:
        preds, mets = models.backtest(df, stage, version, tune=tune)
        algo = models.select_algorithm(mets)
        sm = models.train_final(df, stage, algo, version)
        bt_sel = preds[preds["algorithm"] == algo]
        sm.sim_params = fit_scenario_params(bt_sel, df)
        imp = explain.global_importance(sm, models.stage_frame(df, stage).query("season == season.max()"))
        stage_models[stage] = sm
        models.save_backtest(preds, stage)
        with db.session() as conn:
            db.upsert(conn, "backtest_metrics", mets.to_dict("records"))
            db.upsert(conn, "feature_importance", imp.assign(model_version=version, stage=stage).to_dict("records"))
            conn.execute("UPDATE model_registry SET is_active=0 WHERE stage=?", (stage,))
            db.upsert(conn, "model_registry", [dict(
                model_version=version, stage=stage, algorithm=algo, trained_at=sm.trained_at,
                data_cutoff_race=sm.data_cutoff_race, n_train=sm.n_train, n_features=len(sm.features),
                metrics_json=db.to_json(mets[mets["season"] == "all"].set_index("algorithm")
                                        [["log_loss", "brier", "roc_auc", "top3_hit_rate"]].to_dict("index")),
                params_json=db.to_json({"members": [dict(algo=m.algo, params=m.params, calib=m.calib) for m in sm.members],
                                        "sim_params": sm.sim_params}),
                path=f"apex_{version}.joblib", is_active=1)])
        log.info("Stage %d: selected %s", stage, algo)
    models.save(stage_models, version)
    return version


def predict_upcoming(stage_models, trigger: str, n_races: int = 1) -> list[str]:
    runs = []
    races = predict.upcoming_races(n_races)
    if races.empty:
        return runs
    for _, race in races.iterrows():
        table = features.build(race["race_id"])
        for stage in predict.available_stages(race["race_id"]):
            out = predict.run_prediction(race["race_id"], stage, stage_models, trigger=trigger, table=table)
            runs.append(f"{race['race_id']} stage {stage}: {out.get('run_id')}")
    return runs


FINGERPRINT_SQL = [
    "SELECT race_id, result_status FROM races ORDER BY race_id",
    "SELECT race_id, driver_id, position, position_text, grid, points, status FROM results ORDER BY race_id, driver_id",
    "SELECT race_id, driver_id, position, q1_s, q2_s, q3_s FROM qualifying ORDER BY race_id, driver_id",
    "SELECT race_id, driver_id, position FROM sprint_results ORDER BY race_id, driver_id",
    "SELECT race_id, session_name, driver_id FROM session_driver_stats ORDER BY 1, 2, 3",
    "SELECT race_id, kind, temp_c, precip_mm, precip_prob_pct FROM race_weather ORDER BY 1, 2",
    "SELECT COUNT(*) FROM pit_stops",
    "SELECT kind, entity_id, points FROM standings ORDER BY 1, 2",
    "SELECT run_id FROM prediction_runs ORDER BY run_id",
    "SELECT model_version FROM model_registry ORDER BY 1",
]


def data_fingerprint() -> str:
    """Hash of the substantive data (ignores retrieval timestamps and logs)."""
    import hashlib
    h = hashlib.sha1()
    with db.session() as conn:
        for sql in FINGERPRINT_SQL:
            for row in conn.execute(sql):
                h.update(repr(row).encode())
    return h.hexdigest()


def update(trigger: str = "manual", force_retrain: bool = False, skip_weather: bool = False) -> dict:
    db.init_db()
    with db.session() as conn:
        cur = conn.execute("INSERT INTO update_log (started_at, trigger, status) VALUES (?,?,?)",
                           (db.utcnow(), trigger, "running"))
        log_id = cur.lastrowid
    summary = dict(errors=[])
    before = data_fingerprint()
    status = "success"
    retrained = False
    try:
        ing = ingest.sync_all(include_weather=not skip_weather)
        summary.update(newly_final=ing.newly_final_races, provisional=ing.provisional_races,
                       sessions=ing.sessions_processed, weather=ing.weather_rows)
        summary["errors"] += ing.errors
        stage_models = models.load_active()
        if force_retrain or stage_models is None or ing.has_new_final_data:
            retrain()
            retrained = True
            stage_models = models.load_active()
        else:
            log.info("No newly confirmed results: model not retrained")
        summary["predictions"] = predict_upcoming(stage_models, trigger)
        if summary["errors"]:
            status = "partial"
    except Exception as exc:  # record the failure, keep previous data and models untouched
        status = "failed"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["traceback"] = traceback.format_exc()[-2000:]
        log.exception("Update failed")
    with db.session() as conn:
        conn.execute("UPDATE update_log SET finished_at=?, status=?, new_races=?, retrained=?, summary=? WHERE id=?",
                     (db.utcnow(), status, len(summary.get("newly_final", [])), int(retrained),
                      db.to_json(summary), log_id))
        # Trim the request log so the database stays small.
        conn.execute("DELETE FROM fetch_log WHERE id NOT IN (SELECT id FROM fetch_log ORDER BY id DESC LIMIT 5000)")
    with db.session() as conn:
        conn.execute("VACUUM")
    summary["status"] = status
    summary["changed"] = data_fingerprint() != before
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="apex")
    sub = parser.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--retrain", action="store_true")
    u.add_argument("--trigger", default="manual")
    u.add_argument("--skip-weather", action="store_true")
    sub.add_parser("retrain")
    sub.add_parser("predict")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "update":
        s = update(args.trigger, args.retrain, args.skip_weather)
        print(json.dumps({k: v for k, v in s.items() if k != "traceback"}, indent=2, default=str))
        return 0 if s["status"] != "failed" else 1
    if args.cmd == "retrain":
        db.init_db()
        print(retrain())
        return 0
    if args.cmd == "predict":
        print("\n".join(predict_upcoming(models.load_active(), "manual")))
        return 0


if __name__ == "__main__":
    sys.exit(main())
