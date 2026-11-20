"""Model training, chronological backtesting, calibration and selection.

Primary target: podium (finish in the top three).

For each prediction stage we compare:
  * logistic regression (baseline ML model)
  * random forest
  * LightGBM gradient boosting
  * an ensemble (average log-odds of the two best models)
and three non-ML benchmarks (qualifying position, recent form, constructor
championship position).

Validation is an expanding window: to score season Y, models are trained only
on seasons before Y. Hyper-parameters are chosen on the last training season
(again chronologically), and that same held-out season fits the Platt
calibration. The algorithm with the lowest mean log loss across backtest
seasons is selected for live predictions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, db, features

log = logging.getLogger("apex.models")

ALGORITHMS = ["logreg", "random_forest", "lightgbm"]
ALGO_LABELS = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
               "lightgbm": "Gradient Boosting (LightGBM)", "ensemble": "Ensemble",
               "bench_quali": "Benchmark: qualifying position",
               "bench_form": "Benchmark: recent finishing positions",
               "bench_constructor": "Benchmark: constructor standings"}

PARAM_GRID = {
    "logreg": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0)],
    "random_forest": [{"max_depth": 6, "min_samples_leaf": 20}, {"max_depth": 10, "min_samples_leaf": 10},
                      {"max_depth": None, "min_samples_leaf": 30}],
    "lightgbm": [{"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 300, "min_child_samples": 40},
                 {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 300, "min_child_samples": 40},
                 {"num_leaves": 7, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 80},
                 {"num_leaves": 31, "learning_rate": 0.02, "n_estimators": 400, "min_child_samples": 60}],
}
EPS = 1e-4


def make_estimator(algo: str, params: dict):
    if algo == "logreg":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                         ("scale", StandardScaler()),
                         ("model", LogisticRegression(C=params["C"], max_iter=3000))])
    if algo == "random_forest":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                         ("model", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=7,
                                                          max_features="sqrt", **params))])
    if algo == "lightgbm":
        return LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=7, verbose=-1, **params)
    raise ValueError(algo)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def normalise_per_race(p: np.ndarray, race_ids: np.ndarray, slots: int = 3) -> np.ndarray:
    """Rescale probabilities so each race's podium probabilities sum to `slots`.

    Done in log-odds space (a per-race shift), which keeps every value in (0, 1)
    and preserves the ordering within the race.
    """
    p = np.asarray(p, dtype=float).copy()
    out = p.copy()
    for rid in np.unique(race_ids):
        m = race_ids == rid
        target = min(slots, m.sum())
        z = logit(p[m])
        lo, hi = -20.0, 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if sigmoid(z + mid).sum() > target:
                hi = mid
            else:
                lo = mid
        out[m] = sigmoid(z + (lo + hi) / 2)
    return out


@dataclass
class FittedModel:
    algo: str
    params: dict
    estimator: object
    calib: tuple[float, float]  # Platt a, b on raw log-odds
    features: list[str]

    def raw_logit(self, X: pd.DataFrame) -> np.ndarray:
        return logit(self.estimator.predict_proba(X[self.features])[:, 1])

    def calibrated_logit(self, X: pd.DataFrame) -> np.ndarray:
        a, b = self.calib
        return a * self.raw_logit(X) + b


@dataclass
class StageModel:
    """What gets saved and used for live predictions: one per stage."""
    stage: int
    algo: str  # selected algorithm, or "ensemble"
    members: list[FittedModel]
    features: list[str]
    version: str = ""
    trained_at: str = ""
    n_train: int = 0
    data_cutoff_race: str = ""
    sim_params: dict = field(default_factory=dict)
    background: pd.DataFrame | None = None  # sample of training rows for SHAP

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.calibrated_logit(X) for m in self.members], axis=0)

    def predict_podium(self, X: pd.DataFrame, race_ids) -> np.ndarray:
        return normalise_per_race(sigmoid(self.logit(X)), np.asarray(race_ids))


def _fit_platt(raw_logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(raw_logit.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_algo(train: pd.DataFrame, feats: list[str], algo: str, target: str = "y_podium",
             tune: bool = True) -> tuple[FittedModel, float]:
    """Tune on the last training season, calibrate on it, then refit on everything.

    Returns the fitted model and its validation log loss (used for ensembling).
    """
    seasons = sorted(train["season"].unique())
    val_season = seasons[-1]
    inner_tr = train[train["season"] < val_season]
    val = train[train["season"] == val_season]
    grid = PARAM_GRID[algo] if tune else PARAM_GRID[algo][:1]
    best = None
    for params in grid:
        est = make_estimator(algo, params)
        est.fit(inner_tr[feats], inner_tr[target])
        p = est.predict_proba(val[feats])[:, 1]
        ll = log_loss(val[target], np.clip(p, EPS, 1 - EPS), labels=[0, 1])
        if best is None or ll < best[0]:
            best = (ll, params, logit(p))
    ll, params, val_logit = best
    calib = _fit_platt(val_logit, val[target].values)
    est = make_estimator(algo, params)
    est.fit(train[feats], train[target])
    cal_ll = log_loss(val[target], sigmoid(calib[0] * val_logit + calib[1]), labels=[0, 1])
    return FittedModel(algo, params, est, calib, feats), cal_ll


# ---------------------------------------------------------------------------
# benchmarks: simple lookup tables fitted on the training seasons
# ---------------------------------------------------------------------------

def _rank_lookup(train: pd.DataFrame, test: pd.DataFrame, col: str, target="y_podium") -> np.ndarray:
    """P(podium | rank of `col` within the race), estimated on training seasons."""
    def ranks(d):
        return d.groupby("race_id")[col].rank(method="first").fillna(25).clip(upper=25)
    tr_rank, te_rank = ranks(train), ranks(test)
    table = train.assign(r=tr_rank).groupby("r")[target].mean()
    smooth = (table * 0 + train[target].mean())
    table = table.reindex(range(1, 26)).fillna(smooth.mean() if len(smooth) else 0.1)
    return te_rank.map(table).fillna(0.02).clip(EPS, 1 - EPS).values


BENCHMARKS = {
    "bench_quali": ("quali_pos", [3, 4]),       # needs qualifying, so stages 3-4 only
    "bench_form": ("d_finish_last5", [1, 2, 3, 4]),
    "bench_constructor": ("c_champ_rank", [1, 2, 3, 4]),
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metrics(df: pd.DataFrame, prob_col="prob", y_col="y") -> dict:
    y, p = df[y_col].values.astype(int), df[prob_col].values
    # Top-3 per race: the three highest probabilities are the "predicted podium".
    top3 = df.groupby("race_id")[prob_col].rank(ascending=False, method="first") <= 3
    pred = (p >= 0.5).astype(int)
    out = dict(n_races=int(df["race_id"].nunique()), n_rows=int(len(df)),
               accuracy=accuracy_score(y, pred), precision_=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
               brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
               roc_auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
               top3_hit_rate=float(y[top3.values].mean()))
    return out


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

def stage_frame(df: pd.DataFrame, stage: int) -> pd.DataFrame:
    """Rows usable for a stage: confirmed (final) results, with qualifying data for stages 3-4.

    Provisional results still feed the form features of later races, but are not
    used as training labels until they are confirmed.
    """
    d = df[(df["is_target"] == 0) & df["y_podium"].notna() & (df["result_status"] == "final")].copy()
    if stage >= 3:
        d = d[d["quali_pos"].notna()]
    return d


def backtest(df: pd.DataFrame, stage: int, version: str, first_test: int = config.FIRST_TEST_SEASON,
             tune: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = features.STAGE_FEATURES[stage]
    data = stage_frame(df, stage)
    seasons = sorted(data["season"].unique())
    preds = []
    for season in [s for s in seasons if s >= first_test]:
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        if train["season"].nunique() < 3 or test.empty:
            continue
        fitted, val_ll = {}, {}
        for algo in ALGORITHMS:
            m, ll = fit_algo(train, feats, algo, tune=tune)
            fitted[algo], val_ll[algo] = m, ll
            preds.append(_pred_frame(test, stage, algo, version,
                                     normalise_per_race(sigmoid(m.calibrated_logit(test)), test["race_id"].values)))
        best2 = sorted(val_ll, key=val_ll.get)[:2]
        ens = np.mean([fitted[a].calibrated_logit(test) for a in best2], axis=0)
        preds.append(_pred_frame(test, stage, "ensemble", version,
                                 normalise_per_race(sigmoid(ens), test["race_id"].values)))
        for name, (col, stages) in BENCHMARKS.items():
            if stage in stages:
                p = _rank_lookup(train, test, col)
                preds.append(_pred_frame(test, stage, name, version, normalise_per_race(p, test["race_id"].values)))
        log.info("stage %d season %d done (val logloss %s)", stage, season,
                 {k: round(v, 4) for k, v in val_ll.items()})
    pred_df = pd.concat(preds, ignore_index=True)
    rows = []
    for (algo, season), g in pred_df.groupby(["algorithm", "season"]):
        rows.append(dict(model_version=version, stage=stage, algorithm=algo, season=str(season),
                         **metrics(g.rename(columns={"prob_podium": "prob", "actual_podium": "y"}))))
    for algo, g in pred_df.groupby("algorithm"):
        rows.append(dict(model_version=version, stage=stage, algorithm=algo, season="all",
                         **metrics(g.rename(columns={"prob_podium": "prob", "actual_podium": "y"}))))
    return pred_df, pd.DataFrame(rows)


def _pred_frame(test, stage, algo, version, prob) -> pd.DataFrame:
    return pd.DataFrame(dict(model_version=version, stage=stage, algorithm=algo, season=test["season"].values,
                             race_id=test["race_id"].values, driver_id=test["driver_id"].values,
                             constructor_id=test["constructor_id"].values, prob_podium=prob,
                             actual_podium=test["y_podium"].values.astype(int),
                             actual_position=test["position"].values))


def select_algorithm(metric_df: pd.DataFrame) -> str:
    """Lowest mean per-season log loss among the ML candidates (benchmarks are excluded)."""
    m = metric_df[(metric_df["season"] != "all") & ~metric_df["algorithm"].str.startswith("bench")]
    return m.groupby("algorithm")["log_loss"].mean().idxmin()


def train_final(df: pd.DataFrame, stage: int, algo: str, version: str) -> StageModel:
    feats = features.STAGE_FEATURES[stage]
    data = stage_frame(df, stage)
    if algo == "ensemble":
        fitted = {a: fit_algo(data, feats, a) for a in ALGORITHMS}
        best2 = sorted(fitted, key=lambda a: fitted[a][1])[:2]
        members = [fitted[a][0] for a in best2]
    else:
        members = [fit_algo(data, feats, algo)[0]]
    bg = data[feats].sample(min(100, len(data)), random_state=1)
    return StageModel(stage=stage, algo=algo, members=members, features=feats, version=version,
                      trained_at=db.utcnow(), n_train=len(data),
                      data_cutoff_race=str(data["race_id"].max()), background=bg)


def new_version(cutoff_race: str) -> str:
    return f"v{datetime.now(timezone.utc):%Y%m%d.%H%M}-{cutoff_race}"


def save(models: dict[int, StageModel], version: str) -> str:
    path = config.MODEL_DIR / f"apex_{version}.joblib"
    joblib.dump(models, path, compress=3)
    (config.MODEL_DIR / "active.json").write_text(json.dumps({"version": version, "file": path.name}, indent=2))
    # Keep the three most recent model files on disk (older versions remain in git history and the registry).
    files = sorted(config.MODEL_DIR.glob("apex_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[3:]:
        old.unlink()
    return path.name


def load_active() -> dict[int, StageModel] | None:
    pointer = config.MODEL_DIR / "active.json"
    if not pointer.exists():
        return None
    info = json.loads(pointer.read_text())
    path = config.MODEL_DIR / info["file"]
    if not path.exists():
        return None
    return joblib.load(path)
