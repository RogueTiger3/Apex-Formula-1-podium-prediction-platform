"""Monte Carlo race simulator.

Finishing orders are sampled from a Plackett-Luce model: each driver gets a
log-strength theta, and a race is simulated by adding Gumbel noise to theta and
sorting (this is exactly equivalent to drawing drivers one position at a time in
proportion to exp(theta)). Before the order is drawn each driver may retire
(DNF) with a probability estimated from their team's and their own record.

Linking the simulator to the ML model
-------------------------------------
theta is solved iteratively so that the simulated podium frequency of every
driver matches the calibrated podium probability from the ML model. Win, top-2,
top-5 and top-10 probabilities, the full position distribution and podium
combinations then all come from the same simulated races, so they are always
internally consistent (P(win) <= P(top 2) <= P(podium) <= ...).

Scenario parameters
-------------------
* How much rain and safety cars reshuffle the order (randomness multipliers),
  how much a driver's wet-weather record matters in the rain, and how a race
  pace deficit translates into strength are all estimated by maximum likelihood
  on historical races (see `fit_scenario_params`).
* Values the user sets by hand (for example "this team is 0.5% slower") are
  hypothetical assumptions and are labelled as such in the dashboard.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import islice
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

log = logging.getLogger("apex.simulate")

DEFAULT_PARAMS = dict(scale=1.0, r_wet=1.0, r_dry=1.0, r_sc=1.0, r_nosc=1.0, b_wet=0.0,
                      a_pace=1.5, dnf_mult_wet=1.0, n_wet_races=0, n_sc_races=0, n_pace_races=0,
                      n_races=0)


# ---------------------------------------------------------------------------
# Fitting scenario parameters from history
# ---------------------------------------------------------------------------

def _pl_negloglik(scale: float, races: list[tuple[np.ndarray, np.ndarray]], top_k: int = 10) -> float:
    """Negative Plackett-Luce log likelihood of observed top-k finishing orders."""
    total = 0.0
    for x, order in races:
        z = scale * x
        remaining = np.ones(len(z), dtype=bool)
        for idx in order[:top_k]:
            zr = z[remaining]
            m = zr.max()
            total += z[idx] - (m + np.log(np.exp(zr - m).sum()))
            remaining[idx] = False
    return -total


def _fit_scale(races, bounds=(0.05, 5.0)) -> float:
    if not races:
        return np.nan
    return float(minimize_scalar(lambda s: _pl_negloglik(s, races), bounds=bounds, method="bounded").x)


def _race_inputs(g: pd.DataFrame, score_col: str):
    g = g.reset_index(drop=True)
    fin = g[g["actual_position"].notna()].sort_values("actual_position")
    return g[score_col].values.astype(float), fin.index.values


def fit_scenario_params(bt: pd.DataFrame, feats: pd.DataFrame) -> dict:
    """Estimate simulator parameters from out-of-sample backtest predictions.

    bt: backtest predictions for one algorithm/stage (race_id, driver_id, prob_podium, actual_position)
    feats: feature table with race-level flags (wet, safety car) and race pace.
    """
    p = dict(DEFAULT_PARAMS)
    d = bt.merge(feats[["race_id", "driver_id", "wet", "d_wet_delta", "race_pace_actual", "sc_flag", "dnf"]],
                 on=["race_id", "driver_id"], how="left")
    d["x"] = np.log(np.clip(d["prob_podium"], 1e-4, 1 - 1e-4) / (1 - np.clip(d["prob_podium"], 1e-4, 1 - 1e-4)))
    groups = {rid: g for rid, g in d.groupby("race_id")}
    allr = [_race_inputs(g, "x") for g in groups.values()]
    p["scale"] = _fit_scale(allr)
    p["n_races"] = len(allr)

    wet_ids = [r for r, g in groups.items() if g["wet"].iloc[0] == 1]
    dry_ids = [r for r, g in groups.items() if g["wet"].iloc[0] == 0]
    p["n_wet_races"] = len(wet_ids)
    if len(wet_ids) >= 5 and len(dry_ids) >= 20:
        s_wet = _fit_scale([_race_inputs(groups[r], "x") for r in wet_ids])
        s_dry = _fit_scale([_race_inputs(groups[r], "x") for r in dry_ids])
        p["r_wet"], p["r_dry"] = s_wet / p["scale"], s_dry / p["scale"]
        # Wet-skill coefficient: how much a driver's wet-vs-dry record shifts strength in the rain.
        wet_races = [(groups[r]["x"].values * s_wet, groups[r]["d_wet_delta"].fillna(0).values,
                      _race_inputs(groups[r], "x")[1]) for r in wet_ids]

        def nll(b):
            return _pl_negloglik(1.0, [(x - b * w, o) for x, w, o in wet_races])
        # d_wet_delta is measured in finishing places (negative = better in the wet), so b > 0 helps.
        p["b_wet"] = float(minimize_scalar(nll, bounds=(-0.5, 0.5), method="bounded").x)
        dnf_w = d[d["race_id"].isin(wet_ids)]["dnf"].mean()
        dnf_d = d[d["race_id"].isin(dry_ids)]["dnf"].mean()
        if dnf_d and not np.isnan(dnf_w):
            p["dnf_mult_wet"] = float(dnf_w / dnf_d)

    sc_ids = [r for r, g in groups.items() if g["sc_flag"].iloc[0] == 1]
    nosc_ids = [r for r, g in groups.items() if g["sc_flag"].iloc[0] == 0]
    p["n_sc_races"] = len(sc_ids)
    if len(sc_ids) >= 5 and len(nosc_ids) >= 5:
        s_sc = _fit_scale([_race_inputs(groups[r], "x") for r in sc_ids])
        s_no = _fit_scale([_race_inputs(groups[r], "x") for r in nosc_ids])
        p["r_sc"], p["r_nosc"] = s_sc / p["scale"], s_no / p["scale"]

    # Pace conversion: strength lost per 1% of race pace, from races with lap data.
    pace = [(g.dropna(subset=["race_pace_actual"])) for g in groups.values()]
    pace = [g for g in pace if len(g) >= 10]
    p["n_pace_races"] = len(pace)
    if len(pace) >= 10:
        races = [(-g["race_pace_actual"].values, _race_inputs(g, "x")[1]) for g in
                 (g.reset_index(drop=True) for g in pace)]
        p["a_pace"] = _fit_scale(races, bounds=(0.05, 20.0))
    return p


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Conditions for a simulation run. None = use the estimated probability."""
    rain: bool | None = None
    safety_car: bool | None = None
    dnf_multiplier: float = 1.0
    # Hypothetical, user-chosen race pace changes in percent (positive = slower), by team or driver.
    team_pace_penalty: dict[str, float] = field(default_factory=dict)
    driver_pace_penalty: dict[str, float] = field(default_factory=dict)

    def is_baseline(self) -> bool:
        return (self.rain is None and self.safety_car is None and self.dnf_multiplier == 1.0
                and not any(self.team_pace_penalty.values()) and not any(self.driver_pace_penalty.values()))


@dataclass
class SimResult:
    drivers: list[str]
    n_sims: int
    position_probs: np.ndarray  # drivers x positions
    podium_combos: list[tuple[tuple[str, str, str], float]]
    dnf_rate: np.ndarray
    rain_rate: float
    sc_rate: float

    def summary(self) -> pd.DataFrame:
        cp = self.position_probs.cumsum(axis=1)
        n = len(self.drivers)
        col = lambda k: cp[:, min(k, n) - 1]
        df = pd.DataFrame(dict(driver_id=self.drivers, p_win=col(1), p_p2=self.position_probs[:, 1],
                               p_p3=self.position_probs[:, 2], p_top2=col(2), p_podium=col(3),
                               p_top5=col(5), p_top10=col(10), p_dnf=self.dnf_rate,
                               expected_position=(self.position_probs * np.arange(1, n + 1)).sum(axis=1)))
        df["podium_count"] = np.round(df["p_podium"] * self.n_sims).astype(int)
        return df.sort_values("p_podium", ascending=False).reset_index(drop=True)


def _run(theta, wet_delta, p_dnf, params, scenario: Scenario, p_rain, p_sc, n, rng, keep_orders=False):
    d = len(theta)
    rain = (np.full(n, scenario.rain) if scenario.rain is not None else rng.random(n) < p_rain).astype(bool)
    sc = (np.full(n, scenario.safety_car) if scenario.safety_car is not None else rng.random(n) < p_sc).astype(bool)
    scale = np.where(rain, params["r_wet"], params["r_dry"]) * np.where(sc, params["r_sc"], params["r_nosc"])
    perf = scale[:, None] * theta[None, :] - np.outer(rain, params["b_wet"] * wet_delta)
    perf = perf + rng.gumbel(size=(n, d))
    dnf_p = np.clip(p_dnf[None, :] * scenario.dnf_multiplier * np.where(rain, params["dnf_mult_wet"], 1.0)[:, None], 0, 0.95)
    dnf = rng.random((n, d)) < dnf_p
    perf = np.where(dnf, perf - 1e6, perf)
    order = np.argsort(-perf, axis=1)  # order[s, k] = driver index finishing k+1
    positions = np.empty_like(order)
    rows = np.arange(n)[:, None]
    positions[rows, order] = np.arange(d)[None, :]
    return positions, order, dnf, rain, sc


def simulate(entries: pd.DataFrame, params: dict | None = None, scenario: Scenario | None = None,
             n_sims: int = 10_000, seed: int = 42, calibrate_sims: int = 20_000, theta: np.ndarray | None = None,
             p_rain: float = 0.0, p_sc: float = 0.5) -> tuple[SimResult, np.ndarray]:
    """Simulate a race.

    entries needs: driver_id, team, p_podium (model probability), p_dnf, wet_delta.
    Returns the result and theta (so callers can reuse the calibrated strengths).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    scenario = scenario or Scenario()
    e = entries.reset_index(drop=True)
    wet_delta = e["wet_delta"].fillna(0).values.astype(float)
    p_dnf = e["p_dnf"].fillna(0.08).clip(0.005, 0.5).values.astype(float)
    if theta is None:
        theta = match_strengths(e["p_podium"].values, wet_delta, p_dnf, params, p_rain, p_sc,
                                n=calibrate_sims, seed=seed)
    th = theta.copy()
    # Hypothetical pace changes: percent slower -> strength via the fitted pace coefficient.
    pen = np.array([scenario.team_pace_penalty.get(t, 0.0) for t in e["team"]]) + \
        np.array([scenario.driver_pace_penalty.get(dv, 0.0) for dv in e["driver_id"]])
    th = th - params["a_pace"] * pen
    rng = np.random.default_rng(seed + 1)
    positions, order, dnf, rain, sc = _run(th, wet_delta, p_dnf, params, scenario, p_rain, p_sc, n_sims, rng)
    d = len(th)
    probs = np.zeros((d, d))
    for k in range(d):
        probs[:, k] = (positions == k).mean(axis=0)
    combos = Counter(map(tuple, order[:, :3]))
    names = e["driver_id"].tolist()
    top = [((names[a], names[b], names[c]), cnt / n_sims) for (a, b, c), cnt in islice(combos.most_common(), 10)]
    return SimResult(names, n_sims, probs, top, dnf.mean(axis=0), float(rain.mean()), float(sc.mean())), theta


def match_strengths(p_target, wet_delta, p_dnf, params, p_rain, p_sc, n=20_000, seed=42, iters=12) -> np.ndarray:
    """Solve for theta so simulated podium frequencies match the model's podium probabilities."""
    p_target = np.clip(np.asarray(p_target, float), 1e-4, 0.999)
    theta = params["scale"] * np.log(p_target / (1 - p_target))
    lt = np.log(p_target / (1 - p_target))
    for _ in range(iters):
        rng = np.random.default_rng(seed)  # common random numbers: smooth convergence
        positions, *_ = _run(theta, wet_delta, p_dnf, params, Scenario(), p_rain, p_sc, n, rng)
        f = np.clip((positions < 3).mean(axis=0), 0.5 / n, 1 - 0.5 / n)
        step = lt - np.log(f / (1 - f))
        theta = theta + 0.8 * np.clip(step, -2, 2)
        if np.abs(step).max() < 0.02:
            break
    return theta
