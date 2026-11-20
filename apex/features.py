"""Feature engineering.

Builds one row per (race, driver) with features that only use information
available before that race (and, for later stages, before the relevant session).

Leakage control works by construction:
* every historical aggregate is computed on a series that is shifted by one
  race (`_prior_*` helpers), so a row never sees its own result;
* weekend information is split into stage blocks (practice, qualifying,
  pre-race) and each prediction stage only receives the blocks that exist at
  that point in a real race weekend (see STAGE_FEATURES).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db

# Teams that changed name but kept the same factory and staff. Used only so a
# team's form carries across a rename; displays still use the real entrant name.
TEAM_LINEAGE = {
    "force_india": "aston_martin", "racing_point": "aston_martin",
    "toro_rosso": "rb", "alphatauri": "rb",
    "alfa": "sauber", "audi": "sauber",
    "lotus_f1": "alpine", "renault": "alpine",
    "marussia": "manor",
}

INCIDENT_STATUSES = {"Accident", "Collision", "Collision damage", "Spun off", "Damage", "Puncture",
                     "Fatal accident", "Debris"}
NON_START = {"Did not start", "Withdrew", "Not classified", "Did not qualify", "Did not prequalify",
             "Disqualified", "Excluded"}

# Human-readable labels and categories used by the explainability page.
FEATURE_INFO: dict[str, tuple[str, str]] = {
    "d_finish_ewm": ("Recent finishing positions (weighted)", "Recent form"),
    "d_finish_last5": ("Average finish, last 5 races", "Recent form"),
    "d_podium_ewm": ("Recent podium rate (weighted)", "Recent form"),
    "d_win_ewm": ("Recent win rate (weighted)", "Recent form"),
    "d_points_ewm": ("Recent points per race (weighted)", "Recent form"),
    "d_podium_career": ("Career podium rate", "Long-term ability"),
    "d_experience": ("Career starts", "Long-term ability"),
    "d_season_points": ("Championship points so far", "Championship position"),
    "d_champ_rank": ("Championship position before race", "Championship position"),
    "d_quali_ewm": ("Recent qualifying positions (weighted)", "Qualifying pace"),
    "d_quali_gap_ewm": ("Recent qualifying gap to pole (%)", "Qualifying pace"),
    "d_tm_quali_beat_ewm": ("Out-qualifies team-mate (rate)", "Team-mate comparison"),
    "d_tm_finish_diff_ewm": ("Finish gap to team-mate", "Team-mate comparison"),
    "d_gain_ewm": ("Places gained from grid (weighted)", "Racecraft"),
    "d_dnf_ewm": ("Driver DNF rate (weighted)", "Reliability"),
    "d_wet_delta": ("Wet-race performance vs dry", "Weather"),
    "d_race_pace_ewm": ("Race pace gap to fastest (%)", "Race pace"),
    "d_consistency_ewm": ("Lap-time consistency (lower = steadier)", "Race pace"),
    "d_deg_rel_ewm": ("Tyre degradation vs field", "Tyres"),
    "c_finish_ewm": ("Team recent finishing positions", "Constructor form"),
    "c_points_ewm": ("Team recent points per race", "Constructor form"),
    "c_podium_ewm": ("Team recent podium rate", "Constructor form"),
    "c_quali_ewm": ("Team recent qualifying positions", "Constructor form"),
    "c_mech_dnf_ewm": ("Team mechanical DNF rate", "Reliability"),
    "c_trend": ("Team development trend (points)", "Constructor form"),
    "c_season_points": ("Team championship points so far", "Championship position"),
    "c_champ_rank": ("Team championship position", "Championship position"),
    "c_race_pace_ewm": ("Team race pace gap (%)", "Race pace"),
    "c_driver_gap": ("Gap between the team's two drivers", "Team-mate comparison"),
    "dc_finish_mean": ("Driver's average finish at this circuit", "Circuit suitability"),
    "dc_podium_rate": ("Driver's podium rate at this circuit", "Circuit suitability"),
    "dc_starts": ("Driver's starts at this circuit", "Circuit suitability"),
    "cc_finish_mean": ("Team's average finish at this circuit", "Circuit suitability"),
    "circ_grid_corr": ("Circuit overtaking difficulty (grid-finish correlation)", "Circuit character"),
    "circ_pole_win": ("Circuit pole-to-win rate", "Circuit character"),
    "circ_dnf_rate": ("Circuit DNF rate", "Circuit character"),
    "circ_sc_rate": ("Circuit safety-car rate", "Circuit character"),
    "circ_stops": ("Circuit average pit stops", "Circuit character"),
    "circ_wet_rate": ("Circuit historical wet-race rate", "Weather"),
    "round_frac": ("Point in the season", "Context"),
    "fp_best_gap": ("Practice best-lap gap (%)", "Practice pace"),
    "fp_longrun_gap": ("Practice long-run gap (%)", "Practice pace"),
    "fp_deg_rel": ("Practice tyre degradation vs field", "Tyres"),
    "fp_team_best_gap": ("Team's practice best-lap gap (%)", "Practice pace"),
    "quali_pos": ("Qualifying position", "Qualifying result"),
    "quali_gap": ("Qualifying gap to pole (%)", "Qualifying result"),
    "quali_tm_delta": ("Qualifying positions vs team-mate", "Qualifying result"),
    "sprint_pos": ("Sprint finishing position", "Sprint"),
    "grid": ("Starting grid position", "Grid"),
    "grid_penalty": ("Grid places lost to penalties", "Grid"),
    "wx_precip_mm": ("Race-time rainfall (mm)", "Weather"),
    "wx_temp_c": ("Race-time air temperature", "Weather"),
    "wx_wind_kmh": ("Race-time wind speed", "Weather"),
}

BASE_FEATURES = [
    "d_finish_ewm", "d_finish_last5", "d_podium_ewm", "d_win_ewm", "d_points_ewm",
    "d_podium_career", "d_experience", "d_season_points", "d_champ_rank", "d_quali_ewm",
    "d_quali_gap_ewm", "d_tm_quali_beat_ewm", "d_tm_finish_diff_ewm", "d_gain_ewm", "d_dnf_ewm",
    "d_wet_delta", "d_race_pace_ewm", "d_consistency_ewm", "d_deg_rel_ewm",
    "c_finish_ewm", "c_points_ewm", "c_podium_ewm", "c_quali_ewm", "c_mech_dnf_ewm", "c_trend",
    "c_season_points", "c_champ_rank", "c_race_pace_ewm", "c_driver_gap",
    "dc_finish_mean", "dc_podium_rate", "dc_starts", "cc_finish_mean",
    "circ_grid_corr", "circ_pole_win", "circ_dnf_rate", "circ_sc_rate", "circ_stops", "circ_wet_rate",
    "round_frac",
]
PRACTICE_FEATURES = ["fp_best_gap", "fp_longrun_gap", "fp_deg_rel", "fp_team_best_gap"]
QUALI_FEATURES = ["quali_pos", "quali_gap", "quali_tm_delta", "sprint_pos_preq"]
RACE_FEATURES = ["grid", "grid_penalty", "sprint_pos", "wx_precip_mm", "wx_temp_c", "wx_wind_kmh"]
FEATURE_INFO["sprint_pos_preq"] = ("Sprint finishing position", "Sprint")

STAGE_FEATURES = {
    1: BASE_FEATURES,
    2: BASE_FEATURES + PRACTICE_FEATURES,
    3: BASE_FEATURES + PRACTICE_FEATURES + QUALI_FEATURES,
    4: BASE_FEATURES + PRACTICE_FEATURES + [f for f in QUALI_FEATURES if f != "sprint_pos_preq"] + RACE_FEATURES,
}
LABELS = {"win": 1, "top2": 2, "podium": 3, "top5": 5, "top10": 10}


def label(feature: str) -> str:
    return FEATURE_INFO.get(feature, (feature, "Other"))[0]


def category(feature: str) -> str:
    return FEATURE_INFO.get(feature, (feature, "Other"))[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prior_ewm(df: pd.DataFrame, by, col: str, halflife: float) -> pd.Series:
    """Exponentially weighted mean of earlier rows only (shifted by one)."""
    return df.groupby(by, sort=False)[col].transform(
        lambda s: s.shift(1).ewm(halflife=halflife, min_periods=1, ignore_na=True).mean())


def _prior_mean(df, by, col) -> pd.Series:
    return df.groupby(by, sort=False)[col].transform(lambda s: s.shift(1).expanding().mean())


def _prior_count(df, by, col) -> pd.Series:
    return df.groupby(by, sort=False)[col].transform(lambda s: s.shift(1).expanding().count())


def _load() -> dict[str, pd.DataFrame]:
    q = db.query
    return dict(
        races=q("SELECT * FROM races"),
        results=q("SELECT * FROM results"),
        quali=q("SELECT * FROM qualifying"),
        sprint=q("SELECT * FROM sprint_results"),
        stats=q("SELECT * FROM session_driver_stats"),
        sweather=q("SELECT * FROM session_weather"),
        events=q("SELECT * FROM race_events"),
        wx=q("SELECT * FROM race_weather"),
        pits=q("SELECT race_id, driver_id, COUNT(*) AS stops FROM pit_stops GROUP BY race_id, driver_id"),
    )


def entry_list(race_id: str, t: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Drivers expected to start a race that has not happened yet.

    Uses the qualifying classification once it exists; before that, the line-up
    from the most recent race with results.
    """
    t = t or _load()
    q = t["quali"][t["quali"]["race_id"] == race_id]
    if len(q):
        return q[["driver_id", "constructor_id"]].drop_duplicates("driver_id")
    prev = t["results"][t["results"]["race_id"] < race_id]
    if prev.empty:
        return pd.DataFrame(columns=["driver_id", "constructor_id"])
    last = prev["race_id"].max()
    return prev[prev["race_id"] == last][["driver_id", "constructor_id"]].drop_duplicates("driver_id")


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------

def build(target_race_id: str | None = None, t: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Return the full feature table.

    Rows for every race with results, plus (optionally) rows for `target_race_id`,
    an upcoming race without results, whose labels are NaN.
    """
    t = t or _load()
    races = t["races"].copy()
    races["race_date"] = pd.to_datetime(races["race_start_utc"], utc=True, format="ISO8601")
    res = t["results"].copy()
    if target_race_id is not None and target_race_id not in set(res["race_id"]):
        ent = entry_list(target_race_id, t).assign(race_id=target_race_id)
        res = pd.concat([res, ent], ignore_index=True)
    df = res.merge(races[["race_id", "season", "round", "circuit_id", "race_date", "race_start_utc", "result_status",
                          "quali_start_utc", "sprint_start_utc", "has_sprint"]], on="race_id", how="left")
    df = df.sort_values(["race_date", "race_id", "driver_id"]).reset_index(drop=True)
    df["team"] = df["constructor_id"].map(lambda c: TEAM_LINEAGE.get(c, c))
    has_result = df["status"].notna()
    df["is_target"] = (~has_result).astype(int)

    # ---------------- per-race outcomes (used only via prior aggregates) ----------
    field = df.groupby("race_id")["driver_id"].transform("count")
    df["field_size"] = field
    df["finish_score"] = np.where(df["position"].notna(), df["position"], field)
    df.loc[~has_result, "finish_score"] = np.nan
    status = df["status"].fillna("")
    finished = status.eq("Finished") | status.str.startswith("+") | status.str.contains("Lap")
    df["dnf"] = np.where(has_result, (~finished & df["position"].isna()).astype(float), np.nan)
    mech = has_result & ~finished & ~status.isin(INCIDENT_STATUSES) & ~status.isin(NON_START) & df["position"].isna()
    df["mech_dnf"] = np.where(has_result, mech.astype(float), np.nan)
    for name, k in LABELS.items():
        df[f"y_{name}"] = np.where(has_result, (df["position"] <= k).astype(float), np.nan)
    df["pts"] = df["points"]
    df["gain"] = np.where(has_result & (df["grid"] > 0), df["grid"] - df["finish_score"], np.nan)

    # qualifying data (all stages need history; stage 3+ uses the current race)
    q = t["quali"].copy()
    q["best_s"] = q[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    pole = q.groupby("race_id")["best_s"].transform("min")
    q["quali_gap"] = (q["best_s"] / pole - 1) * 100
    q = q.rename(columns={"position": "quali_pos"})
    df = df.merge(q[["race_id", "driver_id", "quali_pos", "quali_gap"]], on=["race_id", "driver_id"], how="left")

    # sprint
    sp = t["sprint"].rename(columns={"position": "sprint_pos_raw", "points": "sprint_points"})
    df = df.merge(sp[["race_id", "driver_id", "sprint_pos_raw", "sprint_points"]], on=["race_id", "driver_id"], how="left")
    df["sprint_points"] = pd.to_numeric(df["sprint_points"], errors="coerce").fillna(0.0)

    # team-mate comparisons within the same race
    grp = df.groupby(["race_id", "team"])
    for col in ("quali_pos", "finish_score"):
        tot = grp[col].transform("sum")
        cnt = grp[col].transform("count")
        df[f"tm_{col}"] = np.where(cnt == 2, tot - df[col], np.nan)
    df["tm_quali_beat"] = np.where(df["tm_quali_pos"].notna() & df["quali_pos"].notna(),
                                   (df["quali_pos"] < df["tm_quali_pos"]).astype(float), np.nan)
    df["tm_finish_diff"] = df["finish_score"] - df["tm_finish_score"]
    df["quali_tm_delta"] = df["quali_pos"] - df["tm_quali_pos"]

    # OpenF1 race pace / consistency / degradation (2023+)
    st = t["stats"]
    rs = st[st["session_name"] == "R"].copy()
    if len(rs):
        best_med = rs.groupby("race_id")["median_clean_lap_s"].transform("min")
        rs["race_pace"] = (rs["median_clean_lap_s"] / best_med - 1) * 100
        rs["consistency"] = rs["clean_lap_std_s"] / rs["median_clean_lap_s"] * 100
        rs["deg_rel"] = rs["deg_slope_s_per_lap"] - rs.groupby("race_id")["deg_slope_s_per_lap"].transform("median")
        df = df.merge(rs[["race_id", "driver_id", "race_pace", "consistency", "deg_rel"]],
                      on=["race_id", "driver_id"], how="left")
    else:
        df[["race_pace", "consistency", "deg_rel"]] = np.nan

    # wet-race label per race from any source that has it
    wet = pd.Series(np.nan, index=races["race_id"])
    sw = t["sweather"]
    sw = sw[sw["session_name"] == "R"].set_index("race_id")["rainfall_fraction"]
    tyres = st[st["session_name"] == "R"].groupby("race_id")["used_wet_tyres"].mean()
    wxo = t["wx"][t["wx"]["kind"] == "observed"].set_index("race_id")["precip_mm"]
    for rid in races["race_id"]:
        flags = []
        if rid in sw.index:
            flags.append(sw[rid] > 0.1)
        if rid in tyres.index:
            flags.append(tyres[rid] > 0.25)
        if rid in wxo.index and pd.notna(wxo[rid]):
            flags.append(wxo[rid] >= 1.0)
        if flags:
            wet[rid] = float(any(flags))
    df["wet"] = df["race_id"].map(wet)

    # ---------------- driver history ----------------
    by_d = "driver_id"
    df["d_finish_ewm"] = _prior_ewm(df, by_d, "finish_score", 4)
    df["d_finish_last5"] = df.groupby(by_d)["finish_score"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["d_podium_ewm"] = _prior_ewm(df, by_d, "y_podium", 6)
    df["d_win_ewm"] = _prior_ewm(df, by_d, "y_win", 6)
    df["d_points_ewm"] = _prior_ewm(df, by_d, "pts", 5)
    df["d_podium_career"] = _prior_mean(df, by_d, "y_podium")
    df["d_experience"] = _prior_count(df, by_d, "finish_score").fillna(0)
    df["d_quali_ewm"] = _prior_ewm(df, by_d, "quali_pos", 4)
    df["d_quali_gap_ewm"] = _prior_ewm(df, by_d, "quali_gap", 4)
    df["d_tm_quali_beat_ewm"] = _prior_ewm(df, by_d, "tm_quali_beat", 8)
    df["d_tm_finish_diff_ewm"] = _prior_ewm(df, by_d, "tm_finish_diff", 8)
    df["d_gain_ewm"] = _prior_ewm(df, by_d, "gain", 8)
    df["d_dnf_ewm"] = _prior_ewm(df, by_d, "dnf", 10)
    df["d_race_pace_ewm"] = _prior_ewm(df, by_d, "race_pace", 4)
    df["d_consistency_ewm"] = _prior_ewm(df, by_d, "consistency", 6)
    df["d_deg_rel_ewm"] = _prior_ewm(df, by_d, "deg_rel", 6)

    # wet vs dry performance, shrunk towards zero when there are few wet races
    rel = df["finish_score"] - df.groupby("race_id")["finish_score"].transform("mean")
    df["_rel"] = rel
    df["_rel_wet"] = np.where(df["wet"] == 1, rel, np.nan)
    df["_rel_dry"] = np.where(df["wet"] == 0, rel, np.nan)
    wet_mean = _prior_mean(df, by_d, "_rel_wet")
    dry_mean = _prior_mean(df, by_d, "_rel_dry")
    n_wet = _prior_count(df, by_d, "_rel_wet").fillna(0)
    df["d_wet_delta"] = ((wet_mean - dry_mean) * n_wet / (n_wet + 3)).fillna(0)

    # season points / championship rank before the race (results + sprint points)
    df["_season_pts"] = df["pts"].fillna(0) + df["sprint_points"].fillna(0)
    df.loc[~has_result, "_season_pts"] = 0
    df["d_season_points"] = df.groupby(["season", by_d])["_season_pts"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    df["d_champ_rank"] = df.groupby("race_id")["d_season_points"].rank(ascending=False, method="min")

    # ---------------- constructor history ----------------
    c = df.groupby(["race_id", "team"], sort=False).agg(
        race_date=("race_date", "first"), season=("season", "first"),
        c_finish=("finish_score", "mean"), c_points=("_season_pts", "sum"),
        c_podium=("y_podium", "max"), c_quali=("quali_pos", "mean"), c_mech=("mech_dnf", "mean"),
        c_pace=("race_pace", "mean"), c_has=("status", "count")).reset_index()
    c = c.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    for col in ("c_finish", "c_points", "c_podium", "c_quali", "c_mech", "c_pace"):
        c.loc[c["c_has"] == 0, col] = np.nan
    c["c_finish_ewm"] = _prior_ewm(c, "team", "c_finish", 4)
    c["c_points_ewm"] = _prior_ewm(c, "team", "c_points", 4)
    c["c_podium_ewm"] = _prior_ewm(c, "team", "c_podium", 6)
    c["c_quali_ewm"] = _prior_ewm(c, "team", "c_quali", 4)
    c["c_mech_dnf_ewm"] = _prior_ewm(c, "team", "c_mech", 10)
    c["c_race_pace_ewm"] = _prior_ewm(c, "team", "c_pace", 4)
    recent = c.groupby("team")["c_points"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    older = c.groupby("team")["c_points"].transform(lambda s: s.shift(4).rolling(6, min_periods=1).mean())
    c["c_trend"] = recent - older
    c["c_season_points"] = c.groupby(["season", "team"])["c_points"].transform(
        lambda s: s.fillna(0).shift(1).cumsum()).fillna(0)
    c["c_champ_rank"] = c.groupby("race_id")["c_season_points"].rank(ascending=False, method="min")
    df = df.merge(c[["race_id", "team", "c_finish_ewm", "c_points_ewm", "c_podium_ewm", "c_quali_ewm",
                     "c_mech_dnf_ewm", "c_race_pace_ewm", "c_trend", "c_season_points", "c_champ_rank"]],
                  on=["race_id", "team"], how="left")
    gap = df.groupby(["race_id", "team"])["d_finish_ewm"].transform(lambda s: s.max() - s.min() if len(s) == 2 else np.nan)
    df["c_driver_gap"] = gap

    # ---------------- circuit history ----------------
    df = df.sort_values(["race_date", "race_id", "driver_id"]).reset_index(drop=True)
    dc = [by_d, "circuit_id"]
    df["dc_finish_mean"] = _prior_mean(df, dc, "finish_score")
    starts = _prior_count(df, dc, "finish_score").fillna(0)
    pods = df.groupby(dc)["y_podium"].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
    # Shrink towards the driver's career podium rate when the circuit sample is small.
    prior = df["d_podium_career"].fillna(0)
    df["dc_podium_rate"] = (pods + 2 * prior) / (starts + 2)
    df["dc_starts"] = starts
    # Team-at-circuit history must be aggregated per race first: grouping driver rows
    # directly would let one car's result leak into its team-mate's row for the same race.
    tc = df.groupby(["race_id", "team"], sort=False).agg(circuit_id=("circuit_id", "first"), race_date=("race_date", "first"),
                                                         tc_finish=("finish_score", "mean")).reset_index()
    tc = tc.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    tc["cc_finish_mean"] = _prior_mean(tc, ["team", "circuit_id"], "tc_finish")
    df = df.merge(tc[["race_id", "team", "cc_finish_mean"]], on=["race_id", "team"], how="left")

    per_race = df[has_result.values].groupby("race_id").apply(_race_character, include_groups=False)
    per_race = per_race.join(t["events"].set_index("race_id")[["safety_cars", "virtual_safety_cars", "red_flags"]])
    for col in ("safety_cars", "virtual_safety_cars", "red_flags"):
        per_race[col] = pd.to_numeric(per_race[col], errors="coerce")
    per_race["stops"] = t["pits"].groupby("race_id")["stops"].mean()
    per_race["sc"] = (per_race["safety_cars"].fillna(0) + per_race["red_flags"].fillna(0) > 0).astype(float)
    per_race.loc[per_race["safety_cars"].isna(), "sc"] = np.nan
    per_race = per_race.join(wet.rename("wet_flag"))
    rc = races[["race_id", "circuit_id", "race_date"]].merge(per_race, left_on="race_id", right_index=True, how="left")
    rc = rc[rc["race_id"].isin(df["race_id"])].sort_values("race_date")
    for col, out, k, glob in (("grid_corr", "circ_grid_corr", 3, None), ("pole_win", "circ_pole_win", 3, None),
                              ("dnf_rate", "circ_dnf_rate", 3, None), ("sc", "circ_sc_rate", 2, None),
                              ("stops", "circ_stops", 2, None), ("wet_flag", "circ_wet_rate", 3, None)):
        s_sum = rc.groupby("circuit_id")[col].transform(lambda s: s.shift(1).expanding().sum())
        s_cnt = rc.groupby("circuit_id")[col].transform(lambda s: s.shift(1).expanding().count())
        global_prior = rc[col].shift(1).expanding().mean()  # all circuits, earlier races only
        rc[out] = (s_sum.fillna(0) + k * global_prior) / (s_cnt.fillna(0) + k)
    df = df.merge(rc[["race_id", "circ_grid_corr", "circ_pole_win", "circ_dnf_rate", "circ_sc_rate",
                      "circ_stops", "circ_wet_rate"]], on="race_id", how="left")
    max_round = races.groupby("season")["round"].transform("max")
    races["round_frac"] = races["round"] / max_round
    df = df.merge(races[["race_id", "round_frac"]], on="race_id", how="left")

    # ---------------- weekend blocks ----------------
    fp = st[st["session_name"].isin(["FP1", "FP2", "FP3"])].copy()
    if len(fp):
        fp["best_gap"] = (fp["best_lap_s"] / fp.groupby(["race_id", "session_name"])["best_lap_s"].transform("min") - 1) * 100
        fp["lr_gap"] = (fp["long_run_pace_s"] / fp.groupby(["race_id", "session_name"])["long_run_pace_s"].transform("min") - 1) * 100
        fp["deg_rel"] = fp["deg_slope_s_per_lap"] - fp.groupby(["race_id", "session_name"])["deg_slope_s_per_lap"].transform("median")
        agg = fp.groupby(["race_id", "driver_id"]).agg(fp_best_gap=("best_gap", "min"),
                                                        fp_longrun_gap=("lr_gap", "min"),
                                                        fp_deg_rel=("deg_rel", "mean")).reset_index()
        df = df.merge(agg, on=["race_id", "driver_id"], how="left")
        df["fp_team_best_gap"] = df.groupby(["race_id", "team"])["fp_best_gap"].transform("min")
    else:
        df[PRACTICE_FEATURES] = np.nan

    # sprint before qualifying (2024+ format) is usable at the post-qualifying stage
    sprint_first = pd.to_datetime(df["sprint_start_utc"], utc=True, format="ISO8601") < pd.to_datetime(df["quali_start_utc"], utc=True, format="ISO8601")
    df["sprint_pos"] = df["sprint_pos_raw"]
    df["sprint_pos_preq"] = np.where(sprint_first, df["sprint_pos_raw"], np.nan)

    # grid: 0 in Jolpica means pit-lane start
    df["grid_actual"] = df["grid"]
    df["grid"] = df["grid"].where(df["grid"] != 0, df["field_size"])
    df["grid_penalty"] = (df["grid"] - df["quali_pos"]).clip(lower=0)

    wx = t["wx"]
    obs = wx[wx["kind"] == "observed"].set_index("race_id")
    fc = wx[wx["kind"] == "forecast"].set_index("race_id")
    for src, dst in (("precip_mm", "wx_precip_mm"), ("temp_c", "wx_temp_c"), ("wind_kmh", "wx_wind_kmh")):
        df[dst] = df["race_id"].map(obs[src]) if len(obs) else np.nan
        if len(fc):
            df[dst] = df[dst].fillna(df["race_id"].map(fc[src]))

    df["sc_flag"] = df["race_id"].map(per_race["sc"])
    df["race_pace_actual"] = df["race_pace"]
    keep = (["race_id", "season", "round", "circuit_id", "race_date", "driver_id", "constructor_id", "team",
             "is_target", "result_status", "position", "grid_actual", "status", "wet", "sc_flag", "dnf", "race_pace_actual"]
            + [f"y_{k}" for k in LABELS] + sorted(set(sum(STAGE_FEATURES.values(), []))))
    out = df[keep].copy()
    num = sorted(set(sum(STAGE_FEATURES.values(), []))) + [f"y_{k}" for k in LABELS] + [
        "position", "wet", "sc_flag", "dnf", "race_pace_actual", "grid_actual"]
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").astype(float)
    return out


def _race_character(g: pd.DataFrame) -> pd.Series:
    fin = g.dropna(subset=["position"])
    ok = fin[fin["grid"] > 0]
    corr = ok["grid"].corr(ok["position"], method="spearman") if len(ok) > 5 else np.nan
    pole = g[g["grid"] == 1]
    return pd.Series(dict(
        grid_corr=corr,
        pole_win=float((pole["position"] == 1).any()) if len(pole) else np.nan,
        dnf_rate=g["dnf"].mean(),
    ))
