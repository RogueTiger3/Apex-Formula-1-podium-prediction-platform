import numpy as np
import pandas as pd

from apex import db, features


def test_build_has_labels_and_stage_features(synthetic_db):
    df = features.build()
    assert len(df) == 5 * 8 * 10
    for cols in features.STAGE_FEATURES.values():
        assert set(cols) <= set(df.columns)
    assert df["y_podium"].sum() == 5 * 8 * 3  # three podium finishers per race


def test_no_leakage_from_current_or_future_races(synthetic_db):
    """Features for a race must not change when that race's result and every later race are removed."""
    full = features.build()
    target = "2022_05"
    t = features._load()
    later = t["results"]["race_id"] >= target
    t["results"] = t["results"][~later]
    t["quali"] = t["quali"][t["quali"]["race_id"] <= target]  # the target keeps its own qualifying (stage 3 input)
    # The calendar itself stays: it is published before the season starts.
    truncated = features.build(target_race_id=target, t=t)
    cols = features.STAGE_FEATURES[3]
    a = full[full["race_id"] == target].set_index("driver_id")[cols].sort_index()
    b = truncated[truncated["race_id"] == target].set_index("driver_id")[cols].sort_index()
    # Practice/circuit columns that are entirely missing are fine; everything else must match exactly.
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-9)


def test_history_features_ignore_own_result(synthetic_db):
    df = features.build()
    first = df.sort_values("race_date").groupby("driver_id").head(1)
    assert first["d_finish_ewm"].isna().all()  # nothing known before a driver's first race
    assert (first["d_experience"] == 0).all()


def test_upcoming_race_rows(synthetic_db):
    df = features.build("2024_01")
    tgt = df[df["race_id"] == "2024_01"]
    assert len(tgt) == 10
    assert tgt["y_podium"].isna().all()
    assert tgt["d_finish_ewm"].notna().all()
