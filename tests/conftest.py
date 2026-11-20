"""Test fixtures: a small synthetic database so tests run offline and fast.

The synthetic data is only used by the tests. It is generated from a simple
"car strength + noise" process so models have a real signal to learn.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import config, db  # noqa: E402

SEASONS = [2019, 2020, 2021, 2022, 2023]
N_ROUNDS = 8
TEAMS = ["red", "blue", "green", "gold", "grey"]
CIRCUITS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _build(path: Path, upcoming: bool = True):
    rng = np.random.default_rng(0)
    db.init_db(path)
    drivers = [f"d{i}" for i in range(10)]
    team_of = {d: TEAMS[i // 2] for i, d in enumerate(drivers)}
    strength = {t: s for t, s in zip(TEAMS, [2.0, 1.4, 0.8, 0.3, 0.0])}
    skill = {d: rng.normal(0, 0.3) for d in drivers}
    with db.session(path) as conn:
        db.upsert(conn, "circuits", [dict(circuit_id=c, name=c.title(), locality="X", country="Y", lat=0.0, lon=0.0,
                                          source="test", retrieved_at="2024-01-01") for c in CIRCUITS])
        db.upsert(conn, "drivers", [dict(driver_id=d, code=d.upper(), given_name="Driver", family_name=d.upper(),
                                         source="test", retrieved_at="2024-01-01") for d in drivers])
        db.upsert(conn, "constructors", [dict(constructor_id=t, name=t.title(), source="test", retrieved_at="2024-01-01")
                                         for t in TEAMS])
        races, results, quali = [], [], []
        for s in SEASONS + ([2024] if upcoming else []):
            for r in range(1, N_ROUNDS + 1):
                rid = f"{s}_{r:02d}"
                scheduled = s == 2024
                races.append(dict(race_id=rid, season=s, round=r, race_name=f"{CIRCUITS[r-1].title()} GP",
                                  circuit_id=CIRCUITS[r - 1], race_start_utc=f"{s}-0{min(r, 9)}-1{r % 10}T13:00:00+00:00",
                                  quali_start_utc=f"{s}-0{min(r, 9)}-0{r % 10}T13:00:00+00:00", has_sprint=0,
                                  result_status="scheduled" if scheduled else "final", source="test", retrieved_at="x"))
                if scheduled:
                    continue
                q_perf = {d: strength[team_of[d]] + skill[d] + rng.normal(0, 0.4) for d in drivers}
                q_order = sorted(drivers, key=lambda d: -q_perf[d])
                for pos, d in enumerate(q_order, 1):
                    quali.append(dict(race_id=rid, driver_id=d, constructor_id=team_of[d], car_number=int(d[1:]) + 1,
                                      position=pos, q1_s=90 + pos * 0.1, q2_s=None, q3_s=None, source="test", retrieved_at="x"))
                r_perf = {d: q_perf[d] + rng.normal(0, 0.6) for d in drivers}
                dnf = {d: rng.random() < 0.08 for d in drivers}
                order = sorted([d for d in drivers if not dnf[d]], key=lambda d: -r_perf[d]) + [d for d in drivers if dnf[d]]
                for pos, d in enumerate(order, 1):
                    fin = not dnf[d]
                    results.append(dict(race_id=rid, driver_id=d, constructor_id=team_of[d], car_number=int(d[1:]) + 1,
                                        grid=q_order.index(d) + 1, position=pos if fin else None,
                                        position_text=str(pos) if fin else "R", points=float(max(0, 11 - pos)) if fin else 0.0,
                                        laps=50 if fin else 20, status="Finished" if fin else "Engine",
                                        source="test", retrieved_at="x"))
        db.upsert(conn, "races", races)
        db.upsert(conn, "results", results)
        db.upsert(conn, "qualifying", quali)


@pytest.fixture()
def synthetic_db(tmp_path, monkeypatch):
    path = tmp_path / "apex.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "models").mkdir()
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(config, "FIRST_TEST_SEASON", 2022)
    _build(path)
    return path
