from datetime import datetime, timedelta, timezone

from apex import db, features, ingest, models
from apex.sources import jolpica


def _fake_source(monkeypatch, race_start: datetime, with_result: bool):
    iso = race_start.replace(microsecond=0).isoformat()

    def schedule(season, max_age):
        return ([dict(race_id=f"{season}_01", season=season, round=1, race_name="Test GP", circuit_id="alpha",
                      race_start_utc=iso, quali_start_utc=None, sprint_start_utc=None, fp1_start_utc=None,
                      has_sprint=0, source="jolpica", retrieved_at="x")],
                [dict(circuit_id="alpha", name="Alpha", locality="X", country="Y", lat=0.0, lon=0.0,
                      source="jolpica", retrieved_at="x")])

    def results(season, max_age):
        if not with_result:
            return [], [], []
        rows = [dict(race_id=f"{season}_01", driver_id=f"d{i}", constructor_id="red", car_number=i, grid=i,
                     position=i, position_text=str(i), points=0.0, laps=50, status="Finished", time_ms=None,
                     fastest_lap_rank=None, source="jolpica", retrieved_at="x") for i in range(1, 4)]
        return rows, [], []

    monkeypatch.setattr(jolpica, "schedule", schedule)
    monkeypatch.setattr(jolpica, "results", results)
    monkeypatch.setattr(jolpica, "qualifying", lambda s, m: ([], [], []))
    monkeypatch.setattr(jolpica, "sprints", lambda s, m: [])
    monkeypatch.setattr(jolpica, "pit_stops", lambda s, r, max_age: [])
    monkeypatch.setattr(jolpica, "current_standings", lambda max_age=0: [])


def test_new_race_detection_and_provisional_handling(synthetic_db, monkeypatch):
    now = datetime.now(timezone.utc)
    # Race finished 2 hours ago: result exists but is provisional, so it must not count as new final data.
    _fake_source(monkeypatch, now - timedelta(hours=2), with_result=True)
    s = ingest.IngestSummary()
    ingest.sync_jolpica(s, [2030], 2030)
    assert s.provisional_races == ["2030_01"] and not s.has_new_final_data
    assert db.scalar("SELECT result_status FROM races WHERE race_id='2030_01'") == "provisional"
    # Provisional results feed form features but are never training labels.
    df = features.build()
    assert "2030_01" not in set(models.stage_frame(df, 1)["race_id"])

    # Four days later the same result is confirmed: detected exactly once.
    _fake_source(monkeypatch, now - timedelta(days=4), with_result=True)
    s2 = ingest.IngestSummary()
    ingest.sync_jolpica(s2, [2030], 2030)
    assert s2.newly_final_races == ["2030_01"]
    s3 = ingest.IngestSummary()
    ingest.sync_jolpica(s3, [2030], 2030)
    assert s3.newly_final_races == []  # nothing new on a re-run
    assert db.scalar("SELECT COUNT(*) FROM results WHERE race_id='2030_01'") == 3  # no duplicates
