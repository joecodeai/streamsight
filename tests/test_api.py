from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from streamsight.api.main import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'test.db'}", api_key="secret")
    return TestClient(app)


def ev(cam, kind, tid, gid=None, **kw):
    return {
        "camera_id": cam,
        "event_type": kind,
        "local_track_id": tid,
        "global_id": gid,
        "confidence": 0.9,
        "bbox": [0, 0, 10, 20],
        "ts": kw.pop("ts", datetime.now(timezone.utc).isoformat()),
        **kw,
    }


def post(client, events, key="secret"):
    return client.post("/api/v1/events/batch", json={"events": events}, headers={"X-API-Key": key})


def test_ingest_requires_api_key(client):
    assert post(client, [ev("cam1", "track_start", 1)], key="wrong").status_code == 401
    assert post(client, [ev("cam1", "track_start", 1)]).status_code == 202


def test_ingest_validates_payload(client):
    bad = ev("cam1", "not_a_type", 1)
    assert post(client, [bad]).status_code == 422


def test_cross_camera_identity_and_timeline(client):
    t0 = datetime.now(timezone.utc) - timedelta(seconds=30)
    events = [
        ev("cam1", "track_start", 1, ts=t0.isoformat()),
        ev(
            "cam1",
            "reid_match",
            1,
            42,
            ts=(t0 + timedelta(seconds=1)).isoformat(),
            attributes={"matched": False, "similarity": 0.0},
        ),
        ev("cam1", "track_end", 1, 42, ts=(t0 + timedelta(seconds=10)).isoformat()),
        ev("cam2", "track_start", 5, ts=(t0 + timedelta(seconds=12)).isoformat()),
        ev(
            "cam2",
            "reid_match",
            5,
            42,
            ts=(t0 + timedelta(seconds=13)).isoformat(),
            attributes={"matched": True, "similarity": 0.81, "from_camera": "cam1"},
        ),
        ev("cam3", "reid_match", 9, 77),
    ]
    assert post(client, events).json() == {"accepted": 6}

    multi = client.get("/api/v1/identities", params={"min_cameras": 2}).json()
    assert [i["global_id"] for i in multi] == [42]
    assert multi[0]["cameras"] == ["cam1", "cam2"]

    timeline = client.get("/api/v1/identities/42/timeline").json()
    assert [e["camera_id"] for e in timeline] == ["cam1", "cam2"]
    assert timeline[0]["first_seen"].endswith("Z")
    assert client.get("/api/v1/identities/999/timeline").status_code == 404


def test_event_filters_and_summary(client):
    post(
        client,
        [
            ev("cam1", "track_start", 1),
            ev("cam1", "zone_enter", 1, zone="door"),
            ev("cam1", "track_start", 2),
            ev("cam1", "zone_enter", 2, zone="door"),
            ev("cam1", "zone_exit", 2, zone="door"),
            ev("cam2", "fall_suspected", 3),
        ],
    )
    zone_events = client.get("/api/v1/events", params={"event_type": "zone_enter"}).json()
    assert len(zone_events) == 2
    assert len(client.get("/api/v1/events", params={"camera_id": "cam2"}).json()) == 1

    s = client.get("/api/v1/stats/summary").json()
    assert s["cameras"] == 2 and s["cameras_online"] == 2
    assert s["zone_occupancy"] == {"cam1/door": 1}
    assert s["falls_last_hour"] == 1

    cams = {c["id"]: c for c in client.get("/api/v1/cameras").json()}
    assert cams["cam1"]["active_tracks"] == 2


def test_health_metrics_and_dashboard(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert "streamsight_api_events_ingested_total" in client.get("/metrics/").text
    assert "StreamSight" in client.get("/").text
