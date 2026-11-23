import json
import time

import numpy as np

from streamsight.config import CameraConfig, ZoneConfig, load_config
from streamsight.events import TrackEvent
from streamsight.pipeline.publisher import EventPublisher
from streamsight.pipeline.scene import CameraScene
from streamsight.reid import CrossCameraMatcher, InMemoryGallery
from streamsight.tracking.byte_tracker import Track
from streamsight.tracking.kalman import xyxy_to_xyah

from .conftest import ROOT


def test_example_config_is_valid():
    cfg = load_config(ROOT / "configs" / "pipeline.yaml")
    assert len(cfg.cameras) == 3
    assert cfg.detector.classes == [0]


def test_publisher_writes_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    pub = EventPublisher(api_url=None, jsonl_path=str(path), batch_size=2, flush_interval_s=0.05).start()
    for i in range(5):
        pub.publish(TrackEvent(camera_id="cam1", event_type="track_start", local_track_id=i))
    pub.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[0])["camera_id"] == "cam1"
    assert pub.sent == 5


def test_publisher_drops_oldest_when_full():
    pub = EventPublisher(api_url=None, queue_size=3)  # not started: nothing drains the queue
    for i in range(5):
        pub.publish(TrackEvent(camera_id="c", event_type="track_update", local_track_id=i))
    assert pub.dropped == 2
    assert [e.local_track_id for e in pub._queue] == [2, 3, 4]


def _track(tid, box, emb):
    t = Track(
        tid, np.r_[xyxy_to_xyah(np.asarray(box, float)), np.zeros(4)], np.eye(8), 0.9, 0, 0, last_frame=9
    )
    t.embedding = np.asarray(emb, np.float32) / np.linalg.norm(emb)
    return t


def test_scene_emits_lifecycle_events():
    events: list[TrackEvent] = []
    cam = CameraConfig(
        id="cam1", uri="x", zones=[ZoneConfig(name="z", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])]
    )
    scene = CameraScene(cam, events.append, CrossCameraMatcher(InMemoryGallery()), update_interval_s=0.5)
    t = _track(1, [10, 0, 30, 50], [1, 0, 0])

    start = time.monotonic()
    scene.process([t], [], now=start)
    scene.process([t], [], now=start + 1)
    scene.process([], [t])
    kinds = [e.event_type for e in events]
    assert kinds[:3] == ["track_start", "reid_match", "zone_enter"]
    assert "track_update" in kinds
    assert kinds[-2:] == ["zone_exit", "track_end"]
    assert all(e.global_id == t.global_id for e in events[1:])
