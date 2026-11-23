import numpy as np

from streamsight.detection.base import Detections
from streamsight.tracking import ByteTracker, TrackState


def dets(*boxes_scores):
    boxes = np.array([b for b, _ in boxes_scores], np.float32).reshape(-1, 4)
    scores = np.array([s for _, s in boxes_scores], np.float32)
    return Detections(boxes, scores, np.zeros(len(scores), np.int32))


def test_track_keeps_id_while_moving():
    tracker = ByteTracker(min_hits=1)
    ids = set()
    for i in range(20):
        x = 100 + i * 5
        out = tracker.update(dets(([x, 100, x + 40, 200], 0.9)))
        assert len(out) == 1
        ids.add(out[0].track_id)
    assert ids == {1}


def test_two_objects_get_two_ids():
    tracker = ByteTracker(min_hits=1)
    for i in range(10):
        out = tracker.update(dets(([100 + i, 100, 140 + i, 200], 0.9), ([400 - i, 100, 440 - i, 200], 0.9)))
    assert sorted(t.track_id for t in out) == [1, 2]


def test_min_hits_hides_single_frame_detections():
    tracker = ByteTracker(min_hits=3)
    assert tracker.update(dets(([0, 0, 50, 100], 0.9))) == []
    assert tracker.update(dets(([1, 0, 51, 100], 0.9))) == []
    out = tracker.update(dets(([2, 0, 52, 100], 0.9)))
    assert len(out) == 1 and out[0].state == TrackState.CONFIRMED


def test_low_score_detection_keeps_track_alive():
    """ByteTrack's second association: an occluded (low-score) box still continues the track."""
    tracker = ByteTracker(min_hits=1, high_thresh=0.5, low_thresh=0.1)
    for i in range(5):
        tracker.update(dets(([100 + i * 2, 100, 140 + i * 2, 200], 0.9)))
    out = tracker.update(dets(([110, 100, 150, 200], 0.2)))
    assert [t.track_id for t in out] == [1]


def test_lost_track_recovers_then_expires():
    tracker = ByteTracker(min_hits=1, track_buffer=5)
    for i in range(5):
        tracker.update(dets(([100 + i, 100, 140 + i, 200], 0.9)))
    for _ in range(3):  # short occlusion
        assert tracker.update(dets()) == []
    out = tracker.update(dets(([108, 100, 148, 200], 0.9)))
    assert [t.track_id for t in out] == [1]

    for _ in range(7):  # long absence -> removed
        tracker.update(dets())
    assert [t.track_id for t in tracker.removed] == [1] or not tracker.tracks
    assert tracker.tracks == []


def test_embedding_ema_stays_normalised():
    tracker = ByteTracker(min_hits=1)
    t = tracker.update(dets(([0, 0, 10, 20], 0.9)))[0]
    a = np.array([1.0, 0.0], np.float32)
    b = np.array([0.0, 1.0], np.float32)
    t.update_embedding(a, 0.9)
    t.update_embedding(b, 0.9)
    assert np.linalg.norm(t.embedding) == np.float32(1.0) or abs(np.linalg.norm(t.embedding) - 1) < 1e-6
    assert t.embedding[0] > t.embedding[1]
