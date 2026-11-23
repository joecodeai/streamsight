import numpy as np

from streamsight.analytics import FallDetector, ZoneMonitor, point_in_polygon, torso_angle
from streamsight.config import ZoneConfig
from streamsight.tracking.byte_tracker import Track
from streamsight.tracking.kalman import xyxy_to_xyah


def track_at(tid, box, kpts=None):
    t = Track(tid, np.r_[xyxy_to_xyah(np.asarray(box, float)), np.zeros(4)], np.eye(8), 0.9, 0, 0)
    t.keypoints = kpts
    return t


def test_point_in_polygon():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    assert point_in_polygon(5, 5, sq)
    assert not point_in_polygon(15, 5, sq)


def test_zone_enter_exit_and_occupancy():
    zm = ZoneMonitor([ZoneConfig(name="door", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])])
    # foot point = bottom centre of the box
    ev = zm.update([track_at(1, [200, 0, 240, 80])])
    assert ev == [] and zm.occupancy["door"] == 0
    ev = zm.update([track_at(1, [20, 0, 60, 80])])
    assert [(e.zone, e.kind) for e in ev] == [("door", "zone_enter")]
    assert zm.occupancy["door"] == 1
    ev = zm.remove(track_at(1, [20, 0, 60, 80]))
    assert [e.kind for e in ev] == ["zone_exit"]
    assert zm.occupancy["door"] == 0


def _pose(upright: bool):
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 1.0
    if upright:
        k[5, :2], k[6, :2], k[11, :2], k[12, :2] = (40, 20), (60, 20), (42, 80), (58, 80)
    else:
        k[5, :2], k[6, :2], k[11, :2], k[12, :2] = (20, 40), (20, 60), (80, 42), (80, 58)
    return k


def test_torso_angle():
    assert torso_angle(_pose(True)) < 10
    assert torso_angle(_pose(False)) > 80


def test_fall_requires_sustained_horizontal_pose():
    fd = FallDetector(min_frames=3)
    lying = track_at(1, [0, 60, 100, 100], _pose(False))
    standing = track_at(1, [40, 0, 60, 100], _pose(True))
    assert not fd.update(standing)
    assert [fd.update(lying) for _ in range(4)] == [False, False, True, False]
    assert fd.is_fallen(1)
    fd.update(standing)
    assert not fd.is_fallen(1)
