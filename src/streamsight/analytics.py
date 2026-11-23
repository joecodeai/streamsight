"""Scene analytics on top of tracks: zone entry/exit and pose-based fall detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from streamsight.config import ZoneConfig
from streamsight.tracking.byte_tracker import Track

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Ray-casting test; ``polygon`` is (N, 2)."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class ZoneEvent:
    zone: str
    kind: str  # "zone_enter" | "zone_exit"
    track: Track


@dataclass
class ZoneMonitor:
    """Tracks which zone each track's foot point (bottom-centre of the box) is in."""

    zones: list[ZoneConfig]
    _polys: list[np.ndarray] = field(init=False)
    _inside: dict[int, set[str]] = field(default_factory=dict)
    occupancy: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._polys = [np.asarray(z.polygon, np.float32) for z in self.zones]
        self.occupancy = {z.name: 0 for z in self.zones}

    def update(self, tracks: list[Track]) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        for t in tracks:
            x1, _, x2, y2 = t.box
            fx, fy = (x1 + x2) / 2, y2
            now = {
                z.name for z, p in zip(self.zones, self._polys, strict=True) if point_in_polygon(fx, fy, p)
            }
            before = self._inside.get(t.track_id, set())
            events += [ZoneEvent(name, "zone_enter", t) for name in sorted(now - before)]
            events += [ZoneEvent(name, "zone_exit", t) for name in sorted(before - now)]
            self._inside[t.track_id] = now
        self._recount()
        return events

    def remove(self, track: Track) -> list[ZoneEvent]:
        """Track ended: emit exits for every zone it was still in."""
        left = self._inside.pop(track.track_id, set())
        self._recount()
        return [ZoneEvent(name, "zone_exit", track) for name in sorted(left)]

    def _recount(self) -> None:
        self.occupancy = {z.name: 0 for z in self.zones}
        for names in self._inside.values():
            for n in names:
                self.occupancy[n] += 1


def torso_angle(keypoints: np.ndarray, min_vis: float = 0.5) -> float | None:
    """Angle of the shoulder-to-hip line from vertical, in degrees. ``None`` if not visible."""
    k = keypoints
    if min(k[L_SHOULDER, 2], k[R_SHOULDER, 2], k[L_HIP, 2], k[R_HIP, 2]) < min_vis:
        return None
    shoulder = (k[L_SHOULDER, :2] + k[R_SHOULDER, :2]) / 2
    hip = (k[L_HIP, :2] + k[R_HIP, :2]) / 2
    dx, dy = shoulder - hip
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


@dataclass
class FallDetector:
    """Flags a track as fallen when the torso stays near horizontal for ``min_frames`` frames.

    A wide box alone is a weak signal (people bending, carrying things), so the torso
    angle from pose keypoints is required, with box aspect ratio as a second check.
    """

    angle_threshold: float = 60.0
    min_frames: int = 8
    _count: dict[int, int] = field(default_factory=dict)
    _flagged: set[int] = field(default_factory=set)

    def update(self, track: Track) -> bool:
        """Returns True exactly once, on the frame a fall is confirmed."""
        if track.keypoints is None:
            return False
        angle = torso_angle(track.keypoints)
        x1, y1, x2, y2 = track.box
        wide = (x2 - x1) > 0.8 * (y2 - y1)
        lying = angle is not None and angle >= self.angle_threshold and wide
        c = self._count.get(track.track_id, 0)
        c = c + 1 if lying else 0
        self._count[track.track_id] = c
        if lying and c >= self.min_frames and track.track_id not in self._flagged:
            self._flagged.add(track.track_id)
            return True
        if not lying:
            self._flagged.discard(track.track_id)
        return False

    def is_fallen(self, track_id: int) -> bool:
        return track_id in self._flagged

    def remove(self, track: Track) -> None:
        self._count.pop(track.track_id, None)
        self._flagged.discard(track.track_id)
