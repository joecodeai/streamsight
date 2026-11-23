"""Per-camera event logic shared by the Python pipeline and the DeepStream app.

Both pipelines produce tracks differently (ByteTrack in Python, NvDCF inside
DeepStream), but everything after tracking is the same: identity matching,
zones, fall detection and turning state changes into events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from streamsight.analytics import FallDetector, ZoneMonitor
from streamsight.config import CameraConfig
from streamsight.events import TrackEvent, label_for
from streamsight.pipeline import metrics
from streamsight.reid.matcher import CrossCameraMatcher
from streamsight.tracking.byte_tracker import Track


@dataclass
class CameraScene:
    camera: CameraConfig
    publish: callable
    matcher: CrossCameraMatcher | None = None
    update_interval_s: float = 1.0
    fall_detection: bool = False
    zones: ZoneMonitor = field(init=False)
    falls: FallDetector = field(default_factory=FallDetector)
    _started: set[int] = field(default_factory=set)
    _last_update: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.zones = ZoneMonitor(self.camera.zones)

    def _event(self, kind: str, t: Track, **extra) -> TrackEvent:
        kpts = t.keypoints.round(1).tolist() if t.keypoints is not None and kind != "track_end" else None
        return TrackEvent(
            camera_id=self.camera.id,
            event_type=kind,
            local_track_id=t.track_id,
            global_id=t.global_id,
            label=label_for(t.class_id),
            confidence=round(float(t.score), 3),
            bbox=[round(float(v), 1) for v in t.box],
            keypoints=kpts,
            **extra,
        )

    def process(self, tracks: list[Track], ended: list[Track], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        cam = self.camera.id

        for t in tracks:
            if t.track_id not in self._started:
                self._started.add(t.track_id)
                self._last_update[t.track_id] = now
                self.publish(self._event("track_start", t))

            if self.matcher is not None:
                result = self.matcher.assign(cam, t)
                if result is not None:
                    metrics.REID_DECISIONS.labels("matched" if result.matched else "new").inc()
                    self.publish(
                        self._event(
                            "reid_match",
                            t,
                            attributes={
                                "matched": result.matched,
                                "similarity": round(result.score, 3),
                                "from_camera": result.from_camera,
                            },
                        )
                    )

            if self.fall_detection:
                if self.falls.update(t):
                    self.publish(self._event("fall_suspected", t))
                t.meta["fallen"] = self.falls.is_fallen(t.track_id)

            if now - self._last_update.get(t.track_id, now) >= self.update_interval_s:
                self._last_update[t.track_id] = now
                self.publish(self._event("track_update", t))

        for ze in self.zones.update(tracks):
            self.publish(self._event(ze.kind, ze.track, zone=ze.zone))

        for t in ended:
            for ze in self.zones.remove(t):
                self.publish(self._event(ze.kind, ze.track, zone=ze.zone))
            if t.track_id in self._started:
                self.publish(self._event("track_end", t, attributes={"frames": t.hits}))
            self._started.discard(t.track_id)
            self._last_update.pop(t.track_id, None)
            self.falls.remove(t)
            if self.matcher is not None:
                self.matcher.release(cam, t)

        metrics.ACTIVE_TRACKS.labels(cam).set(len(tracks))
        for zone, count in self.zones.occupancy.items():
            metrics.ZONE_OCCUPANCY.labels(cam, zone).set(count)
