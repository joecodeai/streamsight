"""Assigns camera-independent identities (global ids) to per-camera tracks.

Rules:
* A track is matched once it is ``min_track_frames`` old, so its embedding is averaged
  over several frames instead of one possibly blurred crop.
* Identities that are currently visible in the *same* camera are excluded from the
  search: one person cannot be two tracks in one view at the same time.
* Below ``match_threshold`` the track gets a new identity.
* While a track lives, its embedding is added to the gallery periodically, so the
  gallery keeps several views of each person (front, side, different lighting).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from streamsight.reid.gallery import Gallery
from streamsight.tracking.byte_tracker import Track


@dataclass
class MatchResult:
    global_id: int
    matched: bool  # True if linked to an existing identity, False if newly created
    score: float = 0.0
    from_camera: str | None = None


@dataclass
class CrossCameraMatcher:
    gallery: Gallery
    match_threshold: float = 0.65
    min_track_frames: int = 5
    add_every: int = 15
    _active: dict[str, dict[int, int]] = field(default_factory=dict)
    """camera_id -> {local track id -> global id} for tracks alive right now."""
    _since_add: dict[tuple[str, int], int] = field(default_factory=dict)

    def assign(self, camera_id: str, track: Track) -> MatchResult | None:
        """Give ``track`` a global id if it is ready. Returns the result only when a new decision is made."""
        cam = self._active.setdefault(camera_id, {})
        key = (camera_id, track.track_id)

        if track.global_id is not None:
            self._since_add[key] = self._since_add.get(key, 0) + 1
            if track.embedding is not None and self._since_add[key] >= self.add_every:
                self.gallery.add(track.global_id, camera_id, track.embedding)
                self._since_add[key] = 0
            return None

        if track.embedding is None or track.age < self.min_track_frames:
            return None

        busy = set(cam.values())
        hits = self.gallery.search(track.embedding, top_k=1, exclude_ids=busy)
        if hits and hits[0].score >= self.match_threshold:
            result = MatchResult(hits[0].global_id, True, hits[0].score, hits[0].camera_id)
        else:
            result = MatchResult(self.gallery.next_global_id(), False, hits[0].score if hits else 0.0)

        track.global_id = result.global_id
        cam[track.track_id] = result.global_id
        self.gallery.add(result.global_id, camera_id, track.embedding)
        self._since_add[key] = 0
        return result

    def release(self, camera_id: str, track: Track) -> None:
        """Call when a track ends so its identity can be matched again in this camera."""
        self._active.get(camera_id, {}).pop(track.track_id, None)
        self._since_add.pop((camera_id, track.track_id), None)
