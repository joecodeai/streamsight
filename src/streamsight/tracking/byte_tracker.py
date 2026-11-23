"""Multi-object tracker following the ByteTrack association scheme.

Detections are split by confidence. High-confidence boxes are matched first to
all tracks; leftover tracks then get a second chance against low-confidence
boxes (occluded or blurred people often score low but are still real). Only
unmatched high-confidence boxes start new tracks.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box", ECCV 2022.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from streamsight.detection.base import Detections
from streamsight.detection.postprocess import box_iou
from streamsight.tracking.kalman import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    class_id: int
    start_frame: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    last_frame: int = 0
    keypoints: np.ndarray | None = None
    embedding: np.ndarray | None = None
    """Running (EMA-smoothed, L2-normalised) appearance embedding, set by the re-id stage."""
    global_id: int | None = None
    """Identity shared across cameras, assigned by the cross-camera matcher."""
    meta: dict = field(default_factory=dict)

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    @property
    def age(self) -> int:
        return self.last_frame - self.start_frame + 1

    def update_embedding(self, emb: np.ndarray, alpha: float) -> None:
        if self.embedding is None:
            self.embedding = emb
        else:
            mixed = alpha * self.embedding + (1 - alpha) * emb
            self.embedding = mixed / (np.linalg.norm(mixed) + 1e-12)


def _match(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment that rejects pairs whose cost exceeds ``threshold``."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= threshold]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    unmatched_r = [r for r in range(cost.shape[0]) if r not in matched_r]
    unmatched_c = [c for c in range(cost.shape[1]) if c not in matched_c]
    return matches, unmatched_r, unmatched_c


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.kf = KalmanFilter()
        self.tracks: list[Track] = []
        self.frame_id = 0
        self._next_id = 1
        self.removed: list[Track] = []
        """Tracks removed during the last ``update`` call (consumers emit track-end events)."""

    @classmethod
    def from_config(cls, cfg) -> ByteTracker:
        return cls(
            high_thresh=cfg.high_thresh,
            low_thresh=cfg.low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            min_hits=cfg.min_hits,
        )

    def _predict(self) -> None:
        if not self.tracks:
            return
        means = np.stack([t.mean for t in self.tracks])
        covs = np.stack([t.cov for t in self.tracks])
        means, covs = self.kf.multi_predict(means, covs)
        for t, m, c in zip(self.tracks, means, covs, strict=True):
            if t.state == TrackState.LOST:
                m[7] = 0.0  # stop height drift while the object is unseen
            t.mean, t.cov = m, c

    def _apply(self, track: Track, box: np.ndarray, score: float, cls: int, kpts: np.ndarray | None) -> None:
        track.mean, track.cov = self.kf.update(track.mean, track.cov, xyxy_to_xyah(box))
        track.score = float(score)
        track.class_id = int(cls)
        track.keypoints = kpts
        track.last_frame = self.frame_id
        track.hits += 1
        if track.state == TrackState.LOST or (
            track.state == TrackState.TENTATIVE and track.hits >= self.min_hits
        ):
            track.state = TrackState.CONFIRMED

    def update(self, dets: Detections) -> list[Track]:
        """Advance one frame. Returns confirmed tracks that were matched in this frame."""
        self.frame_id += 1
        self.removed = []
        self._predict()

        scores = dets.scores
        high = scores >= self.high_thresh
        low = (scores >= self.low_thresh) & ~high
        idx_high, idx_low = np.flatnonzero(high), np.flatnonzero(low)

        active = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]

        def kp(i: int) -> np.ndarray | None:
            return None if dets.keypoints is None else dets.keypoints[i]

        # 1) high-confidence detections vs confirmed + lost tracks
        track_boxes = np.array([t.box for t in active]).reshape(-1, 4)
        cost = 1.0 - box_iou(track_boxes, dets.boxes[idx_high])
        matches, um_tracks, um_high = _match(cost, self.match_thresh)
        for ti, di in matches:
            d = idx_high[di]
            self._apply(active[ti], dets.boxes[d], scores[d], dets.class_ids[d], kp(d))

        # 2) low-confidence detections vs tracks still unmatched (only those seen last frame)
        remaining = [active[i] for i in um_tracks if active[i].state == TrackState.CONFIRMED]
        track_boxes = np.array([t.box for t in remaining]).reshape(-1, 4)
        cost = 1.0 - box_iou(track_boxes, dets.boxes[idx_low])
        matches, um_rem, _ = _match(cost, 0.5)
        for ti, di in matches:
            d = idx_low[di]
            self._apply(remaining[ti], dets.boxes[d], scores[d], dets.class_ids[d], kp(d))
        for i in um_rem:
            remaining[i].state = TrackState.LOST

        # 3) leftover high-confidence detections vs tentative tracks
        left_high = idx_high[um_high]
        track_boxes = np.array([t.box for t in tentative]).reshape(-1, 4)
        cost = 1.0 - box_iou(track_boxes, dets.boxes[left_high])
        matches, um_tent, um_new = _match(cost, 0.7)
        for ti, di in matches:
            d = left_high[di]
            self._apply(tentative[ti], dets.boxes[d], scores[d], dets.class_ids[d], kp(d))
        for i in um_tent:
            tentative[i].state = TrackState.REMOVED

        # 4) start new tracks
        for di in um_new:
            d = left_high[di]
            if scores[d] < self.new_track_thresh:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(dets.boxes[d]))
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    mean=mean,
                    cov=cov,
                    score=float(scores[d]),
                    class_id=int(dets.class_ids[d]),
                    start_frame=self.frame_id,
                    last_frame=self.frame_id,
                    keypoints=kp(d),
                    state=TrackState.CONFIRMED if self.min_hits <= 1 else TrackState.TENTATIVE,
                )
            )
            self._next_id += 1

        # 5) expire lost tracks
        for t in self.tracks:
            if t.state == TrackState.LOST and self.frame_id - t.last_frame > self.track_buffer:
                t.state = TrackState.REMOVED
        self.removed = [t for t in self.tracks if t.state == TrackState.REMOVED and t.hits >= self.min_hits]
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        return [t for t in self.tracks if t.state == TrackState.CONFIRMED and t.last_frame == self.frame_id]
