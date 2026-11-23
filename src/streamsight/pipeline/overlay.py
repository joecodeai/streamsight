"""Draw tracks, identities, keypoints and zones on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from streamsight.config import ZoneConfig
from streamsight.tracking.byte_tracker import Track

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]  # fmt: skip


def _color(key: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(key * 7919)
    return tuple(int(c) for c in rng.integers(64, 255, 3))


def draw(
    image: np.ndarray,
    tracks: list[Track],
    zones: list[ZoneConfig],
    occupancy: dict[str, int] | None = None,
    draw_keypoints: bool = True,
    fps: float | None = None,
) -> np.ndarray:
    out = image.copy()
    for z in zones:
        pts = np.asarray(z.polygon, np.int32)
        cv2.polylines(out, [pts], True, (0, 200, 255), 2)
        label = f"{z.name}: {occupancy.get(z.name, 0) if occupancy else 0}"
        cv2.putText(out, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        color = _color(t.global_id if t.global_id is not None else t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"#{t.track_id}" + (f" G{t.global_id % 100000}" if t.global_id is not None else "")
        if t.meta.get("fallen"):
            tag += " FALL"
        cv2.putText(out, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if draw_keypoints and t.keypoints is not None:
            k = t.keypoints
            for a, b in SKELETON:
                if k[a, 2] > 0.5 and k[b, 2] > 0.5:
                    cv2.line(
                        out, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), color, 1, cv2.LINE_AA
                    )
    if fps is not None:
        cv2.putText(
            out, f"{fps:.1f} FPS", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
    return out
