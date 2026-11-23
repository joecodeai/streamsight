"""Letterbox resize: keep the aspect ratio, pad to a square network input."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize a BGR image into a ``size`` x ``size`` canvas, centred, without distortion."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, LetterboxInfo(scale, float(left), float(top), w, h)


def letterbox_batch(images: list[np.ndarray], size: int) -> tuple[np.ndarray, list[LetterboxInfo]]:
    """BGR uint8 frames -> (B, 3, size, size) float32 RGB tensor in [0, 1]."""
    canvases, infos = [], []
    for img in images:
        c, info = letterbox(img, size)
        canvases.append(c)
        infos.append(info)
    batch = np.stack(canvases)[..., ::-1]  # BGR -> RGB
    batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
    return batch, infos


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from network input space back to the original image."""
    if boxes.size == 0:
        return boxes.astype(np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_x) / info.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_y) / info.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, info.orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, info.orig_h)
    return out


def scale_keypoints(kpts: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    if kpts.size == 0:
        return kpts.astype(np.float32)
    out = kpts.astype(np.float32).copy()
    out[..., 0] = ((out[..., 0] - info.pad_x) / info.scale).clip(0, info.orig_w)
    out[..., 1] = ((out[..., 1] - info.pad_y) / info.scale).clip(0, info.orig_h)
    return out
