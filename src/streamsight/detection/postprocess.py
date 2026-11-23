"""Decode YOLOv8 / YOLO11 heads and run class-aware NMS in NumPy.

Output layout of an Ultralytics ONNX export is ``(B, C, N)``:

* detect: ``C = 4 + num_classes`` -> ``cx, cy, w, h, class scores...``
* pose:   ``C = 4 + 1 + K * 3``   -> ``cx, cy, w, h, person score, (x, y, v) * K``

Boxes are in network-input pixels. There is no separate objectness score.
"""

from __future__ import annotations

import numpy as np

DecodedFrame = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS. Returns kept indices sorted by descending score."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Class-aware NMS: offset boxes per class so different classes never suppress each other."""
    if boxes.size == 0:
        return np.zeros((0,), np.int64)
    offset = class_ids.astype(np.float32)[:, None] * (boxes.max() + 1.0)
    return nms(boxes + offset, scores, iou_threshold)


def decode_yolo(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    classes: list[int] | None = None,
    num_keypoints: int = 0,
    max_det: int = 300,
) -> list[DecodedFrame]:
    """Decode a raw ``(B, C, N)`` YOLO output into per-image (boxes, scores, class_ids, keypoints)."""
    if output.ndim != 3:
        raise ValueError(f"expected (B, C, N) output, got shape {output.shape}")
    preds = output.transpose(0, 2, 1)  # (B, N, C)
    kpt_dims = num_keypoints * 3
    num_classes = preds.shape[2] - 4 - kpt_dims
    if num_classes < 1:
        raise ValueError(f"output has {preds.shape[2]} channels: too few for {num_keypoints} keypoints")

    results: list[DecodedFrame] = []
    for p in preds:
        cls_scores = p[:, 4 : 4 + num_classes]
        class_ids = cls_scores.argmax(axis=1).astype(np.int32)
        scores = cls_scores[np.arange(len(p)), class_ids]
        mask = scores >= conf_threshold
        if classes is not None:
            mask &= np.isin(class_ids, classes)
        p, scores, class_ids = p[mask], scores[mask].astype(np.float32), class_ids[mask]

        boxes = xywh_to_xyxy(p[:, :4]).astype(np.float32)
        keep = batched_nms(boxes, scores, class_ids, iou_threshold)[:max_det]
        kpts = None
        if num_keypoints:
            kpts = p[keep, 4 + num_classes :].reshape(-1, num_keypoints, 3).astype(np.float32)
        results.append((boxes[keep], scores[keep], class_ids[keep], kpts))
    return results
