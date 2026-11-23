"""Backend-independent detector interface.

Each backend only implements :meth:`Detector._forward` (raw model output for a
preprocessed batch). Letterboxing, decoding, NMS and mapping boxes back to the
original frames are shared, so every backend returns identical results for the
same model.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from streamsight.detection.postprocess import decode_yolo
from streamsight.detection.preprocess import LetterboxInfo, letterbox_batch, scale_boxes, scale_keypoints


@dataclass
class Detections:
    """Detections for one frame, in original image pixel coordinates."""

    boxes: np.ndarray  # (N, 4) float32, x1 y1 x2 y2
    scores: np.ndarray  # (N,) float32
    class_ids: np.ndarray  # (N,) int32
    keypoints: np.ndarray | None = None  # (N, K, 3) float32: x, y, visibility

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @classmethod
    def empty(cls, with_keypoints: bool = False, num_keypoints: int = 17) -> Detections:
        kpts = np.zeros((0, num_keypoints, 3), np.float32) if with_keypoints else None
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), kpts)


@dataclass
class DetectorTiming:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    history: list[float] = field(default_factory=list)


class Detector(ABC):
    """YOLO-family detector (YOLOv8 / YOLO11 output layout, optional pose head)."""

    backend_name = "base"

    def __init__(
        self,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        classes: list[int] | None = None,
        task: str = "detect",
        max_batch: int = 16,
        num_keypoints: int = 17,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.task = task
        self.max_batch = max_batch
        self.num_keypoints = num_keypoints if task == "pose" else 0
        self.timing = DetectorTiming()

    @property
    @abstractmethod
    def fixed_batch(self) -> int | None:
        """Batch size baked into the model, or ``None`` if the batch axis is dynamic."""

    @abstractmethod
    def _forward(self, batch: np.ndarray) -> np.ndarray:
        """Run the network on a (B, 3, H, W) float32 batch and return its (B, C, N) output."""

    def warmup(self, runs: int = 3) -> None:
        dummy = np.zeros((self.fixed_batch or 1, 3, self.input_size, self.input_size), np.float32)
        for _ in range(runs):
            self._forward(dummy)

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        return self.detect(frames)

    def detect(self, frames: list[np.ndarray]) -> list[Detections]:
        results: list[Detections] = []
        step = self.fixed_batch or self.max_batch
        for i in range(0, len(frames), step):
            results.extend(self._detect_chunk(frames[i : i + step]))
        return results

    def _detect_chunk(self, frames: list[np.ndarray]) -> list[Detections]:
        t0 = time.perf_counter()
        batch, infos = letterbox_batch(frames, self.input_size)
        n = len(frames)
        if self.fixed_batch and n < self.fixed_batch:
            pad = np.zeros((self.fixed_batch - n, *batch.shape[1:]), batch.dtype)
            batch = np.concatenate([batch, pad], axis=0)
        t1 = time.perf_counter()
        raw = self._forward(batch)[:n]
        t2 = time.perf_counter()
        out = self._postprocess(raw, infos)
        t3 = time.perf_counter()
        self.timing.preprocess_ms = (t1 - t0) * 1000
        self.timing.inference_ms = (t2 - t1) * 1000
        self.timing.postprocess_ms = (t3 - t2) * 1000
        return out

    def _postprocess(self, raw: np.ndarray, infos: list[LetterboxInfo]) -> list[Detections]:
        decoded = decode_yolo(
            raw,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            classes=self.classes,
            num_keypoints=self.num_keypoints,
        )
        results = []
        for (boxes, scores, cls, kpts), info in zip(decoded, infos, strict=True):
            boxes = scale_boxes(boxes, info)
            if kpts is not None:
                kpts = scale_keypoints(kpts, info)
            results.append(Detections(boxes, scores, cls, kpts))
        return results
