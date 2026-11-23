"""Person appearance embeddings with an OSNet re-identification model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from streamsight.detection.onnx_backend import ort_providers, read_input_meta

# ImageNet statistics, which OSNet was trained with (RGB order).
_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def crop_patches(frame: np.ndarray, boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Crop xyxy boxes from a BGR frame and resize to ``size`` = (width, height).

    Returns an (N, 3, H, W) float32 batch, RGB, normalised with ImageNet statistics.
    """
    h, w = frame.shape[:2]
    out = np.empty((len(boxes), 3, size[1], size[0]), np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, max(x2, x1 + 1))), int(min(h, max(y2, y1 + 1)))
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            patch = np.zeros((size[1], size[0], 3), np.uint8)
        patch = cv2.resize(patch, size, interpolation=cv2.INTER_LINEAR)[..., ::-1].astype(np.float32)
        out[i] = ((patch - _MEAN) / _STD).transpose(2, 0, 1)
    return out


class ReIdEmbedder:
    """Batched OSNet inference. Handles both static and dynamic batch ONNX exports."""

    def __init__(self, model_path: str | Path, device: str = "cuda", max_batch: int = 32) -> None:
        import onnxruntime as ort

        meta = read_input_meta(model_path)
        _, _, h, w = meta.shape
        self.size = (int(w) if isinstance(w, int) else 128, int(h) if isinstance(h, int) else 256)
        self.fixed_batch = meta.fixed_batch
        self.max_batch = self.fixed_batch or max_batch
        profile = None if self.fixed_batch else (meta.name, (3, self.size[1], self.size[0]), self.max_batch)
        self.session = ort.InferenceSession(
            str(model_path), providers=ort_providers(device, fp16=True, profile=profile)
        )
        self.input_name = meta.name
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """L2-normalised (N, D) embeddings for the given boxes in one frame."""
        if len(boxes) == 0:
            return np.zeros((0, self.dim), np.float32)
        patches = crop_patches(frame, boxes, self.size)
        feats = []
        for i in range(0, len(patches), self.max_batch):
            chunk = patches[i : i + self.max_batch]
            n = len(chunk)
            if self.fixed_batch and n < self.fixed_batch:
                chunk = np.concatenate(
                    [chunk, np.zeros((self.fixed_batch - n, *chunk.shape[1:]), np.float32)]
                )
            feats.append(self.session.run(None, {self.input_name: chunk})[0][:n])
        emb = np.concatenate(feats).astype(np.float32)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
