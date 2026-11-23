from __future__ import annotations

from streamsight.config import DetectorConfig
from streamsight.detection.base import Detector


def create_detector(cfg: DetectorConfig) -> Detector:
    common = dict(
        input_size=cfg.input_size,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        classes=cfg.classes,
        task=cfg.task,
        max_batch=cfg.max_batch,
    )
    if cfg.backend == "onnxruntime":
        from streamsight.detection.onnx_backend import OnnxRuntimeDetector

        return OnnxRuntimeDetector(cfg.model, device=cfg.device, fp16=cfg.fp16, **common)
    if cfg.backend == "openvino":
        from streamsight.detection.openvino_backend import OpenVinoDetector

        return OpenVinoDetector(cfg.model, device=cfg.device, **common)
    raise ValueError(f"unknown detector backend: {cfg.backend}")
