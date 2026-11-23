"""Typed configuration loaded from YAML.

Every pipeline component reads its settings from one :class:`AppConfig`, so a
deployment is fully described by a single file (see ``configs/pipeline.yaml``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ZoneConfig(BaseModel):
    """A named polygon in image pixel coordinates. Tracks entering or leaving it emit events."""

    name: str
    polygon: list[tuple[float, float]] = Field(min_length=3)


class CameraConfig(BaseModel):
    id: str
    name: str = ""
    uri: str
    loop: bool = True
    """For file sources: restart at the end so a file behaves like a live camera."""
    realtime: bool = True
    """For file sources: pace reads to the file's FPS instead of reading as fast as possible."""
    zones: list[ZoneConfig] = Field(default_factory=list)


class DetectorConfig(BaseModel):
    backend: Literal["onnxruntime", "openvino"] = "onnxruntime"
    model: str = "models/yolo11n.onnx"
    task: Literal["detect", "pose"] = "detect"
    device: Literal["cpu", "cuda", "tensorrt"] = "cuda"
    """ONNX Runtime execution provider (OpenVINO maps cpu/cuda to CPU/GPU)."""
    fp16: bool = True
    """Only used by the TensorRT execution provider."""
    input_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    classes: list[int] | None = Field(default_factory=lambda: [0])
    """COCO class ids to keep. ``None`` keeps all 80 classes. Default: person only."""
    max_batch: int = 16


class TrackerConfig(BaseModel):
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    """Maximum IoU distance (1 - IoU) accepted for a match."""
    track_buffer: int = 30
    """Frames a lost track is kept before it is removed."""
    min_hits: int = 3
    """Frames before a new track is reported (filters one-frame false positives)."""


class ReIdConfig(BaseModel):
    enabled: bool = True
    model: str = "models/osnet_x0_25_msmt17.onnx"
    device: Literal["cpu", "cuda", "tensorrt"] = "cuda"
    gallery: Literal["memory", "milvus"] = "memory"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "streamsight_reid"
    match_threshold: float = 0.65
    """Minimum cosine similarity to link a track to an existing identity."""
    min_track_frames: int = 5
    """A track must be this old before it is matched, so its embedding is stable."""
    embed_every: int = 5
    """Compute a new embedding for a track every N frames."""
    ema_alpha: float = 0.9
    """Smoothing factor for a track's running embedding."""


class EventsConfig(BaseModel):
    api_url: str | None = "http://localhost:8000"
    api_key: str | None = None
    jsonl_path: str | None = None
    batch_size: int = 100
    flush_interval_s: float = 1.0
    update_interval_s: float = 1.0
    """How often a live track publishes a position update (start/end/zone events are always sent)."""
    queue_size: int = 10_000


class MetricsConfig(BaseModel):
    enabled: bool = True
    port: int = 9100


class OutputConfig(BaseModel):
    save_dir: str | None = None
    """Write annotated video per camera to this folder."""
    draw_keypoints: bool = True


class AppConfig(BaseModel):
    cameras: list[CameraConfig] = Field(min_length=1)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    reid: ReIdConfig = Field(default_factory=ReIdConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    target_fps: float = 25.0

    @field_validator("cameras")
    @classmethod
    def _unique_camera_ids(cls, cams: list[CameraConfig]) -> list[CameraConfig]:
        ids = [c.id for c in cams]
        if len(ids) != len(set(ids)):
            raise ValueError("camera ids must be unique")
        return cams


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)
