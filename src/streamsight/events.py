"""Event schema shared by the pipelines (producers) and the API (consumer)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "track_start",
    "track_update",
    "track_end",
    "zone_enter",
    "zone_exit",
    "fall_suspected",
    "reid_match",
]


class TrackEvent(BaseModel):
    camera_id: str
    event_type: EventType
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    local_track_id: int
    global_id: int | None = None
    label: str = "person"
    confidence: float = 0.0
    bbox: list[float] = Field(default_factory=list, max_length=4)
    keypoints: list[list[float]] | None = None
    zone: str | None = None
    attributes: dict = Field(default_factory=dict)


class EventBatch(BaseModel):
    events: list[TrackEvent] = Field(max_length=5000)


COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]  # fmt: skip


def label_for(class_id: int) -> str:
    return COCO_LABELS[class_id] if 0 <= class_id < len(COCO_LABELS) else str(class_id)
