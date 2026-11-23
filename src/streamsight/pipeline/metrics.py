"""Prometheus metrics exported by the pipeline (scraped on ``metrics.port``)."""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

FRAMES = Counter("streamsight_frames_processed_total", "Frames run through the pipeline", ["camera"])
PIPELINE_FPS = Gauge("streamsight_pipeline_fps", "Processed frames per second", ["camera"])
SOURCE_FPS = Gauge("streamsight_source_fps", "Frames per second read from the source", ["camera"])
SOURCE_UP = Gauge("streamsight_source_connected", "1 if the camera stream is connected", ["camera"])
ACTIVE_TRACKS = Gauge("streamsight_active_tracks", "Confirmed tracks in the current frame", ["camera"])
ZONE_OCCUPANCY = Gauge("streamsight_zone_occupancy", "Tracks inside a zone", ["camera", "zone"])
INFERENCE_SECONDS = Histogram(
    "streamsight_inference_seconds",
    "Detector forward pass time per batch",
    ["backend"],
    buckets=(0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.25),
)
STAGE_SECONDS = Histogram(
    "streamsight_stage_seconds",
    "Time per pipeline stage per batch",
    ["stage"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25),
)
REID_DECISIONS = Counter("streamsight_reid_decisions_total", "Cross-camera identity decisions", ["result"])
EVENTS_PUBLISHED = Counter("streamsight_events_published_total", "Events delivered")
EVENTS_DROPPED = Counter("streamsight_events_dropped_total", "Events dropped (queue full or API errors)")


def serve(port: int) -> None:
    start_http_server(port)
