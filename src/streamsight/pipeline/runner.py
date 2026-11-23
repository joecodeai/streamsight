"""Multi-camera pipeline: one batched detector pass over the newest frame of every camera.

Loop per iteration:
1. collect the newest unseen frame from each camera
2. detect on all of them in one batch
3. per camera: track -> embed tracks that are due -> scene logic -> draw/save
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from streamsight.config import AppConfig
from streamsight.detection import create_detector
from streamsight.pipeline import metrics
from streamsight.pipeline.overlay import draw
from streamsight.pipeline.publisher import EventPublisher
from streamsight.pipeline.scene import CameraScene
from streamsight.pipeline.sources import VideoSource
from streamsight.reid import CrossCameraMatcher, ReIdEmbedder, create_gallery
from streamsight.tracking import ByteTracker

log = logging.getLogger(__name__)


@dataclass
class CameraState:
    source: VideoSource
    tracker: ByteTracker
    scene: CameraScene
    writer: cv2.VideoWriter | None = None
    processed: int = 0
    fps: float = 0.0
    _window: list[float] = field(default_factory=list)
    _last_embed: dict[int, int] = field(default_factory=dict)

    def tick(self) -> None:
        now = time.perf_counter()
        self._window.append(now)
        while self._window and now - self._window[0] > 2.0:
            self._window.pop(0)
        if len(self._window) > 1:
            self.fps = (len(self._window) - 1) / (self._window[-1] - self._window[0])


class Pipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.detector = create_detector(cfg.detector)
        self.detector.warmup()
        log.info("detector: %s (%s)", cfg.detector.model, self.detector.backend_name)

        self.embedder = None
        self.matcher = None
        if cfg.reid.enabled:
            self.embedder = ReIdEmbedder(cfg.reid.model, device=cfg.reid.device)
            gallery = create_gallery(cfg.reid, dim=self.embedder.dim)
            self.matcher = CrossCameraMatcher(gallery, cfg.reid.match_threshold, cfg.reid.min_track_frames)

        self.publisher = EventPublisher.from_config(cfg.events).start()
        self.cameras: dict[str, CameraState] = {}
        for cam in cfg.cameras:
            scene = CameraScene(
                cam,
                self.publisher.publish,
                self.matcher,
                cfg.events.update_interval_s,
                fall_detection=cfg.detector.task == "pose",
            )
            self.cameras[cam.id] = CameraState(
                VideoSource(cam.id, cam.uri, cam.loop, cam.realtime),
                ByteTracker.from_config(cfg.tracker),
                scene,
            )
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _embed(self, state: CameraState, image: np.ndarray, tracks: list) -> None:
        every = self.cfg.reid.embed_every
        due = [
            t
            for t in tracks
            if t.embedding is None
            or state.tracker.frame_id - state._last_embed.get(t.track_id, -every) >= every
        ]
        if not due:
            return
        embs = self.embedder(image, np.stack([t.box for t in due]))
        for t, e in zip(due, embs, strict=True):
            t.update_embedding(e, self.cfg.reid.ema_alpha)
            state._last_embed[t.track_id] = state.tracker.frame_id

    def _writer_for(self, state: CameraState, image: np.ndarray) -> cv2.VideoWriter | None:
        if not self.cfg.output.save_dir:
            return None
        if state.writer is None:
            out = Path(self.cfg.output.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            h, w = image.shape[:2]
            path = str(out / f"{state.source.camera_id}.mp4")
            state.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        return state.writer

    def run(self, duration_s: float | None = None, max_frames: int | None = None) -> dict:
        for s in self.cameras.values():
            s.source.start()
        if self.cfg.metrics.enabled:
            metrics.serve(self.cfg.metrics.port)

        started = time.perf_counter()
        infer_ms: list[float] = []
        total = 0
        try:
            while not self._stop:
                if duration_s and time.perf_counter() - started >= duration_s:
                    break
                if max_frames and total >= max_frames:
                    break
                frames = [f for s in self.cameras.values() if (f := s.source.read_new()) is not None]
                if not frames:
                    if all(
                        s.source.is_file and not s.source.loop and not s.source._thread.is_alive()
                        for s in self.cameras.values()
                    ):
                        break
                    time.sleep(0.002)
                    continue

                t0 = time.perf_counter()
                detections = self.detector([f.image for f in frames])
                t1 = time.perf_counter()
                metrics.INFERENCE_SECONDS.labels(self.detector.backend_name).observe(
                    self.detector.timing.inference_ms / 1000
                )
                metrics.STAGE_SECONDS.labels("detect").observe(t1 - t0)
                infer_ms.append(self.detector.timing.inference_ms)

                for frame, dets in zip(frames, detections, strict=True):
                    state = self.cameras[frame.camera_id]
                    ts = time.perf_counter()
                    tracks = state.tracker.update(dets)
                    metrics.STAGE_SECONDS.labels("track").observe(time.perf_counter() - ts)
                    if self.embedder is not None and tracks:
                        te = time.perf_counter()
                        self._embed(state, frame.image, tracks)
                        metrics.STAGE_SECONDS.labels("reid").observe(time.perf_counter() - te)
                    for t in state.tracker.removed:
                        state._last_embed.pop(t.track_id, None)
                    state.scene.process(tracks, state.tracker.removed)

                    state.processed += 1
                    state.tick()
                    metrics.FRAMES.labels(frame.camera_id).inc()
                    metrics.PIPELINE_FPS.labels(frame.camera_id).set(state.fps)
                    metrics.SOURCE_FPS.labels(frame.camera_id).set(state.source.fps)
                    metrics.SOURCE_UP.labels(frame.camera_id).set(1 if state.source.connected else 0)

                    writer = self._writer_for(state, frame.image)
                    if writer is not None:
                        writer.write(
                            draw(
                                frame.image,
                                tracks,
                                state.scene.camera.zones,
                                state.scene.zones.occupancy,
                                self.cfg.output.draw_keypoints,
                                state.fps,
                            )
                        )
                total += len(frames)
        finally:
            elapsed = time.perf_counter() - started
            for s in self.cameras.values():
                s.source.stop()
                if s.writer is not None:
                    s.writer.release()
                # close open tracks so the API sees track_end / zone_exit for everyone still in view
                still_open = [t for t in s.tracker.tracks if t.hits >= self.cfg.tracker.min_hits]
                s.scene.process([], still_open)
            self.publisher.close()

        identities = (
            len(self.matcher.gallery) if self.matcher and hasattr(self.matcher.gallery, "__len__") else None
        )
        return {
            "backend": self.detector.backend_name,
            "elapsed_s": round(elapsed, 1),
            "frames_total": total,
            "cameras": {
                cid: {"frames": s.processed, "avg_fps": round(s.processed / elapsed, 1) if elapsed else 0.0}
                for cid, s in self.cameras.items()
            },
            "detector_ms_p50": round(float(np.percentile(infer_ms, 50)), 2) if infer_ms else None,
            "detector_ms_p95": round(float(np.percentile(infer_ms, 95)), 2) if infer_ms else None,
            "identities": identities,
            "events_sent": self.publisher.sent,
            "events_dropped": self.publisher.dropped,
        }
