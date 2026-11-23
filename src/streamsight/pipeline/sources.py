"""Threaded video sources.

Each camera is read on its own thread and only the newest frame is kept, so a slow
consumer never builds up latency: stale frames are dropped, not queued. RTSP
sources reconnect with exponential backoff; file sources can loop and are paced
to their native FPS so they behave like live cameras.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# TCP avoids the packet loss (grey smeared frames) common with RTSP over UDP.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


@dataclass
class Frame:
    camera_id: str
    index: int
    ts: float
    image: np.ndarray


class VideoSource:
    def __init__(self, camera_id: str, uri: str, loop: bool = True, realtime: bool = True) -> None:
        self.camera_id = camera_id
        self.uri = uri
        self.is_file = not uri.startswith(("rtsp://", "rtmp://", "http://", "https://")) and not uri.isdigit()
        self.loop = loop
        self.realtime = realtime
        self._latest: Frame | None = None
        self._consumed = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"src-{camera_id}", daemon=True)
        self.connected = False
        self.fps = 0.0
        self.frames_read = 0
        self.reconnects = 0

    def start(self) -> VideoSource:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def read_new(self) -> Frame | None:
        """Newest frame if it has not been returned before, else ``None``."""
        with self._lock:
            f = self._latest
            if f is None or f.index == self._consumed:
                return None
            self._consumed = f.index
            return f

    def _open(self) -> cv2.VideoCapture:
        src = int(self.uri) if self.uri.isdigit() else self.uri
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG) if not isinstance(src, int) else cv2.VideoCapture(src)

    def _run(self) -> None:
        backoff = 1.0
        index = 0
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                log.warning("%s: cannot open %s, retrying in %.0fs", self.camera_id, self.uri, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                self.reconnects += 1
                continue
            self.connected = True
            backoff = 1.0
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            period = 1.0 / native_fps if self.is_file and self.realtime else 0.0
            next_t = time.perf_counter()
            window_start, window_frames = time.perf_counter(), 0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                index += 1
                self.frames_read += 1
                with self._lock:
                    self._latest = Frame(self.camera_id, index, time.time(), img)
                window_frames += 1
                now = time.perf_counter()
                if now - window_start >= 2.0:
                    self.fps = window_frames / (now - window_start)
                    window_start, window_frames = now, 0
                if period:
                    next_t += period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.perf_counter()
            cap.release()
            self.connected = False
            if self.is_file and not self.loop:
                log.info("%s: end of file", self.camera_id)
                return
            if not self.is_file:
                log.warning("%s: stream dropped, reconnecting", self.camera_id)
                self.reconnects += 1
                self._stop.wait(backoff)
