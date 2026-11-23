"""Non-blocking event publisher.

The video loop only enqueues events. A background thread batches them and POSTs
to the API (with retry), and/or appends them to a JSONL file. If the API is down
for long enough that the bounded queue fills up, the oldest events are dropped
and counted, instead of stalling video processing.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from pathlib import Path

import httpx

from streamsight.events import TrackEvent
from streamsight.pipeline import metrics

log = logging.getLogger(__name__)


class EventPublisher:
    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        jsonl_path: str | None = None,
        batch_size: int = 100,
        flush_interval_s: float = 1.0,
        queue_size: int = 10_000,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = f"{api_url.rstrip('/')}/api/v1/events/batch" if api_url else None
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.jsonl = Path(jsonl_path) if jsonl_path else None
        if self.jsonl:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._queue: collections.deque[TrackEvent] = collections.deque(maxlen=queue_size)
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._client = client or (httpx.Client(timeout=5.0) if self.endpoint else None)
        self._thread = threading.Thread(target=self._run, name="event-publisher", daemon=True)
        self.sent = 0
        self.dropped = 0

    @classmethod
    def from_config(cls, cfg) -> EventPublisher:
        return cls(
            cfg.api_url, cfg.api_key, cfg.jsonl_path, cfg.batch_size, cfg.flush_interval_s, cfg.queue_size
        )

    def start(self) -> EventPublisher:
        self._thread.start()
        return self

    def publish(self, event: TrackEvent) -> None:
        with self._cv:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
                metrics.EVENTS_DROPPED.inc()
            self._queue.append(event)
            if len(self._queue) >= self.batch_size:
                self._cv.notify()

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify()
        self._thread.join(timeout=timeout)
        if self._client:
            self._client.close()

    def _take_batch(self) -> list[TrackEvent]:
        with self._cv:
            n = min(len(self._queue), self.batch_size)
            return [self._queue.popleft() for _ in range(n)]

    def _run(self) -> None:
        while True:
            with self._cv:
                if not self._queue and not self._stop.is_set():
                    self._cv.wait(self.flush_interval_s)
            batch = self._take_batch()
            if batch:
                self._deliver(batch)
            elif self._stop.is_set():
                return

    def _deliver(self, batch: list[TrackEvent]) -> None:
        if self.jsonl:
            with self.jsonl.open("a", encoding="utf-8") as f:
                for e in batch:
                    f.write(e.model_dump_json() + "\n")
        if not self.endpoint:
            self.sent += len(batch)
            metrics.EVENTS_PUBLISHED.inc(len(batch))
            return
        payload = {"events": [json.loads(e.model_dump_json()) for e in batch]}
        delay = 0.5
        for attempt in range(4):
            try:
                r = self._client.post(self.endpoint, json=payload, headers=self.headers)
                r.raise_for_status()
                self.sent += len(batch)
                metrics.EVENTS_PUBLISHED.inc(len(batch))
                return
            except httpx.HTTPError as exc:
                if attempt == 3 or self._stop.is_set():
                    log.warning("dropping %d events after error: %s", len(batch), exc)
                    break
                time.sleep(delay)
                delay *= 2
        self.dropped += len(batch)
        metrics.EVENTS_DROPPED.inc(len(batch))
