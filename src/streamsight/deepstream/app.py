"""DeepStream pipeline: the high-throughput path for NVIDIA GPUs.

    nvurisrcbin x N -> nvstreammux -> nvinfer (YOLO11, TensorRT FP16)
        -> nvtracker (NvDCF) -> nvinfer (OSNet re-id, tensor meta)
        -> [probe: identities, zones, events] -> tiler -> osd -> encoder | fakesink

Decoding, scaling, inference and tracking all stay in GPU memory (NVMM). Python only
touches metadata in a pad probe: boxes, track ids and the 512-D re-id embeddings. The
code after tracking is shared with the Python pipeline (``CameraScene``), so both
produce identical events.

Requires DeepStream 7.x/8.x with the pyds bindings, and the parser library built in
deepstream/parser (``make -C deepstream/parser``).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from streamsight.config import AppConfig, load_config
from streamsight.pipeline import metrics
from streamsight.pipeline.publisher import EventPublisher
from streamsight.pipeline.scene import CameraScene
from streamsight.reid import CrossCameraMatcher, create_gallery
from streamsight.tracking.byte_tracker import Track, TrackState
from streamsight.tracking.kalman import xyxy_to_xyah

log = logging.getLogger("streamsight.deepstream")

ROOT = Path(__file__).resolve().parents[3]
DS_ROOT = Path("/opt/nvidia/deepstream/deepstream")
REID_DIM = 512


def _uri(uri: str) -> str:
    if "://" in uri:
        return uri
    return Path(uri).resolve().as_uri()


@dataclass
class SourceStats:
    frames: int = 0
    window_start: float = field(default_factory=time.perf_counter)
    window_frames: int = 0
    fps: float = 0.0

    def tick(self) -> None:
        self.frames += 1
        self.window_frames += 1
        now = time.perf_counter()
        if now - self.window_start >= 2.0:
            self.fps = self.window_frames / (now - self.window_start)
            self.window_start, self.window_frames = now, 0


class TrackBook:
    """Turns NvDCF object ids into ``Track`` objects that persist across frames."""

    def __init__(self, lost_after: int = 30, min_hits: int = 3) -> None:
        self.lost_after = lost_after
        self.min_hits = min_hits
        self.tracks: dict[int, Track] = {}

    def observe(self, oid: int, frame: int, box: np.ndarray, score: float, cls: int) -> Track:
        t = self.tracks.get(oid)
        mean = np.r_[xyxy_to_xyah(box), np.zeros(4)]
        if t is None:
            t = Track(oid, mean, np.eye(8), score, cls, start_frame=frame, last_frame=frame)
            self.tracks[oid] = t
        else:
            t.mean, t.score, t.last_frame = mean, score, frame
            t.hits += 1
        if t.hits >= self.min_hits:
            t.state = TrackState.CONFIRMED
        return t

    def expire(self, frame: int) -> list[Track]:
        gone = [t for t in self.tracks.values() if frame - t.last_frame > self.lost_after]
        for t in gone:
            del self.tracks[t.track_id]
        return [t for t in gone if t.hits >= self.min_hits]


class DeepStreamApp:
    def __init__(self, cfg: AppConfig, args: argparse.Namespace) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        self.Gst, self.GLib = Gst, GLib
        Gst.init(None)
        self.cfg = cfg
        self.args = args
        self.cameras = cfg.cameras
        self.publisher = EventPublisher.from_config(cfg.events).start()
        self.matcher = None
        if cfg.reid.enabled:
            gallery = create_gallery(cfg.reid, dim=REID_DIM)
            self.matcher = CrossCameraMatcher(gallery, cfg.reid.match_threshold, cfg.reid.min_track_frames)
        self.scenes = [
            CameraScene(cam, self.publisher.publish, self.matcher, cfg.events.update_interval_s)
            for cam in self.cameras
        ]
        self.books = [TrackBook(min_hits=cfg.tracker.min_hits) for _ in self.cameras]
        self.stats = [SourceStats() for _ in self.cameras]
        self.loop = GLib.MainLoop()
        self.pipeline = self._build()

    # ---------------------------------------------------------------- pipeline
    def _make(self, factory: str, name: str, **props):
        el = self.Gst.ElementFactory.make(factory, name)
        if el is None:
            raise RuntimeError(f"GStreamer element '{factory}' is missing; is DeepStream installed?")
        for k, v in props.items():
            el.set_property(k.replace("_", "-"), v)
        return el

    def _build(self):
        Gst = self.Gst
        n = len(self.cameras)
        live = any("://" in c.uri and not c.uri.startswith("file://") for c in self.cameras)
        p = Gst.Pipeline.new("streamsight")

        mux = self._make(
            "nvstreammux",
            "mux",
            batch_size=n,
            width=self.args.mux_width,
            height=self.args.mux_height,
            batched_push_timeout=40000,
            live_source=live,
        )
        p.add(mux)
        self._mux_sinks = []
        for i, cam in enumerate(self.cameras):
            src = self._make("nvurisrcbin", f"src{i}", uri=_uri(cam.uri))
            if live:
                src.set_property("rtsp-reconnect-interval", 10)
                src.set_property("latency", 200)
            elif cam.loop and not self.args.output:
                # When recording, play each file once so the MP4 gets a clean end of stream.
                src.set_property("file-loop", True)
            p.add(src)
            sink_pad = mux.request_pad_simple(f"sink_{i}")
            self._mux_sinks.append(sink_pad)
            src.connect("pad-added", self._on_pad_added, sink_pad)

        cfg_dir = Path(self.args.ds_config_dir)
        pgie = self._make("nvinfer", "pgie", config_file_path=str(cfg_dir / "pgie_yolo11.txt"), batch_size=n)
        tracker = self._make(
            "nvtracker",
            "tracker",
            ll_lib_file=str(DS_ROOT / "lib" / "libnvds_nvmultiobjecttracker.so"),
            ll_config_file=str(cfg_dir / "tracker_nvdcf.yml"),
            tracker_width=640,
            tracker_height=384,
        )
        chain = [pgie, tracker]
        if self.cfg.reid.enabled:
            chain.append(self._make("nvinfer", "sgie", config_file_path=str(cfg_dir / "sgie_osnet.txt")))

        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        tiler = self._make("nvmultistreamtiler", "tiler", rows=rows, columns=cols, width=1280, height=720)
        conv = self._make("nvvideoconvert", "conv")
        osd = self._make("nvdsosd", "osd")
        chain += [tiler, conv, osd]

        if self.args.output:
            enc_conv = self._make("nvvideoconvert", "enc_conv")
            enc = self._make("nvv4l2h264enc", "enc", bitrate=4_000_000)
            parse = self._make("h264parse", "parse")
            muxer = self._make("qtmux", "qtmux")
            sink = self._make("filesink", "sink", location=self.args.output, sync=False)
            chain += [enc_conv, enc, parse, muxer, sink]
        else:
            chain.append(self._make("fakesink", "sink", sync=self.args.sync))

        for el in chain:
            p.add(el)
        mux.link(chain[0])
        for a, b in zip(chain, chain[1:], strict=False):
            if not a.link(b):
                raise RuntimeError(f"failed to link {a.get_name()} -> {b.get_name()}")

        # Metadata is complete (detections + track ids + embeddings) after the last nvinfer.
        probe_el = chain[2] if self.cfg.reid.enabled else tracker
        probe_el.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._probe, None)

        bus = p.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)
        return p

    def _on_pad_added(self, _src, pad, sink_pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps.get_structure(0).get_name().startswith("video") and not sink_pad.is_linked():
            pad.link(sink_pad)

    def _on_message(self, _bus, msg) -> None:
        t = msg.type
        if t == self.Gst.MessageType.EOS:
            log.info("end of stream")
            self.loop.quit()
        elif t == self.Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            log.error("%s: %s (%s)", msg.src.get_name(), err.message, dbg)
            self.loop.quit()
        elif t == self.Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            log.warning("%s: %s", msg.src.get_name(), w.message)

    # ------------------------------------------------------------------- probe
    @staticmethod
    def _embedding(obj) -> np.ndarray | None:
        import pyds

        item = obj.obj_user_meta_list
        while item is not None:
            um = pyds.NvDsUserMeta.cast(item.data)
            if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tm, 0)
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                v = np.ctypeslib.as_array(ptr, shape=(REID_DIM,)).copy()
                return v / (np.linalg.norm(v) + 1e-12)
            item = item.next
        return None

    def _probe(self, pad, info, _data):
        import pyds

        buf = info.get_buffer()
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            frame = pyds.NvDsFrameMeta.cast(l_frame.data)
            sid = frame.pad_index
            book, scene, stats = self.books[sid], self.scenes[sid], self.stats[sid]
            fnum = frame.frame_num
            seen: list[Track] = []

            l_obj = frame.obj_meta_list
            while l_obj is not None:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                if obj.object_id != 0xFFFFFFFFFFFFFFFF:  # untracked objects have no id
                    r = obj.rect_params
                    box = np.array([r.left, r.top, r.left + r.width, r.top + r.height], np.float32)
                    t = book.observe(int(obj.object_id), fnum, box, float(obj.confidence), int(obj.class_id))
                    emb = self._embedding(obj)
                    if emb is not None:
                        t.update_embedding(emb, self.cfg.reid.ema_alpha)
                    if t.state == TrackState.CONFIRMED:
                        seen.append(t)
                    label = f"#{t.track_id}" + (
                        f" G{t.global_id % 100000}" if t.global_id is not None else ""
                    )
                    obj.text_params.display_text = label
                l_obj = l_obj.next

            scene.process(seen, book.expire(fnum))
            stats.tick()
            cam_id = self.cameras[sid].id
            metrics.FRAMES.labels(cam_id).inc()
            metrics.PIPELINE_FPS.labels(cam_id).set(stats.fps)
            l_frame = l_frame.next
        return self.Gst.PadProbeReturn.OK

    # --------------------------------------------------------------------- run
    def run(self) -> dict:
        if self.cfg.metrics.enabled:
            metrics.serve(self.cfg.metrics.port)
        if self.args.duration:
            self.GLib.timeout_add(int(self.args.duration * 1000), self._send_eos)
        signal.signal(signal.SIGINT, lambda *_: self._send_eos())
        started = time.perf_counter()
        self.pipeline.set_state(self.Gst.State.PLAYING)
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(self.Gst.State.NULL)
            elapsed = time.perf_counter() - started
            for book, scene in zip(self.books, self.scenes, strict=True):
                scene.process([], [t for t in book.tracks.values() if t.hits >= book.min_hits])
            self.publisher.close()
        gallery = self.matcher.gallery if self.matcher else None
        return {
            "pipeline": "deepstream",
            "elapsed_s": round(elapsed, 1),
            "cameras": {
                c.id: {"frames": s.frames, "avg_fps": round(s.frames / elapsed, 1) if elapsed else 0.0}
                for c, s in zip(self.cameras, self.stats, strict=True)
            },
            "total_fps": round(sum(s.frames for s in self.stats) / elapsed, 1) if elapsed else 0.0,
            "identities": len(gallery) if gallery is not None and hasattr(gallery, "__len__") else None,
            "events_sent": self.publisher.sent,
            "events_dropped": self.publisher.dropped,
        }

    def _send_eos(self) -> bool:
        log.info("stopping")
        if self.args.output:
            # End every muxer input; the muxer forwards EOS and qtmux finalises the file.
            for pad in self._mux_sinks:
                pad.send_event(self.Gst.Event.new_eos())
        else:
            self.loop.quit()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamSight DeepStream pipeline")
    parser.add_argument("-c", "--config", default="configs/pipeline.yaml")
    parser.add_argument("--ds-config-dir", default=str(ROOT / "deepstream" / "configs"))
    parser.add_argument("--output", help="write the tiled, annotated video to this .mp4")
    parser.add_argument("--sync", action="store_true", help="play files in real time instead of max speed")
    parser.add_argument("--duration", type=float, help="stop after N seconds")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--mux-width", type=int, default=1280)
    parser.add_argument("--mux-height", type=int, default=720)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.no_api:
        cfg.events.api_url = None
    try:
        app = DeepStreamApp(cfg, args)
    except (ImportError, ValueError, RuntimeError) as exc:
        sys.exit(f"cannot start DeepStream pipeline: {exc}")
    print(json.dumps(app.run(), indent=2))


if __name__ == "__main__":
    main()
