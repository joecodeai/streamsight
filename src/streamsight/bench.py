"""Benchmark detector backends on the local machine.

Measures the model forward pass only (no decode or NMS), so the numbers compare
runtimes, not post-processing. Writes JSON and a Markdown table to benchmarks/results/.

    streamsight-bench --model models/yolo11n.onnx --batches 1 4 8
    streamsight-bench --backends ort-cpu openvino-cpu      # CPU-only machine
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BACKENDS = ["ort-cpu", "ort-cuda", "ort-tensorrt", "openvino-cpu", "trtexec-fp32", "trtexec-fp16"]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _time(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    return samples


def bench_python(backend: str, model: str, batch: int, size: int, iters: int, warmup: int) -> list[float]:
    x = np.random.default_rng(0).random((batch, 3, size, size), dtype=np.float32)
    if backend.startswith("ort-"):
        import onnxruntime as ort

        from streamsight.detection.onnx_backend import ort_providers, read_input_meta

        device = backend.removeprefix("ort-")
        meta = read_input_meta(model)
        profile = (meta.name, (3, size, size), max(batch, 8)) if device == "tensorrt" else None
        sess = ort.InferenceSession(model, providers=ort_providers(device, fp16=True, profile=profile))
        if not sess.get_providers()[0].lower().startswith(device):
            raise RuntimeError(f"{device} provider unavailable (got {sess.get_providers()[0]})")
        return _time(lambda: sess.run(None, {meta.name: x}), iters, warmup)
    if backend == "openvino-cpu":
        import openvino as ov

        core = ov.Core()
        compiled = core.compile_model(core.read_model(model), "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        req = compiled.create_infer_request()
        return _time(lambda: req.infer({0: x}), iters, warmup)
    raise ValueError(backend)


def bench_trtexec(model: str, batch: int, size: int, precision: str, input_name: str) -> dict:
    shape = f"{input_name}:{batch}x3x{size}x{size}"
    cmd = [TRTEXEC, f"--onnx={model}", f"--shapes={shape}", f"--minShapes={shape}", f"--optShapes={shape}",
           f"--maxShapes={shape}", "--duration=5", "--warmUp=500", "--useCudaGraph"]  # fmt: skip
    if precision == "fp16":
        cmd.append("--fp16")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if out.returncode != 0:
        raise RuntimeError(out.stdout[-500:] + out.stderr[-500:])
    lat = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, "
                    r"percentile\(90%\) = [\d.]+ ms, percentile\(95%\) = ([\d.]+) ms", out.stdout)  # fmt: skip
    if not lat:
        raise RuntimeError("could not parse trtexec output")
    return {"mean_ms": float(lat.group(1)), "p50_ms": float(lat.group(2)), "p95_ms": float(lat.group(3))}


def _environment(model: str, size: int) -> dict:
    import onnxruntime

    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": Path(model).name,
        "input": f"{size}x{size}",
        "gpu": _gpu_name(),
        "cpu": _cpu_name(),
        "onnxruntime": onnxruntime.__version__,
    }
    try:
        import openvino

        meta["openvino"] = openvino.__version__.split("-")[0]
    except ImportError:
        pass
    return meta


def write_results(out: Path, meta: dict, rows: list[dict]) -> None:
    """Rewritten after every measurement, so an interrupted run keeps what it has."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "detector_backends.json").write_text(json.dumps({"meta": meta, "results": rows}, indent=2))
    lines = [
        f"Model `{meta['model']}` at {meta['input']}, forward pass only. GPU: {meta['gpu']}. CPU: {meta['cpu']}. "
        f"ONNX Runtime {meta['onnxruntime']}" + (f", OpenVINO {meta['openvino']}" if "openvino" in meta else "")
        + f". Measured {meta['date']}.",
        "",
        "| Backend | Batch | p50 latency (ms) | p95 latency (ms) | Throughput (img/s) |",
        "|---|---:|---:|---:|---:|",
    ]  # fmt: skip
    lines += [
        f"| {r['backend']} | {r['batch']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['images_per_s']:.0f} |"
        for r in rows
    ]
    (out / "detector_backends.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="models/yolo11n.onnx")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    from streamsight.detection.onnx_backend import read_input_meta

    input_name = read_input_meta(args.model).name
    meta = _environment(args.model, args.size)
    out = Path(args.out)
    rows: list[dict] = []
    for backend in args.backends:
        for batch in args.batches:
            label = f"{backend} b{batch}"
            try:
                if backend.startswith("trtexec"):
                    stats = bench_trtexec(args.model, batch, args.size, backend.split("-")[1], input_name)
                else:
                    iters = args.iters if "cpu" not in backend else max(20, args.iters // 5)
                    s = np.asarray(bench_python(backend, args.model, batch, args.size, iters, args.warmup))
                    stats = {
                        "mean_ms": float(s.mean()),
                        "p50_ms": float(np.percentile(s, 50)),
                        "p95_ms": float(np.percentile(s, 95)),
                    }
            except Exception as exc:  # report and keep going; one missing runtime should not stop the run
                print(f"{label:24s} skipped: {str(exc).splitlines()[0][:120]}", flush=True)
                continue
            stats["images_per_s"] = batch * 1000.0 / stats["mean_ms"]
            rows.append({"backend": backend, "batch": batch, **{k: round(v, 3) for k, v in stats.items()}})
            write_results(out, meta, rows)
            print(f"{label:24s} p50 {stats['p50_ms']:8.2f} ms   p95 {stats['p95_ms']:8.2f} ms   "
                  f"{stats['images_per_s']:9.1f} img/s", flush=True)  # fmt: skip

    print(f"\nwrote {out / 'detector_backends.md'}")


if __name__ == "__main__":
    main()
