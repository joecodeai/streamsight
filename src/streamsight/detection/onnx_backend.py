"""ONNX Runtime backend: CPU, CUDA or TensorRT execution provider."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from streamsight.detection.base import Detector

_dlls_loaded = False


def _preload_cuda_libs() -> None:
    """Load CUDA/cuDNN from the nvidia-* pip wheels, so no system CUDA toolkit is needed."""
    global _dlls_loaded
    if _dlls_loaded:
        return
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        with contextlib.suppress(Exception):  # wheels not installed: rely on system libraries
            ort.preload_dlls()
    _dlls_loaded = True


@dataclass(frozen=True)
class InputMeta:
    name: str
    shape: tuple  # (batch, C, H, W); dynamic axes are strings or None

    @property
    def fixed_batch(self) -> int | None:
        b = self.shape[0]
        return b if isinstance(b, int) and b > 0 else None


def read_input_meta(model_path: str | Path) -> InputMeta:
    """Input name and shape without initialising a GPU provider."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    return InputMeta(inp.name, tuple(inp.shape))


def ort_providers(
    device: str,
    fp16: bool = True,
    cache_dir: str = "models/trt_cache",
    profile: tuple[str, tuple[int, int, int], int] | None = None,
) -> list:
    """Execution providers in priority order; ONNX Runtime falls back down the list.

    ``profile`` = (input name, (C, H, W), max batch). With a dynamic batch axis the
    TensorRT provider otherwise builds a new engine for every batch size it sees,
    which stalls the pipeline for minutes; one profile covering 1..max avoids that.
    """
    if device != "cpu":
        _preload_cuda_libs()
    if device == "tensorrt":
        os.makedirs(cache_dir, exist_ok=True)
        opts = {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache_dir,
        }
        if profile is not None:
            name, (c, h, w), max_b = profile
            opt_b = min(max_b, 4)
            opts["trt_profile_min_shapes"] = f"{name}:1x{c}x{h}x{w}"
            opts["trt_profile_opt_shapes"] = f"{name}:{opt_b}x{c}x{h}x{w}"
            opts["trt_profile_max_shapes"] = f"{name}:{max_b}x{c}x{h}x{w}"
        return [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxRuntimeDetector(Detector):
    backend_name = "onnxruntime"

    def __init__(self, model_path: str | Path, device: str = "cuda", fp16: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        import onnxruntime as ort

        meta = read_input_meta(model_path)
        if isinstance(meta.shape[2], int):
            self.input_size = int(meta.shape[2])
        self._fixed_batch = meta.fixed_batch
        profile = None
        if self._fixed_batch is None:
            profile = (meta.name, (3, self.input_size, self.input_size), self.max_batch)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=ort_providers(device, fp16, profile=profile)
        )
        self.active_providers = self.session.get_providers()
        self.backend_name = f"onnxruntime-{self.active_providers[0].replace('ExecutionProvider', '').lower()}"
        self.input_name = meta.name

    @property
    def fixed_batch(self) -> int | None:
        return self._fixed_batch

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0]
