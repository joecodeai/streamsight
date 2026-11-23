"""OpenVINO backend: Intel CPU, iGPU or NPU, with no NVIDIA dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streamsight.detection.base import Detector

_DEVICE_MAP = {"cpu": "CPU", "cuda": "GPU", "tensorrt": "GPU"}


class OpenVinoDetector(Detector):
    backend_name = "openvino"

    def __init__(self, model_path: str | Path, device: str = "cpu", **kwargs) -> None:
        super().__init__(**kwargs)
        import openvino as ov

        core = ov.Core()
        ov_device = _DEVICE_MAP.get(device, device.upper())
        if ov_device not in core.available_devices:
            ov_device = "CPU"
        model = core.read_model(str(model_path))
        self.compiled = core.compile_model(model, ov_device, {"PERFORMANCE_HINT": "THROUGHPUT"})
        self.request = self.compiled.create_infer_request()
        self.backend_name = f"openvino-{ov_device.lower()}"
        shape = model.inputs[0].get_partial_shape()
        self._fixed_batch = shape[0].get_length() if shape[0].is_static else None
        if shape[2].is_static and shape[2].get_length() != self.input_size:
            self.input_size = shape[2].get_length()

    @property
    def fixed_batch(self) -> int | None:
        return self._fixed_batch

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        result = self.request.infer({0: batch})
        return next(iter(result.values()))
