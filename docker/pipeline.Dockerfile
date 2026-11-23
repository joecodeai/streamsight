# Python pipeline with ONNX Runtime GPU (CUDA + TensorRT providers) and OpenVINO.
# CUDA, cuDNN and TensorRT come from pip wheels, so a plain Python base image is enough;
# the host only needs the NVIDIA driver and the NVIDIA Container Toolkit.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[openvino,milvus]" "onnxruntime-gpu[cuda,cudnn]" tensorrt-cu13

COPY configs ./configs
COPY scripts ./scripts

# Models are mounted at runtime (see docker-compose.yml), not baked into the image.
VOLUME ["/app/models", "/app/models/trt_cache"]
EXPOSE 9100

ENTRYPOINT ["streamsight-pipeline"]
CMD ["-c", "configs/pipeline.rtsp.yaml"]
