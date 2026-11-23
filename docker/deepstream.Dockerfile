# DeepStream pipeline (x86_64 dGPU). Build on a host with the NVIDIA Container Toolkit.
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-pip python3-gi python3-gst-1.0 \
 && rm -rf /var/lib/apt/lists/*

ARG PYDS_URL=https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl
RUN pip3 install "${PYDS_URL}"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip3 install ".[milvus]"

COPY deepstream ./deepstream
RUN make -C deepstream/parser

COPY configs ./configs
COPY scripts ./scripts

VOLUME ["/app/models"]
EXPOSE 9100

ENTRYPOINT ["streamsight-deepstream", "--ds-config-dir", "/app/deepstream/configs"]
CMD ["-c", "configs/pipeline.rtsp.yaml"]
