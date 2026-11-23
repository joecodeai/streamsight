PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install models samples engines parser api run deepstream bench test lint format up down

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

venv: ## create the virtual environment
	$(PY) -m venv --system-site-packages $(VENV)
	$(BIN)/pip install --upgrade pip

install: venv ## install the package with GPU, API and dev extras
	$(BIN)/pip install -e ".[api,openvino,milvus,dev,export]" "onnxruntime-gpu[cuda,cudnn]"

models: ## export YOLO11 detect/pose to ONNX and fetch OSNet
	$(BIN)/python scripts/export_models.py

samples: ## download the EPFL multi-camera sample videos (H.264)
	$(BIN)/python scripts/download_samples.py

engines: ## build TensorRT engines for DeepStream with trtexec
	scripts/build_engines.sh 4 fp16

parser: ## build the DeepStream YOLO11 bbox parser
	$(MAKE) -C deepstream/parser

api: ## run the API + dashboard on :8000 (SQLite)
	$(BIN)/streamsight-api

run: ## run the Python pipeline on the sample videos
	$(BIN)/streamsight-pipeline -c configs/pipeline.yaml

deepstream: parser ## run the DeepStream pipeline on the sample videos
	$(BIN)/streamsight-deepstream -c configs/pipeline.yaml

bench: ## benchmark detector backends on this machine
	$(BIN)/streamsight-bench

test: ## run the tests
	$(BIN)/pytest

lint: ## ruff lint + format check
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

format:
	$(BIN)/ruff format src tests scripts

up: ## start API, Postgres, Milvus, RTSP simulator and monitoring
	docker compose up -d

down:
	docker compose down
