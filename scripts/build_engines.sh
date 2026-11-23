#!/usr/bin/env bash
# Build TensorRT engines ahead of time with trtexec.
#
# DeepStream can build engines itself on first start, but doing it offline is faster
# to deploy (no multi-minute warm-up per container) and avoids builder issues inside
# nvinfer. Engine names follow nvinfer's convention so the configs pick them up.
#
# Usage: scripts/build_engines.sh [max_batch] [precision]
#   max_batch  largest number of cameras per detector batch (default 4)
#   precision  fp16 (default) or fp32

set -euo pipefail

MAX_BATCH="${1:-4}"
PRECISION="${2:-fp16}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
MODELS="${MODELS:-models}"

if [[ ! -x "$TRTEXEC" ]]; then
  echo "trtexec not found at $TRTEXEC (set TRTEXEC=...)" >&2
  exit 1
fi

flags=()
[[ "$PRECISION" == "fp16" ]] && flags+=(--fp16)

build() {
  local onnx="$1" input="$2" chw="$3" max_b="$4" out="$5"
  local opt_b=$(( max_b < 4 ? max_b : 4 ))
  echo "building $out (batch 1..$max_b, $PRECISION)"
  "$TRTEXEC" --onnx="$onnx" "${flags[@]}" \
    --minShapes="${input}:1x${chw}" \
    --optShapes="${input}:${opt_b}x${chw}" \
    --maxShapes="${input}:${max_b}x${chw}" \
    --saveEngine="$out" --skipInference > "${out}.log" 2>&1 \
    || { echo "failed, see ${out}.log" >&2; exit 1; }
}

build "$MODELS/yolo11n.onnx" images 3x640x640 "$MAX_BATCH" \
  "$MODELS/yolo11n.onnx_b${MAX_BATCH}_gpu0_${PRECISION}.engine"
build "$MODELS/osnet_x0_25_msmt17.onnx" input 3x256x128 16 \
  "$MODELS/osnet_x0_25_msmt17.onnx_b16_gpu0_${PRECISION}.engine"

echo "done"
ls -lh "$MODELS"/*.engine
