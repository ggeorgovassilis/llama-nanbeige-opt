#!/usr/bin/env bash
# Build and run the KV probe, then analyze the dump.
# Usage: kv-probe.sh [extra args passed to kv-probe, e.g. -p "..." ]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="llama-cpp-cuda-build:local"
MODELS_DIR="/home/george/apps/llamacpp/models"
MODEL_NAME="Nanbeige4.2-3B-Q4_K_M.gguf"

MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
if [[ ! -e "$MODEL_PATH" ]]; then
    echo "error: model not found: $MODEL_PATH" >&2
    exit 1
fi
MODEL_REAL="$(realpath "$MODEL_PATH")"

mkdir -p "$REPO_DIR/plan/logs"

echo "==> rebuilding libllama (pick up get_v_storage)"
docker run --rm --gpus all \
    -v "$REPO_DIR":/app \
    -w /app \
    "$IMAGE" \
    bash -c 'cmake --build build-cuda --target llama -j"$(nproc)"'

echo "==> compiling kv-probe"
docker run --rm --gpus all \
    -v "$REPO_DIR":/app \
    -w /app \
    "$IMAGE" \
    bash -c '
        /usr/bin/c++ -std=c++17 -O2 \
            -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DLLAMA_SHARED \
            -I/app/src -I/app/include -I/app/ggml/include -I/app/build-cuda/src -I/app/common \
            /app/plan/kv-probe.cpp -o /app/build-cuda/bin/kv-probe \
            -L/app/build-cuda/bin -lllama -lggml -lggml-base -lggml-cpu \
            -Wl,-rpath,/app/build-cuda/bin
    '

echo "==> running kv-probe"
docker run --rm --gpus all \
    --shm-size 2gb \
    --ulimit memlock=-1:-1 \
    -e CUDA_DEVICE_SCHEDULE=4 \
    -e OMP_WAIT_POLICY=PASSIVE \
    -e GGML_BACKEND_PATH=/app/build-cuda/bin/libggml-cuda.so \
    -e LD_LIBRARY_PATH=/app/build-cuda/bin \
    -v "$REPO_DIR":/app \
    -v "${MODEL_REAL}:/models/${MODEL_NAME}:ro" \
    -w /app \
    "$IMAGE" \
    /app/build-cuda/bin/kv-probe \
        -m "/models/$MODEL_NAME" \
        -o /app/plan/logs/kv-dump.bin \
        "$@"

echo "==> analyzing dump"
python3 "$REPO_DIR/plan/kv-similarity.py" "$REPO_DIR/plan/logs/kv-dump.bin"
