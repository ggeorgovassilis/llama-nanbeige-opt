#!/usr/bin/env bash
# Run the CUDA llama-server inside Docker (OpenAI-compatible API on :8080).
# Usage: run-server.sh [model-name] [extra llama-server args...]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="llama-cpp-cuda-build:local"
MODELS_DIR="/home/george/apps/llamacpp/models"
MODEL_NAME="Nanbeige4.2-3B-Q4_K_M.gguf"
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODEL_NAME="$1"
    shift || true
fi
PORT="${PORT:-8080}"

# Resolve the model symlink to its real path (models live on an encrypted FUSE mount).
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
if [[ ! -e "$MODEL_PATH" ]]; then
    echo "error: model not found: $MODEL_PATH" >&2
    exit 1
fi
MODEL_REAL="$(realpath "$MODEL_PATH")"

exec docker run --rm --gpus all -p "$PORT:8080" \
    --shm-size 2gb \
    --ulimit memlock=-1:-1 \
    -e CUDA_DEVICE_SCHEDULE=4 \
    -e OMP_WAIT_POLICY=PASSIVE \
    -v "$REPO_DIR":/app \
    -v "${MODEL_REAL}:/models/${MODEL_NAME}:ro" \
    -w /app \
    "$IMAGE" \
    /app/build-cuda/bin/llama-server \
        -m "/models/$MODEL_NAME" \
        --n-gpu-layers 99 \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --parallel 1 \
        --host 0.0.0.0 \
        --port 8080 \
        "$@"
