#!/usr/bin/env bash
# Run the CUDA one-shot completion inside Docker against a GGUF model.
# Usage: run-cli.sh [model-name] [extra completion args...]
# Default model: Nanbeige4.2-3B-Q4_K_M.gguf
#
# Uses the `llama completion` subcommand (one-shot, non-interactive).
# The standalone `llama-cli` binary is interactive-only and spins on stdin, so do
# not use it for scripted generation.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="llama-cpp-cuda-build:local"
MODELS_DIR="/home/george/apps/llamacpp/models"
MODEL_NAME="Nanbeige4.2-3B-Q4_K_M.gguf"
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODEL_NAME="$1"
    shift || true
fi

# Resolve the model symlink to its real path (models live on an encrypted FUSE mount).
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
if [[ ! -e "$MODEL_PATH" ]]; then
    echo "error: model not found: $MODEL_PATH" >&2
    exit 1
fi
MODEL_REAL="$(realpath "$MODEL_PATH")"

exec docker run --rm --gpus all \
    --shm-size 2gb \
    --ulimit memlock=-1:-1 \
    -e CUDA_DEVICE_SCHEDULE=4 \
    -e OMP_WAIT_POLICY=PASSIVE \
    -v "$REPO_DIR":/app \
    -v "${MODEL_REAL}:/models/${MODEL_NAME}:ro" \
    -w /app \
    "$IMAGE" \
    /app/build-cuda/bin/llama completion \
        -m "/models/$MODEL_NAME" \
        -ngl 99 \
        -ctk q8_0 \
        -ctv q8_0 \
        --temperature 0.6 \
        -no-cnv \
        "$@"
