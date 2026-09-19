#!/usr/bin/env bash
# Build llama.cpp (CUDA, sm_89 only) inside a Docker container.
# Produces build-cuda/bin/{llama, llama-cli, llama-server}.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="llama-cpp-cuda-build:local"
BUILD_DIR="build-cuda"

# 1. Build the CUDA build image (skipped fast if unchanged)
#    Context is plan/ only (the image has no COPY steps), to avoid tarring the whole repo (incl. .git).
docker build -t "$IMAGE" -f "$REPO_DIR/plan/Dockerfile.cuda" "$REPO_DIR/plan"

# 2. Configure (if needed) and build the host-scoped targets
docker run --rm --gpus all \
    -v "$REPO_DIR":/app \
    -w /app \
    "$IMAGE" \
    bash -c '
        set -euo pipefail
        cmake -B build-cuda -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=89 \
            -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TOOLS=ON \
            -DLLAMA_BUILD_SERVER=ON \
            -DLLAMA_BUILD_APP=ON \
            -DLLAMA_BUILD_UI=OFF \
            .
        cmake --build build-cuda --target llama-app llama-cli llama-server -j"$(nproc)"
    '

echo
echo "Build complete. Binaries:"
ls -l "$REPO_DIR/$BUILD_DIR/bin/" | grep -E 'llama(-cli|-server)?$' || true
