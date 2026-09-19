# Build & Run Instructions

Environment captured for this host:

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04 (x86-64) |
| CPU | 16 cores, 62 GiB RAM |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB VRAM |
| Compute capability | `8.9` (Ada Lovelace, `sm_89`) |
| Driver | `595.91.07` (CUDA 13.2) |
| Host toolchain | gcc 13.3, cmake 3.28.3 (no `nvcc`, no `ninja` on host) |
| Container runtime | Docker 29.8 + nvidia-container-toolkit (verified `--gpus all`) |

## 1. Strategy

`nvcc` is **not** installed on the host. To avoid polluting host dependencies we build **inside a
Docker container** using the official `nvidia/cuda` devel image (CUDA 12.8.1 is the llama.cpp
default and fully supports `sm_89`). The workspace is bind-mounted, so the binaries land on the host
filesystem under `build-cuda/bin/`.

Builds are **scoped to this host only**:

- `-DCMAKE_CUDA_ARCHITECTURES=89` -> emit CUDA code only for `sm_89` (no all-arch / PTX bloat).
- `-DGGML_NATIVE=ON` -> `-march=native` CPU codegen (this is the default; stated explicitly).

## 2. One-command build

```sh
bash plan/build.sh
```

`plan/build.sh`:

1. Builds the build image from `plan/Dockerfile.cuda` (tag `llama-cpp-cuda-build:local`).
2. Runs `cmake` configure + `cmake --build` inside the container (GPU passed through with `--gpus all`).
3. Produces `build-cuda/bin/llama`, `build-cuda/bin/llama-cli`, `build-cuda/bin/llama-server`.

### Manual build (equivalent commands)

```sh
docker build -t llama-cpp-cuda-build:local -f plan/Dockerfile.cuda .

docker run --rm --gpus all \
  -v "$(pwd)":/app -w /app \
  llama-cpp-cuda-build:local \
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

docker run --rm --gpus all \
  -v "$(pwd)":/app -w /app \
  llama-cpp-cuda-build:local \
  cmake --build build-cuda --target llama-app llama-cli llama-server -j"$(nproc)"
```

> Note: the unified `llama` binary is the CMake target `llama-app` (see `app/CMakeLists.txt`).
> Build artifacts are written by the container as `root`. If you need to modify/rebuild
> natively later, run `sudo chown -R "$USER" build-cuda`.

## 3. Running

Models live under `/home/george/apps/llamacpp/models` (mostly symlinks into `/mnt/crypt_data/llm`
and `/mnt/plain_data/llm`). The run scripts resolve each model symlink with `realpath` and mount the
**single resolved file** into the container (same pattern as `/home/george/apps/llamacpp/llama-server`),
instead of mounting the whole encrypted FUSE tree.

```sh
# One-shot completion (defaults to the Nanbeige model)
bash plan/run-cli.sh -p "The capital of France is" -n 32

# OpenAI-compatible server on http://localhost:8080
bash plan/run-server.sh
```

The Nanbeige model (`Nanbeige4.2-3B-Q4_K_M.gguf`, ~2.6 GiB) fully offloads to the 8 GiB GPU
(`-ngl 99`). Both scripts apply the host-tuned flags:

- `-ctk q8_0 -ctv q8_0` - quantise the K/V cache to 8-bit (faster on this GPU).
- `--parallel 1` - single server slot (server only).
- `--shm-size 2gb` - avoid CUDA shared-memory OOM (Docker default is 64 MB).
- `--ulimit memlock=-1:-1` - allow pinned memory.
- `-e CUDA_DEVICE_SCHEDULE=4 -e OMP_WAIT_POLICY=PASSIVE` - same as the reference setup.

### One-shot completion (correct tool)

`llama-cli` is an **interactive** chat REPL: it prints its `> ` prompt in a loop. Running it without a
TTY (or with `-p` on a model that has a chat template) makes it spin at 100% CPU spamming `> ` on
EOF stdin - this is the "dockerd at 190% CPU, container spamming `>`" symptom, not a CUDA problem.

For scripted generation use the **`llama completion` subcommand** (registered under
`LLAMA_EXAMPLE_COMPLETION`, which is what actually accepts `-no-cnv`):

```sh
MODEL_REAL="$(realpath /home/george/apps/llamacpp/models/Nanbeige4.2-3B-Q4_K_M.gguf)"
docker run --rm --gpus all --shm-size 2gb --ulimit memlock=-1:-1 \
  -e CUDA_DEVICE_SCHEDULE=4 -e OMP_WAIT_POLICY=PASSIVE \
  -v "$(pwd)":/app \
  -v "${MODEL_REAL}:/models/Nanbeige4.2-3B-Q4_K_M.gguf:ro" \
  -w /app llama-cpp-cuda-build:local \
  /app/build-cuda/bin/llama completion \
    -m /models/Nanbeige4.2-3B-Q4_K_M.gguf -ngl 99 -ctk q8_0 -ctv q8_0 \
    -no-cnv -p "The capital of France is" -n 32 -s 42
```

Verified output: `The capital of France is Paris.` (load ~1.1 s, ~47 tok/s).

### Manual run (server)

```sh
MODEL_REAL="$(realpath /home/george/apps/llamacpp/models/Nanbeige4.2-3B-Q4_K_M.gguf)"
docker run --rm --gpus all -p 8080:8080 --shm-size 2gb --ulimit memlock=-1:-1 \
  -e CUDA_DEVICE_SCHEDULE=4 -e OMP_WAIT_POLICY=PASSIVE \
  -v "$(pwd)":/app \
  -v "${MODEL_REAL}:/models/Nanbeige4.2-3B-Q4_K_M.gguf:ro" \
  -w /app llama-cpp-cuda-build:local \
  /app/build-cuda/bin/llama-server \
    -m /models/Nanbeige4.2-3B-Q4_K_M.gguf -ngl 99 -ctk q8_0 -ctv q8_0 \
    --parallel 1 --host 0.0.0.0 --port 8080
```

Then verify:

```sh
curl -s http://localhost:8080/health
# {"status":"ok"}

curl -s http://localhost:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of Japan is","n_predict":24,"temperature":0.0}'
```

Verified: `Tokyo.` at ~45 tok/s (`/completion` returns `"stop":true`).

## 4. Running tests

Tests are registered by CTest (`LLAMA_BUILD_TESTS=ON` is set in `build.sh`). The test binaries are
not built by the fast target list above; build them first, then run CTest inside the container.

```sh
# Build all test binaries (slow first time: test-backend-ops compiles many kernels)
docker run --rm --gpus all -v "$(pwd)":/app -w /app llama-cpp-cuda-build:local \
  cmake --build build-cuda -j"$(nproc)"

# Run the full suite (inside the container, since it links CUDA libs)
docker run --rm --gpus all -v "$(pwd)":/app -w /app/build-cuda llama-cpp-cuda-build:local \
  ctest --output-on-failure
```

Quick smoke test (single test target, no GPU needed):

```sh
docker run --rm -v "$(pwd)":/app -w /app llama-cpp-cuda-build:local \
  bash -c 'cmake --build build-cuda --target test-tokenizer-0 -j"$(nproc)" && ./build-cuda/bin/test-tokenizer-0'
```

## 5. Native host build (no CUDA) - optional

For a CPU-only binary that runs directly on the host (no Docker at runtime):

```sh
cmake -B build-cpu -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build-cpu -j"$(nproc)" --target llama-cli llama-server
```

## 6. Rebuilds / iteration

- Re-run `bash plan/build.sh` after source changes (CMake + Ninja are incremental).
- Add `ccache` for faster repeat builds: install it in `plan/Dockerfile.cuda` and mount a persistent
  cache dir (`-v ~/.cache/llama-ccache:/root/.cache/ccache`).
