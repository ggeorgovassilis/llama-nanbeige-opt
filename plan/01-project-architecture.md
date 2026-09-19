# llama.cpp Project Architecture

> Fork: `llama-nanbeige-opt`
>
> Baseline: upstream `ggml-org/llama.cpp` at commit `60b06ab9a` (branch `llama-nanbeige-opt` == `master`).
> Reported version: `0.4.1-dev` (see `CMakeLists.txt`).
>
> The fork currently has **no local code changes**: its stated goal is to reduce the memory
> footprint of the Nanbeige LLM (see `02-nanbeige-architecture.md`).

## 1. Overview

llama.cpp is a plain C/C++ LLM (and VLM) inference framework with minimal dependencies. Its two
layers are:

1. **`ggml`** - a tensor / compute-graph library with multiple hardware backends and integer
   quantization support. It knows nothing about "language models".
2. **`llama.cpp`** (this repo) - everything on top of `ggml`: model loading (GGUF), per-architecture
   graph construction, KV-cache management, sampling, and the CLI/server tools.

Build and runtime flow:

```
GGUF file --load--> llama_model (tensors + hyperparams)
                       |
                       v
              llama_graph_context (per-arch forward graph)
                       |
                       v
              ggml compute graph (ggml_tensor ops)
                       |
                       v
         ggml backend(s): CUDA / Metal / Vulkan / CPU / ...
                       |
                       v
                    sampling -> tokens
```

## 2. Top-level directory layout

| Path | Purpose |
| --- | --- |
| `ggml/` | The ggml tensor library (vendored as a submodule / subtree). See section 3. |
| `include/` | Public API: `llama.h` (C API) and `llama-cpp.h` (C++ API). |
| `src/` | llama.cpp core implementation (the `llama` library). See section 4. |
| `common/` | Shared CLI infrastructure: argument parsing, chat/sampling, chat templates, parsers. |
| `tools/` | Executable programs: `cli`, `server`, `quantize`, `perplexity`, `llama-bench`, etc. |
| `app/` | The unified `llama` binary (`llama cli`, `llama serve`, ...). |
| `vendor/` | Third-party single-header / small libs: cpp-httplib, nlohmann-json, miniaudio, stb, etc. |
| `conversion/` | Python scripts to convert HF / other checkpoints to GGUF (`convert_hf_to_gguf.py`). |
| `gguf-py/` | Python GGUF reader/writer library. |
| `grammars/` | Example GBNF grammars for constrained decoding. |
| `tests/` | CTest test suite (C++ and Python helpers). |
| `examples/` | Minimal C/C++ API usage examples. |
| `pocs/` | Proof-of-concept programs. |
| `docs/` | Documentation (build guides, backend notes, development guides). |
| `cmake/` | CMake helper modules and cross-compile toolchains. |
| `.devops/` | Dockerfiles and packaging specs (cpu/cuda/rocm/...). |
| `models/` | Per-model GGUF metadata templates used by `convert_hf_to_gguf.py`. |
| `ci/`, `scripts/`, `requirements/`, `benches/` | CI, misc scripts, pip requirements, benchmarks. |

## 3. `ggml/` - the tensor library

Entry points: `ggml/include/ggml.h` (C API + tensor/op types) and `ggml/src/ggml.c` /
`ggml.cpp` (graph construction and CPU dispatch).

Subsystems under `ggml/src/`:

- **Core**: `ggml.c`, `ggml.cpp`, `ggml-impl.h`, `ggml-common.h`.
- **Quantization**: `ggml-quants.c/.h` (Q4_0 ... Q8_0, K-quants, i-quants, 1.5-6 bit).
- **Memory / scheduling**: `ggml-alloc.c` (tensor arena allocator), `ggml-backend*.cpp`
  (backend interface, registration, multi-backend meta-scheduler), `ggml-backend-dl.*`
  (dynamic backend loading), `ggml-threading.*`, `ggml-opt.cpp`.
- **Backends** (one directory each):
  - `ggml-cpu/` - CPU kernels (AVX/AVX2/AVX512/AMX/NEON/RVV; KleidiAI hooks).
  - `ggml-cuda/` - NVIDIA CUDA kernels (cuBLAS, FlashAttention, MMQ, CUDA graphs).
  - `ggml-metal/` - Apple Metal.
  - `ggml-vulkan/`, `ggml-sycl/` (Intel), `ggml-hip/` (AMD), `ggml-musa/` (Moore Threads),
    `ggml-opencl/` (Adreno), `ggml-cann/` (Ascend), `ggml-hexagon/` (Snapdragon),
    `ggml-webgpu/`, `ggml-openvino/`, `ggml-virtgpu/`, `ggml-rpc/` (remote),
    `ggml-blas/`, `ggml-zdnn/` (IBM zDNN), `ggml-zendnn/` (AMD ZenDNN), `ggml-et/`.
- **GGUF format**: `ggml/src/gguf.cpp`.

Backend selection is compile-time (CMake options such as `GGML_CUDA`, `GGML_METAL`,
`GGML_VULKAN`) and runtime (`--device`, `--list-devices`).

## 4. `src/` - the llama library

Builds the `llama` library from the public `include/llama.h` interface.

| Files | Responsibility |
| --- | --- |
| `llama-model-loader.cpp`, `llama-mmap.*` | Load GGUF, memory-map tensors, lazy loading. |
| `llama-model.cpp`, `llama-hparams.cpp`, `llama-cparams.cpp` | Model object, hyperparameters, context params. |
| `llama-arch.cpp/.h` | Architecture registry / metadata keys (`LLM_ARCH_*`, `LLM_KV_*`, `LLM_TENSOR_*`). |
| `models/models.h` + `models/*.cpp` | **One file per model architecture** (see section 5). |
| `llama-graph.cpp/.h` | Builds the ggml forward graph from an architecture description. |
| `llama-context.cpp/.h` | Inference context: state, batches, decoding. |
| `llama-batch.cpp/.h` | Batch (sequence) handling. |
| `llama-kv-cache*.cpp/.h` | KV-cache implementations (classic, SWA, MSA, DSA, hybrid, etc.). |
| `llama-memory*.cpp/.h` | Recurrent / hybrid model hidden-state memory (RWKV, Mamba, ...). |
| `llama-vocab.cpp/.h`, `unicode.*` | Tokenizer and vocab handling. |
| `llama-sampler.cpp/.h` | Sampling pipelines (top-k/top-p, min-p, temperature, ...). |
| `llama-quant.cpp/.h` | Quantization of tensors. |
| `llama-adapter.cpp`, `llama-chat.cpp`, `llama-grammar.cpp`, `llama-io.cpp` | LoRA adapters, chat helpers, grammars, I/O. |
| `llama-version.h.in` | Generated version header. |

## 5. Per-architecture model implementations (`src/models/`)

Each architecture is a class derived from `llama_model_base` (see `models/models.h`) and
implements three hooks:

- `load_arch_hparams()` - read architecture-specific metadata keys.
- `load_arch_tensors()` - declare and bind tensors to named GGUF tensors.
- `build_arch_graph()` - build the ggml forward graph (attention, FFN, rope, norm, ...).

There are 150+ files: `llama.cpp` (LLaMA 1/2/3), `qwen*.cpp`, `gemma*.cpp`, `mistral*.cpp`,
`mamba*.cpp`, `rwkv*.cpp`, `deepseek*.cpp`, **`nanbeige.cpp`**, and many more. Common sub-blocks
(attention, FFN, RMS norm, RoPE, MoE) are shared via `models.h` / `build_*` helpers in the graph
context so each file stays small.

## 6. `common/` - shared tooling

- `arg.*` - command-line argument definition / parsing.
- `sampling.*`, `chat.*` - sampling and chat orchestration.
- `chat-template*`, `jinja/` - chat templates and the dedicated Jinja engine.
- `parsers/`, `chat-peg-parser.*`, `chat-auto-parser.*` - PEG-based and auto-detected output parsers.
- `console.*`, `log.*`, `json.*`, `download.*`, `speculative.*`, `reasoning-budget.*`, `ngram-*`.
- `build-info.cmake` / `build-info.cpp.in` - injects git/version info at build time.

## 7. `tools/`, `app/`, and executable targets

Key executable targets (built from CMake):

| Target | Source dir | Binary | Notes |
| --- | --- | --- | --- |
| `llama-cli` | `tools/cli/` | `llama-cli` | Interactive / non-interactive chat CLI. |
| `llama-server` | `tools/server/` | `llama-server` | OpenAI-compatible HTTP server. |
| `llama` (target `llama-app`) | `app/` | `llama` | Unified binary: `llama cli`, `llama serve`, etc. |
| `llama-quantize` | `tools/quantize/` | `llama-quantize` | Quantize GGUF files. |
| `llama-perplexity` | `tools/perplexity/` | `llama-perplexity` | Perplexity evaluation. |
| `llama-bench` | `tools/llama-bench/` | `llama-bench` | Benchmarking. |
| `llama-imatrix`, `llama-tokenize`, `llama-gguf-split`, ... | `tools/` | ... | Various utilities. |

The `llama` (app) binary is a dispatcher that links the `llama-cli-impl`, `llama-server-impl`,
and other `*-impl` libraries.

## 8. Build system

- **CMake** is the primary build system (`CMakeLists.txt`, `cmake/*.cmake`, `CMakePresets.json`).
- A thin `Makefile` wraps CMake for convenience.
- Key CMake options (see `docs/build.md` for the full matrix):
  - Backends: `GGML_CUDA`, `GGML_METAL`, `GGML_VULKAN`, `GGML_SYCL`, `GGML_HIP`, `GGML_MUSA`, ...
  - `GGML_NATIVE` - compile CPU code for the build host (`-march=native`).
  - `CMAKE_CUDA_ARCHITECTURES` - which NVIDIA compute capabilities to emit (e.g. `89` for RTX 4000).
  - `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_TOOLS`, `LLAMA_BUILD_SERVER`, `LLAMA_BUILD_APP`,
    `LLAMA_BUILD_EXAMPLES`, `LLAMA_BUILD_UI`.
- Output layout: `build-*/bin/` (binaries + shared libs), CTest registered under `tests/`.

## 9. Testing

- CTest is enabled when `LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS` (default in standalone builds).
- Tests live in `tests/` (C++ unit tests, plus Python/shell helpers). Each test is both a CMake
  executable target and a CTest entry (`tests/CMakeLists.txt`).
- Run with `ctest --test-dir build-*`. See `03-build-and-run.md` for exact commands.
