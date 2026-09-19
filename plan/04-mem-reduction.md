# Memory Reduction - Nanbeige VRAM

Goal: reduce the VRAM the Nanbeige model consumes while running in llama.cpp, without
cheating. Track the investigation, measurements, and results here.

## Baseline

The reference run (measured against the unmodified `llama-nanbeige-opt` branch):

```sh
bash plan/run-cli.sh --reasoning-budget 100 \
  -p "Convert 3 to binary, shift left by two places, convert back to decimal, print the result" \
  -c 48000
```

| Metric | Value |
| --- | --- |
| VRAM peak | 7028 MiB of 8188 MiB (~86%) while the prompt runs |
| Prompt eval | 51.73 ms / 21 tokens (405.91 t/s) |
| Eval | 7501.15 ms / 374 runs (49.86 t/s) |
| Total | 7648.31 ms / 395 tokens |
| Graphs reused | 372 |

Timings vary with sampling (temperature 0.6); the VRAM peak and buffer sizes are deterministic.
Full measurement log: `plan/logs/baseline-*.log` (see "Recorded baseline" below).

## Goal

Reduce VRAM usage **without cheating**. Specifically the comparison must keep fixed:

- `-ngl 99` (full GPU offload; do not offload fewer layers)
- `-c 48000` (context size; do not shrink the cache)
- `-ctk q8_0 -ctv q8_0` (cache quantisation; do not lower it)

Small performance hits are acceptable. A temporary `-c` sweep is allowed purely as a
diagnostic to confirm which allocation dominates, but is not a result.

## Context

Nanbeige 4.2 is a **looped transformer**: 22 physical layers applied in `T = 2` passes for
44 effective block applications. See `plan/02-nanbeige-architecture.md` for the architecture
notes and the llama.cpp implementation details in `src/models/nanbeige.cpp`.

Initial hypothesis (from `02-nanbeige-architecture.md` and conversations with experts):
llama.cpp allocates model weights twice for the looped architecture.

Initial code review (to be confirmed by measurement, not assumed):

- Weights are **already aliased**: `load_arch_tensors()` in `src/models/nanbeige.cpp` shares
  physical tensors across passes with `layers[i + j*n_phys] = layers[i]`.
- The KV cache is sized from `hparams.n_layer_all` = 44 layers (`src/llama-kv-cache.cpp`),
  i.e. **2x** a standard 22-layer model, because each layer-pass has its own cache index.

So the expert "weights allocated twice" hypothesis may actually be the 2x KV cache. The
analysis below decides this with numbers.

## Analysis

Buffer breakdown from the `-lv 4` baseline (device `CUDA0`, 7807 MiB total):

| Buffer | CUDA0 | Host |
| --- | --- | --- |
| Model (weights) | 2177.95 MiB | 273.80 MiB (mapped) |
| KV cache | 4394.50 MiB | 0 |
| Compute (workspace) | 330.50 MiB | 59.09 MiB |
| Output | 0 | 0.63 MiB |

Device total = 2177.95 + 4394.50 + 330.50 = **6902 MiB** (matches the `-fit` projection).
Peak VRAM = **7028 MiB**.

Findings:

1. **Weights are NOT doubled.** Model buffer = 2177.95 + 273.80 = 2451.75 MiB ~ 2.39 GiB,
   which equals the GGUF file size (2.39 GiB). `load_arch_tensors()` already aliases the
   22 physical layers across both passes (`layers[i + j*n_phys] = layers[i]`).
2. **The KV cache is the 2x consumer.** It is sized at `n_layer_all` = 44 layers
   (`src/llama-kv-cache.cpp` line 102), giving 4394.50 MiB (K q8_0 2197.25 MiB + V q8_0
   2197.25 MiB). Halving to 22 layers saves ~2197 MiB.

   Important correction: this 2x is a count of **storage slots**, not **duplicated
   values**. Pass 1 layer 0 and pass 2 layer 0 write genuinely different K/V (see
   "Why the KV is not duplicated" below).
3. **Compute buffer is small** (330.50 MiB), not worth optimising first.

Conclusion: the expert "weights allocated twice" hypothesis is refuted. The dominant
allocation is the 2x KV cache. **Theory B (KV sharing, 44 -> 22) is the target.**

### Decision

Proceed with Theory B: make pass 2 (layers 22-43) reuse pass 1 (layers 0-21) cache slots,
via the existing `n_layer_kv_from_start` + `layer_reuse_cb` mechanism (same pattern as
`gemma3n`/`gemma4`). This halves the KV cache (~4394 -> ~2197 MiB), dropping device use
~6902 -> ~4705 MiB (~86% -> ~60% VRAM).

This is a semantic approximation (Ouro-style final-pass cache reuse): pass 2 shares cache
slots with pass 1, so pass 1 attention reads final-pass K/V instead of intermediate K/V.
It needs a quality check, not just a VRAM check.

## Why the KV is not duplicated

The 2x is storage slots, not duplicated content. The KV values across the two passes are
computed from different hidden states and are not expected to match:

- Pass 1, physical layer 0, position `p`: input is the token embedding (+ RoPE).
- Pass 2, physical layer 0, position `p`: input is the output of 22 full layers.

Same weights, same position, but a 22-layer-transformed input. No reason for equality.

So Theory B is **not deduplication**. It is **slot aliasing** via `layer_reuse_cb`:

- `get_k`/`get_v` (attention read) and `cpy_k`/`cpy_v` (K/V store) both route through
  `map_layer_ids.at(il)` in `src/llama-kv-cache.cpp`.
- `reuse(22) = 0` makes layer 22 read *and* write the same tensor as layer 0.
- Within one token's decode: layer 0 writes pass-1 K/V, then layer 22 overwrites it with
  pass-2 K/V. Net effect: only the final-pass K/V survives; both passes attend to it.

This is the **Ouro / LoopLM final-pass cache reuse** approximation, not a memory trick
that leaves computation intact. Pass 1's query now attends to pass-2 keys (a semantic
change), so quality must be validated, not assumed.

Why there is hope:

- **Precedent**: Ouro stores only the final loop's KV and shares it across iterations;
  Nanbeige 4.2 descends from that family.
- **Information content**: pass-2 K/V is the fully contextualized representation; pass-1
  K/V is a half-processed intermediate. Attending to a more refined key can be neutral.
- **Plumbing is proven**: `gemma3n`/`gemma4` already use `n_layer_kv_from_start` +
  `layer_reuse_cb` (for a different reason, but the machinery is battle-tested).

The risk: pass-1 queries were calibrated against pass-1 keys, so reuse is a
 train/inference mismatch. Whether *this* checkpoint tolerates it is empirical.

## Test plan (probe before branch)

Before writing the reuse callback, measure the actual relationship between the two
passes' K/V so the hope is grounded in data, not assumption:

1. Run a short fixed prompt (8-16 tokens) on the unmodified build.
2. After the graph executes, for each physical layer `i` in `[0, 21]`, read
   `kv_self.get_k_storage(i)` and `get_k_storage(i + 22)` (distinct in baseline), same for V.
3. At the last token position, compute **cosine similarity** and **relative L2 error**
   between layer `i` and `i + 22`, averaged over the 8 KV heads.
4. Report the distribution across the 22 layers.

Reading the result:

- **High cosine (~>0.9)**: pass-2 K/V is a refinement of pass-1; the approximation likely
  holds. The expected, likely case.
- **"Predictable" via a cheap transform**: a red herring for memory. Reconstructing
  pass-1 K/V from pass-2 costs a read/write transform that defeats the purpose; reuse
  does not reconstruct, it replaces.
- **Low cosine (~0.5)**: large mismatch; expect quality degradation, a stop signal.

Similarity is **diagnostic, not the gate**. The real gate is behavioral: does the
reused-cache build still give the correct answer on the target prompt and a couple of
held-out prompts, at acceptable tok/s.

## Theories

- **A - weight dedup**: refuted by measurement (weights already aliased).
- **B - KV sharing across passes (44 -> 22)**: reuse pass 1's cache slots in pass 2
  (Ouro final-pass reuse). Target; requires the probe + a quality check.
- **C - activation/workspace recompute**: lower priority, deeper ggml scheduler change.

## Intention

1. Implement the K/V probe and get the per-layer cosine similarity between pass 1 and
   pass 2 (Phase 4a).
2. If the similarity supports it, branch `theory-b-kv-share` and wire the reuse callback
   (Phase 4b).
3. Validate VRAM (expect ~4705 MiB device use, ~60%), tok/s, and answer quality on the
   target prompt plus held-out prompts (Phase 5).
4. If the similarity is low, stop and reconsider rather than shipping a degraded model.

## Recorded baseline

Command:

```sh
bash plan/vram-monitor.sh --tag baseline -- \
  bash plan/run-cli.sh --reasoning-budget 100 \
    -p "Convert 3 to binary, shift left by two places, convert back to decimal, print the result" \
    -c 48000 -lv 4
```

Logs:

- `plan/logs/baseline-20260919-142242.vram.log` - VRAM samples (0.5 s) + peak.
- `plan/logs/baseline-output.log` - llama output with `-lv 4` (buffer sizes).

Key lines:

- `print_info: n_layer = 44`, `n_layer_all = 44`, `n_embd_k_gqa = 1024`,
  `n_embd_v_gqa = 1024` (44 = 22 physical x 2 loops).
- `load_tensors: CUDA0 model buffer size = 2177.95 MiB`,
  `CPU_Mapped model buffer size = 273.80 MiB`.
- `llama_kv_cache: CUDA0 KV buffer size = 4394.50 MiB`,
  `size = 4394.50 MiB (48128 cells, 44 layers, 1/1 seqs)`.
- `sched_reserve: CUDA0 compute buffer size = 330.50 MiB`.

## Results

### Phase 4a: KV probe

Probe: `plan/kv-probe.cpp` + `plan/kv-probe.sh` + `plan/kv-similarity.py`. It loads the
model (`-ngl 99`, `q8_0` K/V), decodes a 10-token prompt, reads the raw K/V rows for all
44 logical layers via `llama_get_memory`, and dumps them to `plan/logs/kv-dump.bin`.
The script dequantizes `q8_0` (34-byte blocks: fp16 scale + 32 int8) and compares
physical layer `i` (pass 1) against `i + 22` (pass 2).

Cosine similarity between pass-1 and pass-2 K/V (over 10 positions):

- Overall K: mean `0.804`, min `-0.413`.
- Overall V: mean `0.632`, min `-0.087`.
- K by layer: layer 0 `-0.19` (uncorrelated); layer 1 `0.70`; layers 2-19 `0.84-0.90`;
  layers 20-21 `0.72-0.79`.
- V by layer: layer 0 `-0.03`; rises monotonically to layer 16 `0.79`, then tails off.

Reading:

- Middle layers (2-19) K is a refinement of pass-1, as hoped. V is weaker (`~0.65-0.70`).
- Boundary layers diverge: layer 0 is essentially uncorrelated (pass-1 layer 0 sees raw
  embeddings; pass-2 layer 0 sees the fully-processed hidden state). Layers 20-21 also
  drop.
- Per-position: position 0 is usually the *most* similar (0.96-0.99 for mid layers),
  position 1 the least (0.57-0.77); positions 2+ are uniform.

Conclusion: similarity is **moderate, not clean**. Reuse is a real train/inference
mismatch, worst at the first/last layers, not the near-identity we hoped for. This is
inconclusive for the Ouro-style approximation, so the go/no-go falls to the behavioral
test (Phase 4b branch + Phase 5 validation).

### Probe findings (implementation notes)

- `llama_kv_cache` exposes `get_k_storage(il)` but not the V side. Added
  `get_v_storage(int32_t il)` to `src/llama-kv-cache.{h,cpp}` (mirrors `get_k_storage`,
  returns `layers[ikv].v`) so the probe can dump both K and V.
- Dump format: magic `0x4b565052`, version 1, then `n_layer`, `n_pos`, then per-layer
  header (`il, k_type, k_nembd, k_row, v_type, v_nembd, v_row`) and raw row bytes.
- `q8_0` row size gotcha: `block_q8_0` is `{ ggml_half d; int8_t qs[32] }` = **34 bytes**
  per 32 values (fp16 scale, not fp32). First analyzer pass assumed 36 bytes and failed
  with `row size 1088 != 32*36`; corrected to 34 (1088 = 32*34).
- `attn_rot_k = attn_rot_v = 1` (head length 128 is a multiple of 64, so Hadamard
  rotation is active), but the stored K/V rows are still plain `q8_0`, so no rotation
  handling is needed in the analyzer.
- Probe totals: `KV buffer size = 46.75 MiB (512 cells, 44 layers)`, `K (q8_0): 23.38 MiB,
  V (q8_0): 23.38 MiB` - consistent with the baseline's 44-layer split.

### Recommendation

- Proceed to Phase 4b: branch `theory-b-kv-share`, wire `layer_reuse_cb` for
  `LLM_ARCH_NANBEIGE` so pass-2 layers (22-43) reuse pass-1 slots (0-21), halving KV
  from 44 to 22 layers. Expected device use ~4705 MiB (~60%).
- Gate the merge on Phase 5 behavioral tests (below), not on the cosine numbers alone.

### Open questions (need user decision before Phase 4b)

1. Proceed with the full-reuse branch despite weak layer 0 (and 20-21)? Optional
   fallback: share only middle layers (more invasive plumbing, likely not worth it).
2. Pass/fail criterion for Phase 5: correct answer on the target prompt + at least one
   of two held-out prompts, vs a stricter bar.

Resolved by user: "Agreed. proceed with the implementation" -> full reuse, gated on the
behavioral test.

## Phase 4b: implementation (branch `theory-b-kv-share`)

Three edits wire pass-2 layers (22-43) to reuse pass-1 cache slots (0-21):

- `src/models/nanbeige.cpp` (`load_arch_hparams`, inside the `n_loops > 1` block after
  `n_layer_all`): set `hparams.n_layer_kv_from_start = n_layer_phys;` so only pass 1
  "owns" KV slots.
- `src/llama-model.cpp` (`create_memory`, default branch): add a `LLM_ARCH_NANBEIGE`
  reuse callback after the `GEMMA3N`/`GEMMA4` block:

  ```cpp
  if (arch == LLM_ARCH_NANBEIGE && hparams.n_layer_kv_from_start >= 0) {
      reuse = [&](uint32_t il) {
          if (il >= (uint32_t) hparams.n_layer_kv_from_start) {
              return (int32_t) il - hparams.n_layer_kv_from_start;
          }

          return -1;
      };
  }
  ```

- `src/llama-model.cpp`: wire `reuse` into the `llama_kv_cache` constructor (the default
  branch's second-to-last argument, was `nullptr`).

Builds clean (`llama`, `llama-cli`, `llama-server`). This is the overwrite semantics
described in "Why the KV is not duplicated": layer 0 writes pass-1 K/V, layer 22
overwrites it with pass-2 K/V; only final-pass K/V survives.

## Phase 5: validation (behavioral gate)

Measured on the target prompt, same flags (`-ngl 99`, `-c 48000`, `-ctk q8_0 -ctv q8_0`):

| Metric | Baseline | Theory B | Delta |
| --- | --- | --- | --- |
| VRAM peak | 7028 MiB | 4830 MiB | -2198 MiB (-31%) |
| KV buffer | 4394.50 MiB (44 layers) | 2197.25 MiB (22 layers) | -2197.25 MiB |
| Eval speed | 49.86 t/s | 49.70 t/s | ~0 |
| Answer | `12` (correct) | gibberish | FAILED |

The VRAM goal is met exactly as predicted: `size = 2197.25 MiB (48128 cells, 22 layers,
1/1 seqs)`, peak 4830 MiB (~59% of the 8188 MiB device, down from ~86%). Tok/s is
unchanged (49.70 vs 49.86), so the reuse adds no measurable compute cost.

**Answer quality fails.** With the reuse active the model emits degenerate, non-stop
gibberish instead of `12`:

```text
Convert 3 to binary, shift left by two places, convert back to decimal, print the result. A pseud
s\\\\\\\\ffff HM\\\\ ... Gord\\\\ergen_gif ... nasa ... Gd Gord ... gif ... Dmicrosoft ...
```

The output never produces an end-of-stream token; without a `-n` token cap it runs until
the context is exhausted (the earlier 60 s "timeout" was this non-termination, not slow
generation - eval speed is unchanged at ~50 t/s).

### Why it failed

This is the stop signal the probe predicted. Layer 0 K cosine was `-0.19` (anti-correlated):
pass-1 layer 0 sees raw embeddings, pass-2 layer 0 sees the fully-processed hidden state,
so pass-1 queries attending to pass-2 keys collapse. The Ouro final-pass reuse
approximation does **not** hold for this checkpoint - the two passes compute genuinely
different K/V that is not interchangeable.

### Decision

**Theory B (full KV reuse) fails the behavioral gate and is rejected.** The branch
`theory-b-kv-share` is kept for reference but must not be merged. VRAM was reduced as
intended, but the model is no longer usable, so this is not a valid memory reduction.

No further reuse variant is worth pursuing: layer 0 is anti-correlated across passes,
which is structural (raw vs processed input), not a tuning detail. Sharing only "middle"
layers would still corrupt layers 0-1 and 20-21 and add invasive plumbing for no
principled gain.

Remaining honest options for this fork's goal (none implemented):

- A model-level change (train/inference loop that is KV-reuse-aware) - out of scope for
  a llama.cpp memory patch.
- Non-cheating reductions already ruled out by the fixed constraints (`-ngl 99`,
  `-c 48000`, `q8_0` cache): lowering any of them shrinks memory but is excluded.

Conclusion: with the stated constraints, the 2x KV cache is inherent to this checkpoint's
looped architecture and cannot be halved without destroying quality. The baseline 7028 MiB
is the honest minimum under the "no cheating" rules.
