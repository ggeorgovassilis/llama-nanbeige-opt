# Memory Reduction Investigation (distilled findings)

This is a condensed summary of an experiment to reduce Nanbeige VRAM in llama.cpp.
The full experimental log, the KV probe, and the VRAM monitor live on the
`theory-b-kv-share` branch. This branch carries only the distilled conclusion.

## What the theory was

Nanbeige 4.2 is a looped transformer: 22 physical layers run in T = 2 passes for
44 effective block applications (see `02-nanbeige-architecture.md`).

The goal was to reduce VRAM "without cheating": keep full GPU offload
(`-ngl 99`), keep the context size (`-c 48000`), and keep the q8_0 KV cache type.

Initial hypothesis: llama.cpp allocates model weights twice for the looped
architecture, and that 2x is the dominant reclaimable cost.

## What was tried

1. Measured the buffer breakdown on the unmodified branch (`-lv 4`):

   | Buffer | CUDA0 |
   | --- | --- |
   | Model (weights) | 2177.95 MiB |
   | KV cache | 4394.50 MiB (44 layers) |
   | Compute | 330.50 MiB |

   Peak VRAM: 7028 MiB.

2. Weights are NOT doubled. The model buffer (2177.95 + 273.80 MiB mapped) equals
   the GGUF size, because `load_arch_tensors()` already aliases the 22 physical
   layers across both passes. The expert "weights allocated twice" hypothesis is
   refuted.

3. The real 2x is the KV cache: it is sized at `n_layer_all` = 44 layers, 2x a
   normal 22-layer model. This is 2x storage slots, not duplicated values: pass 1
   and pass 2 write genuinely different K/V.

4. Built a KV probe to measure cosine similarity between pass-1 and pass-2 K/V per
   layer. Result: moderate, not clean. Layer 0 is anti-correlated (K cosine
   ~-0.19), middle layers ~0.84-0.90.

5. Implemented "Theory B": alias pass-2 cache slots onto pass-1 using the existing
   `n_layer_kv_from_start` + `layer_reuse_cb` mechanism (same pattern as
   gemma3n/gemma4), halving the KV cache from 44 to 22 layers. This is the
   Ouro-style "final-pass cache reuse" approximation.

## What was found

- VRAM goal met: KV cache 4394.50 -> 2197.25 MiB, peak VRAM 7028 -> 4830 MiB
  (-31%), with no change to eval speed (~50 t/s).
- Quality gate FAILED: the model emits non-terminating gibberish instead of the
  correct answer. Pass-1 and pass-2 K/V are not interchangeable, worst at layer 0
  (raw embeddings vs fully-processed hidden state). The Ouro final-pass reuse
  approximation does not hold for this checkpoint.

## Conclusion

With the fixed constraints, the 2x KV cache is inherent to this checkpoint's
looped architecture and cannot be halved without destroying output quality. The
baseline 7028 MiB is the honest minimum under the "no cheating" rules.

The KV-sharing code change is experimental and must not be merged.
