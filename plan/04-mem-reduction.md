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

## Next theories

Two more theories are under consideration. Outlined here; tested on their own
branches.

### Theory R - verify against the reference implementation

The Ouro article claims final-pass cache reuse works "with little quality loss",
but Theory B produced gibberish. Either the checkpoint was not trained for shared
KV (the 44-layer cache is irreducible), or llama.cpp's loop semantics diverge from
the reference and the anti-correlated layer 0 is a bug, not a property of the
model.

Hypothesis: the loop is depth unrolling, so pass 1 and pass 2 must use identical
RoPE positions. If `src/models/nanbeige.cpp` (or the nanbeige path in
`convert_hf_to_gguf.py`) gives pass 2 different positions, or misplaces `loop_norm`
/ the residual, then pass-1 and pass-2 K/V are artificially different and sharing
fails for the wrong reason.

Plan: diff `src/models/nanbeige.cpp` and `convert_hf_to_gguf.py` against the HF /
transformers Nanbeige 4.2 reference; confirm pass 2 uses pass 1's positions. If a
divergence exists, a small fix could make the caches near-identical and sharing
"just works".

### Theory E - selective KV sharing

The probe gave a per-layer safety map rather than a single verdict:

- Layer 0: K cosine ~-0.19 (anti-correlated, keep separate)
- Layer 1: ~0.70 (borderline)
- Layers 2-19: ~0.84-0.90 (safe to share)
- Layers 20-21: ~0.72-0.79 (risky, feed the loop-carried state)

Theory B shared all 22 and failed because layer 0 poisoned pass 1's input, which
propagated into pass 2. The next test is to share only the safe middle and keep the
boundary layers with their own pass-2 slots.

Variants:

- Threshold: keep pass-2 physical layers 0-1 separate, share 2-21. Saves ~20
  layers (~2.0 GiB, KV 4394 -> ~2400 MiB).
- Bitmask (matches the probe map): keep 0, 1, 20, 21 separate, share 2-19. Saves
  ~18 layers (~1.8 GiB).

Plumbing is a small extension of the existing `has_kv` / `layer_reuse_cb`
mechanism. Expected peak VRAM ~5100-5300 MiB. Risk: cosine is diagnostic, not a
guarantee; corruption compounds through the loop.
