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

## Follow-up theories (resolved)

Two follow-ups were considered after Theory B failed. Both are now closed; the
full writeup and the reference diff live on the `theory_r` branch.

### Theory R - verify against the reference implementation (RESOLVED: no divergence)

Question: does llama.cpp's loop semantics match the HF reference, or is the
anti-correlated layer 0 a bug that made sharing fail for the wrong reason?

Result: **llama.cpp matches the reference exactly.** The HF modeling code
(`modeling_nanbeige.py`) uses `_get_loop_cache_layer_idx(layer_idx, loop_idx,
num_hidden_layers) = layer_idx + loop_idx * num_hidden_layers`, i.e. 44 separate KV
slots (pass 1 -> 0-21, pass 2 -> 22-43). RoPE positions, the inter-pass norm, the
final norm, the residual stream, and weight aliasing all match `nanbeige.cpp`.

The reference *has* a `loop_share_kv` feature, but it requires
`enable_double_loop_split=True` (a separate architecture variant), and both are
`False` in `Nanbeige4.2-3B`. It is not a free inference-side switch - it must be
trained with the flag on. Theory B's sharing was exactly this flag enabled on a
model that never saw it, which is why it gibberished.

Conclusion: no divergence, no fix. The 44-slot cache is reference-faithful and
correct.

### Theory E - selective KV sharing (RESOLVED: failed empirically)

The probe suggested sharing only the "safe" middle layers (cosine 0.84-0.90) and
keeping the boundary layers separate. Theory R did not resolve this (it answered a
different, correctness question), so it was tested directly on branch `theory_e`.

Implementation: pass-2 physical layers 0-1 keep their own KV slots; pass-2 layers
2-21 alias pass-1 slots (`n_layer_kv_from_start = 24` plus a selective reuse
callback). KV cache 4394.50 -> 2397 MiB (24 layers), VRAM ~7028 -> ~5030 MiB.

Result: FAILED. The short gate ("Convert 3 to binary, shift left by two places,
convert back to decimal, print the result", answer 12) still emits non-terminating
gibberish instead of "12". Protecting the two worst layers is not enough: the
corruption is asymmetric (pass 1 reads pass-2 past keys in the shared slots) and
compounds across the 20 shared layers and the loop, so even the "similar" middle
cannot be shared.

Conclusion: selective sharing is refuted by measurement, same as full sharing. The
0.84-0.90 cosine is a weak prior - attention amplifies small key differences and
compounds them across the loop. The 44-slot cache is required for this checkpoint.

Note: the reference's own `loop_share_kv` uses read-only semantics (pass 2 reads
pass 1 KV, no overwrite) and needs `enable_double_loop_split=True`. That is a
different, trained behaviour, not the overwrite aliasing tested here.

## Final conclusion

All four theories are closed:

- **A (weights doubled)**: refuted by measurement (weights already aliased).
- **B (full KV sharing)**: halves VRAM but destroys output; the model was not
  trained for shared KV.
- **R (llama.cpp diverges from reference)**: refuted by diff; llama.cpp is correct.
- **E (selective KV sharing)**: saves ~2 GiB VRAM but still destroys output,
  measured on the short gate.

The 2x KV cache is an inherent, reference-faithful cost of this checkpoint's
looped architecture. Under the fixed constraints (`-ngl 99`, `-c 48000`, `q8_0`
cache), the baseline **7028 MiB is the honest minimum**. The only honest levers are
lower cache quant, smaller context, or a checkpoint trained with shared KV - none
available under the "no cheating" rules.
