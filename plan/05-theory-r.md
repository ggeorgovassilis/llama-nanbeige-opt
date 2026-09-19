# Theory R - Verify loop semantics against the reference

> Status: investigation, no code changes yet.
> Branch: `theory_r`.
> Precedes this doc: `plan/04-mem-reduction.md` (Theory B result).

## Motivation

Theory B (alias pass-2 KV slots onto pass-1, halving the cache 44 -> 22) reduced
VRAM from 7028 to 4830 MiB but broke the model: it emits non-terminating gibberish.

There are two possible root causes, and they lead to very different conclusions:

1. **The checkpoint was not trained for shared KV.** In that case the 44-layer
   cache is irreducible and we are done (the `04-mem-reduction.md` conclusion
   stands).
2. **llama.cpp's loop semantics diverge from the reference.** In that case the
   two passes' K/V are *artificially* different, and the anti-correlated layer 0
   the probe found is a symptom of a bug, not a property of the model. A small fix
   could make pass-1 and pass-2 K/V near-identical, after which Theory B sharing
   would work and recover ~2.2 GiB.

The evidence we have is genuinely ambiguous. The probe found layer 0 K
anti-correlated (-0.19), but that reading is downstream of whatever the graph
actually computes. If the graph applies the wrong RoPE positions, the wrong loop
norm, or the wrong KV indexing across passes, then pass-1 and pass-2 K/V diverge
for a fixable reason and the probe was measuring the bug.

Theory R is the cheap, no-code first step: diff llama.cpp's Nanbeige loop against
the reference implementation (HF/transformers or the official repo) and decide
which of the two cases is real.

## What the code already shows

Read from `src/models/nanbeige.cpp` (graph constructor) and the conversion path:

| Aspect | llama.cpp behavior | Looks right? |
| --- | --- | --- |
| RoPE positions | `inp_pos = build_inp_pos()` built once, shared by both passes | Yes - depth unrolling means same token = same position |
| RoPE type | `LLAMA_ROPE_TYPE_NORM` (`src/llama-model.cpp`) | To confirm against reference |
| Weights | `layers[i + j*n_phys] = layers[i]` aliases pass 2 onto pass 1 | Yes |
| KV slots | 44 separate slots (`n_layer_all = 44`) | **The open question** |
| Inter-pass norm | `loop_norm` after pass 1 (`il == 21`), reuses `output_norm` weights, unless `skip_loop_final_norm` | To confirm (weight reuse) |
| Final norm | `output_norm` (`result_norm`) at the very end | Yes |
| Residual | `inpL` carries across all 44 layers, no reset between passes | Yes |

Two things already look correct and narrow the search:

- **RoPE positions are shared across passes.** `inp_pos` is built once and passed
  to every `ggml_rope_ext` in the `for (il = 0; il < n_layer; ++il)` loop, so pass
  2 uses exactly pass 1's positions. The "different RoPE positions" hypothesis is
  likely refuted by reading, but should still be confirmed against the reference.
- **Weights are correctly aliased.** Pass 2 is the same 22 physical layers.

## The central open question

How does the reference implementation handle the KV cache across the two passes?

- **Shared (22 slots)**: the loop re-enters the same 22 layer modules, each with
  one KV cache entry. Pass 2 overwrites (or reads) pass 1's slots. This is the
  naive implementation of a looped module in transformers, and it means the model
  was *trained* with shared KV. Under this reading, llama.cpp's 44-slot cache is
  the divergence, and sharing should work once the semantics are reproduced
  exactly.
- **Separate (44 slots)**: the reference materializes two distinct caches, one per
  pass. Under this reading llama.cpp is correct and the 2x is irreducible.

A strong hint toward "shared": `conversion/nanbeige.py` reads `num_loops` and
`skip_loop_final_norm` straight out of the HF `config.json`, which means the
reference model is *defined as a loop* (22 layer modules, applied twice), not as 44
literal layers. A looped module in a standard transformers forward naturally shares
its per-layer `past_key_values` across passes.

This creates the paradox Theory R must resolve: if the reference shares KV and was
trained that way, why did our Theory B sharing produce gibberish? The likely answer
is that our sharing *semantics* were subtly wrong, not that sharing is wrong. See
"Reconcile with Theory B" below.

## What we need to pin down (checklist)

1. **KV semantics across passes.** Does the reference share per-layer KV slots
   between pass 1 and pass 2? When pass 2 runs, does it (a) overwrite pass 1's KV
   in place, (b) read pass 1's KV without writing, or (c) use a fresh second cache?
   This is the decisive question.

2. **RoPE positions across passes.** Confirm the reference gives each token the
   same position in both passes (no loop-index offset). Compare `rope_type`
   (NORM vs NONE) and any frequency scaling.

3. **Inter-pass norm.** Does the reference apply a norm at the loop boundary? Is it
   a dedicated weight, or the same tensor as the final `output_norm`? The GGUF only
   carries `output_norm`, and the graph reuses it for `loop_norm`; if the reference
   has a separate loop-boundary norm weight, that is a concrete divergence and a
   bug source.

4. **Residual and stream reset.** Confirm the residual stream is NOT reset between
   passes (pass 2's layer 0 sees pass 1's layer 21 output), and that nothing else
   (e.g. an implicit output projection) is inserted at the boundary.

5. **Causal masking / attention head layout.** Confirm GQA grouping (8 KV heads,
   48 Q heads) and the causal mask are identical across passes and match the
   reference.

## The decisive experiment: reconcile with Theory B

The reference semantics, once known, directly predict what the shared-cache
experiment should do:

- If the reference **overwrites** (pass 2 writes its own K/V into pass 1's slots
  and reads them back), then our Theory B (`reuse(il) = il - 22`) was already the
  right idea, and gibberish implies either an extra divergence (checklist items 2-5)
  or that the checkpoint truly was not trained for sharing.

- If the reference **reads without writing** (pass 2 attends to pass 1's K/V but
  keeps a separate or absent pass-2 cache), then our Theory B was wrong by design:
  it made both passes *overwrite* the shared slot, so pass 1 ended up attending to
  pass 2's keys for all past positions (the "mixed-space attention" failure
  described in `04-mem-reduction.md`). The correct implementation would instead let
  pass 1 own the slot and let pass 2 read it - a different reuse callback, possibly
  with pass 2's writes suppressed or diverted.

Either way, the reference tells us the exact semantics to reproduce before we
touch memory again.

## Experiment plan

1. **Locate the reference.** Pull the HF `config.json` and the modeling code for
   `Nanbeige/Nanbeige4.2-3B` (transformers `modeling_nanbeige.py` if it exists, or
   the official repo). If only the config is available, read the attention/loop
   flags from it; if the code is available, read the forward pass directly.

2. **Classify the KV semantics** (shared vs separate, overwrite vs read-only) and
   fill the checklist. Record the answer in this doc.

3. **Diff against `src/models/nanbeige.cpp`.** For every checklist item, mark
   "matches" or "diverges". Produce a table.

4. **Branch on the result:**
   - No divergence, reference uses separate KV: close Theory R, the 2x cache is
     confirmed irreducible, `04-mem-reduction.md` stands.
   - Divergence found: fix `nanbeige.cpp` (and/or the conversion) to match the
     reference, rebuild, re-run the KV probe. If pass-1/pass-2 K/V become
     near-identical, re-test Theory B sharing with the corrected semantics.

5. **Re-run the behavioral gate** on any fix before considering it validated:
   the target prompt plus at least one held-out prompt, with the VRAM monitor
   confirming the memory win.

   Gate prompts (short prompt first, to fail fast; long prompt only if the short
   one passes):

   - **Short (fail-fast)**: the binary-shift math prompt from
     `04-mem-reduction.md`. If this regresses, stop - do not advance.
   - **Long (real-world)**: run only if the short prompt passes. Feeds the full
     `README.md` into the context and asks which backends are supported. This
     exercises long-range attention across many cache rows, i.e. exactly where a
     shared-cache scheme is most likely to degrade.

     ```sh
     TEXT=$(cat README.md) time bash plan/run-cli.sh --reasoning-budget 1000 \
       -p "Your task: read the text below and tell me which backends are supported. Text: $TEXT" \
       --repeat-penalty 1.15 --repeat-last-n 128 -c 48000
     ```

     A passing result must name the correct backends and stay coherent across the
     long context (no repetition collapse, no gibberish), at an eval speed within
     a small factor of baseline. Confirm the VRAM win with the monitor on this run
     too, since the longer prompt fills more cache rows and is the true worst case
     for the shared-cache scheme.

## Open risks

- The reference modeling code may not be published (the model is recent). If only
  `config.json` is available, the KV semantics may have to be inferred from the
  Ouro/LoopLM paper (ByteDance, Zhu et al. 2025), which describes the canonical
  shared-KV loop, plus the config's `num_loops` / `skip_loop_final_norm` fields.
- A "corrected" sharing scheme may still not match the checkpoint if the model was
  trained with a detail we cannot observe from config alone.
