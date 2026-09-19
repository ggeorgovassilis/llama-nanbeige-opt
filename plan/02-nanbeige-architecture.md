# Nanbeige Architecture: Looped Transformer (Looped Depth Sharing)

> Source: [Sebastian Raschka - LLM Architecture Gallery: Looped Depth Sharing](https://sebastianraschka.com/llm-architecture-gallery/looped-depth-sharing/)
> plus the in-tree implementation at `src/models/nanbeige.cpp`.

## 1. Concept

A **looped transformer** (also called **recurrent depth** or **looped depth sharing**) reuses the
same transformer block stack several times. Hidden states from one pass become the inputs to the
next pass, while the layer weights are shared.

This is a form of **depth-wise weight tying**. If a model has `L` distinct blocks and the stack is
applied `T` passes, the effective number of block applications is:

```
effective block applications = L * T
```

In a standard transformer `T = 1`; in a looped transformer `T > 1`.

### Trade-off

- **Parameter count** stays at `L` layers worth of weights (they are reused), so model weights are
  small relative to the effective depth.
- **Compute** grows with `T` (the forward pass runs through the stack `T` times).
- **Activation and KV-cache memory** also grow with `T`, depending on implementation.

So looped depth shifts the parameter-vs-computation trade-off: more compute per parameter.

## 2. Lineage

- **Universal Transformers** (Dehghani et al., 2018) - original idea: recurrent transition function
  with per-token adaptive halting.
- **Recurrent Depth** (Geiping et al., 2025) - unrolls a recurrent block to scale test-time compute.
- **Mixture-of-Recursions** (Bae et al., 2025) - token-level routing; different tokens get different
  recursion depths.
- **Ouro / LoopLM** (ByteDance, Zhu et al., 2025) - explicit looped LLM with learned exit gating.
- **Nanbeige 4.2** (2026) - the simplest variant: a fixed two-pass loop, no routing and no exit gate.

## 3. Nanbeige 4.2 specifics

- **Model size**: ~3B parameters (`Nanbeige4.2-3B`).
- **Physical depth**: 22 distinct transformer layers.
- **Passes**: `T = 2` - the hidden states pass through the same 22-layer stack **twice**.
- **Effective depth**: `22 * 2 = 44` block applications.
- **Looping policy**: fixed, unconditional two passes for every token (no per-token routing, no
  early-exit / halting - unlike Ouro).
- **Reported quality**: two passes gave the best trade-off, retaining ~75% of the token efficiency
  of a standard architecture; additional passes yielded little benefit and slowed training.

```
input -> [22 shared layers] -> loop_norm -> [same 22 shared layers] -> output_norm -> logits
              pass 1                                pass 2
```

The two passes share the exact same weights (the "green return path" re-enters the stack).

## 4. What is shared, and what is not

- **Shared (weights)**: all 22 layers' parameters are reused across passes. Parameter memory is that
  of 22 layers, not 44.
- **Not shared (KV cache)**: each layer-pass combination has its own KV cache entry. With 2 passes
  over 22 layers, KV-cache memory is that of 44 layers, i.e. 2x a standard 22-layer model.
- **Loop normalization**: a norm (RMS) is applied between passes (can be disabled).

Reference KV-cache sizing from the article (Ouro example): 4 passes x 48 layers x 16 KV heads x 128
head-dim = 1.5 MiB/token in bf16. The same principle scales Nanbeige's cache linearly with passes.

## 5. llama.cpp implementation (`src/models/nanbeige.cpp`)

The model is registered as `LLM_ARCH_NANBEIGE` and implements the generic `llama_model_base` hooks.

Key metadata keys (see `src/llama-arch.h` / `src/llama-arch.cpp`):

- `num_loops` (`LLM_KV_NUM_LOOPS`) - number of passes `T` (default 1).
- `skip_loop_final_norm` (`LLM_KV_SKIP_LOOP_FINAL_NORM`) - skip the between-pass norm.

Key implementation facts:

1. `load_arch_hparams()`:
   - Reads `n_loops` (asserted `>= 1`) and `n_layer_phys` (physical layer count = 22).
   - Bound-checks `n_layer_phys * n_loops <= LLAMA_MAX_LAYERS`.
   - **Expands the logical layer count** to `n_layer_all = n_layer_phys * n_loops` (44) by copying
     per-layer head/FFN/SWA/recr hyperparameters, so the rest of the runtime sees 44 logical layers.
2. `load_arch_tensors()`:
   - Declares tensors only for the physical layers (`n_phys` = 22).
   - **Shares the physical weights across loops** with `layers[i + j * n_phys] = layers[i]`.
   - `rope_freqs` is declared once and `TENSOR_DUPLICATED` for later layers.
3. `build_arch_graph()`:
   - Iterates `n_layer` (44) logical layers. Because weights are aliased, the second pass re-runs the
     same 22 physical layers.
   - Between passes (at `(il + 1) % n_phys == 0` and not the last layer) it inserts a `loop_norm`
     (`build_norm` with `output_norm`) unless `skip_loop_final_norm`.
   - Attention and FFN blocks are identical per layer to a standard dense transformer (RMS norm,
     GQA, RoPE, SILU parallel FFN).

### Memory implications (why this fork exists)

The current implementation already shares **weights** (so weight memory is already minimal), but the
**KV cache is NOT shared** - the comment in `load_arch_tensors()` states: "Share physical weights
across loops; **each slot still has its own KV index**."

Memory-reduction opportunities to explore:

1. **KV-cache reuse across passes.** The article notes Ouro can reuse only the final pass's cache
   during autoregressive decoding, cutting cache memory by the pass count with little quality loss.
   A similar strategy for Nanbeige could halve KV-cache memory (44 -> 22 layers of cache).
2. **Activation memory** during the two passes (trading memory for recompute).
3. **Reduced precision / streaming KV-cache** for the shared cache.

These are candidates for the fork's goal of reducing Nanbeige memory use; none are implemented yet.
