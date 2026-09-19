// Diagnostic probe for the Nanbeige looped-transformer KV cache.
// Runs one short prompt, then dumps the raw (quantized) K and V cache rows of every
// logical layer for the used cells to a binary file. A Python script dequantizes and
// compares pass-1 vs pass-2 layers (cosine similarity + relative L2).
//
// This file is a scratch diagnostic and is not part of the llama.cpp build.

#include "llama.h"
#include "llama-kv-cache.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const uint32_t DUMP_MAGIC = 0x4b565052u; // "KVPR"
static const uint32_t DUMP_VERSION = 1u;

static void usage(const char * prog) {
    fprintf(stderr, "usage: %s -m MODEL [-p PROMPT] [-o OUTFILE]\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox jumps over the lazy dog";
    std::string out_path = "kv-dump.bin";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99; // full offload, same as baseline

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    cparams.offload_kqv = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(512);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, false);
    if (n_tokens < 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    tokens.resize(n_tokens);

    fprintf(stderr, "prompt tokens: %d\n", n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // The KV cache is the memory module of a default (non-recurrent) context.
    llama_memory_t mem = llama_get_memory(ctx);
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) {
        fprintf(stderr, "memory is not a llama_kv_cache\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<uint32_t> layer_ids = kv->get_layer_ids();
    const uint32_t n_layer = (uint32_t) layer_ids.size();
    const uint32_t n_pos = (uint32_t) n_tokens;

    fprintf(stderr, "kv layers: %u, positions: %u\n", n_layer, n_pos);

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "failed to open output: %s\n", out_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    auto wr = [&](const void * p, size_t n) { out.write((const char *) p, n); };

    wr(&DUMP_MAGIC, sizeof(DUMP_MAGIC));
    wr(&DUMP_VERSION, sizeof(DUMP_VERSION));
    wr(&n_layer, sizeof(n_layer));
    wr(&n_pos, sizeof(n_pos));

    // gather tensors first, then write per-layer headers and data
    struct layer_dump {
        uint32_t il;
        ggml_tensor * k;
        ggml_tensor * v;
    };
    std::vector<layer_dump> layers;
    layers.reserve(n_layer);

    for (uint32_t il : layer_ids) {
        ggml_tensor * k = kv->get_k_storage((int32_t) il);
        ggml_tensor * v = kv->get_v_storage((int32_t) il);
        if (!k || !v) {
            fprintf(stderr, "layer %u missing K or V storage\n", il);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        layers.push_back({ il, k, v });

        uint32_t k_type = (uint32_t) k->type;
        uint32_t k_nembd = (uint32_t) k->ne[0];
        uint32_t k_row = (uint32_t) k->nb[1];
        uint32_t v_type = (uint32_t) v->type;
        uint32_t v_nembd = (uint32_t) v->ne[0];
        uint32_t v_row = (uint32_t) v->nb[1];

        wr(&il, sizeof(il));
        wr(&k_type, sizeof(k_type));
        wr(&k_nembd, sizeof(k_nembd));
        wr(&k_row, sizeof(k_row));
        wr(&v_type, sizeof(v_type));
        wr(&v_nembd, sizeof(v_nembd));
        wr(&v_row, sizeof(v_row));
    }

    std::vector<char> buf;

    for (const auto & ld : layers) {
        const size_t k_row = ld.k->nb[1];
        const size_t v_row = ld.v->nb[1];

        for (uint32_t p = 0; p < n_pos; ++p) {
            buf.resize(k_row);
            ggml_backend_tensor_get(ld.k, buf.data(), (size_t) p * k_row, k_row);
            out.write(buf.data(), k_row);

            buf.resize(v_row);
            ggml_backend_tensor_get(ld.v, buf.data(), (size_t) p * v_row, v_row);
            out.write(buf.data(), v_row);
        }
    }

    out.close();
    fprintf(stderr, "wrote %s\n", out_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
