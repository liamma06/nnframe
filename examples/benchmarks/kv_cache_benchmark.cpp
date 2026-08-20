#include "core/tensor.h"
#include "modules/attention.h"
#include "infer/kv_block_pool.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// compares two ways of generating a sequence of tokens through SelfAttention:
//   no-cache : each new token recomputes attention over the ENTIRE sequence so far (O(n^2) total)
//   with KV cache : each new token only computes attention using its own row + the cached K/V (O(n) total)

namespace {
    TensorPtr random_token(size_t embed_dim, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<scalar_t> row;
        for (size_t j = 0; j < embed_dim; j++) row.push_back(dist(rng));
        return std::make_shared<Tensor>(std::vector<size_t>{1, embed_dim}, row);
    }

    TensorPtr append_row(const TensorPtr& seq, const TensorPtr& new_row, size_t embed_dim) {
        size_t old_len = seq->shape()[0];
        std::vector<scalar_t> data;
        for (size_t i = 0; i < old_len; i++)
            for (size_t j = 0; j < embed_dim; j++)
                data.push_back(seq->at({i, j}));
        for (size_t j = 0; j < embed_dim; j++)
            data.push_back(new_row->at({0, j}));
        return std::make_shared<Tensor>(std::vector<size_t>{old_len + 1, embed_dim}, data);
    }
}

int main() {
    const size_t embed_dim = 32;
    const size_t num_heads = 4;
    const size_t num_tokens = 200;

    std::mt19937 rng(42);
    SelfAttention attn(embed_dim, num_heads);

    // pre-generate the same token sequence so both runs do identical work
    std::vector<TensorPtr> tokens;
    for (size_t i = 0; i < num_tokens; i++)
        tokens.push_back(random_token(embed_dim, rng));

    // --- no cache: recompute full attention over the growing sequence every step ---
    auto start_no_cache = std::chrono::high_resolution_clock::now();
    TensorPtr seq = tokens[0];
    attn.forward(seq);
    for (size_t i = 1; i < num_tokens; i++) {
        seq = append_row(seq, tokens[i], embed_dim);
        attn.forward(seq);
    }
    auto end_no_cache = std::chrono::high_resolution_clock::now();

    // --- with KV cache: each step only processes the single new token ---
    const size_t block_size = 16;
    const size_t max_blocks = (num_tokens + block_size - 1) / block_size;
    const size_t sequence_id = 0;

    auto start_cache = std::chrono::high_resolution_clock::now();
    KVBlockPool pool(num_heads, embed_dim / num_heads, max_blocks, block_size);
    attn.forward(tokens[0], pool, sequence_id);
    for (size_t i = 1; i < num_tokens; i++) {
        attn.forward(tokens[i], pool, sequence_id);
    }
    auto end_cache = std::chrono::high_resolution_clock::now();

    double no_cache_ms = std::chrono::duration<double, std::milli>(end_no_cache - start_no_cache).count();
    double cache_ms = std::chrono::duration<double, std::milli>(end_cache - start_cache).count();

    std::cout << "Tokens generated: " << num_tokens << "\n";
    std::cout << "No cache:   " << no_cache_ms << " ms\n";
    std::cout << "With cache: " << cache_ms << " ms\n";
    std::cout << "Speedup:    " << (no_cache_ms / cache_ms) << "x\n";
}
