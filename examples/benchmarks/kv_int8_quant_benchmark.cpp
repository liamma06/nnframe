#include "core/tensor.h"
#include "modules/char_model.h"
#include "infer/sampler.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// compares CharModel::generate() with a plain (float) KVBlockPool vs. an int8-quantized one:
//   time      : quantize/dequantize adds work per step, so this checks the real overhead
//   memory    : the actual point of quantization -- int8 storage vs float storage per KV slot

int main() {
    const size_t vocab_size = 50;
    const size_t embedding_dim = 64;
    const size_t num_heads = 4;
    const size_t num_transformer_blocks = 4;
    const size_t prompt_len = 8;
    const size_t max_new_tokens = 200;

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> token_dist(0, vocab_size - 1);

    std::vector<size_t> prompt;
    for (size_t i = 0; i < prompt_len; i++) prompt.push_back(token_dist(rng));

    CharModel model(vocab_size, embedding_dim, num_heads, num_transformer_blocks);
    Sampler sampler(42);

    auto start_plain = std::chrono::high_resolution_clock::now();
    auto out_plain = model.generate(prompt, sampler, max_new_tokens, 0.8f, 5, /*use_quantized_kv_cache=*/false);
    auto end_plain = std::chrono::high_resolution_clock::now();

    auto start_quant = std::chrono::high_resolution_clock::now();
    auto out_quant = model.generate(prompt, sampler, max_new_tokens, 0.8f, 5, /*use_quantized_kv_cache=*/true);
    auto end_quant = std::chrono::high_resolution_clock::now();

    double plain_ms = std::chrono::duration<double, std::milli>(end_plain - start_plain).count();
    double quant_ms = std::chrono::duration<double, std::milli>(end_quant - start_quant).count();

    // KV cache memory footprint per sequence, computed the same way KVBlockPool sizes its pool:
    // one K and one V buffer, each [num_heads, max_blocks * block_size, head_dim]
    size_t head_dim = embedding_dim / num_heads;
    size_t block_size = 16;
    size_t total_len = prompt_len + max_new_tokens;
    size_t max_blocks = (total_len + block_size - 1) / block_size;
    size_t total_slots = num_heads * max_blocks * block_size;

    size_t plain_bytes_per_block = total_slots * head_dim * sizeof(scalar_t) * 2; // K + V
    size_t quant_bytes_per_block = (total_slots * head_dim * sizeof(int8_t) + total_slots * sizeof(float)) * 2; // K + V, data + scales

    size_t plain_total_bytes = plain_bytes_per_block * num_transformer_blocks;
    size_t quant_total_bytes = quant_bytes_per_block * num_transformer_blocks;

    std::cout << "Tokens generated: " << max_new_tokens << "\n";
    std::cout << "Plain KV cache:     " << plain_ms << " ms\n";
    std::cout << "Quantized KV cache: " << quant_ms << " ms\n";
    std::cout << "Slowdown:           " << (quant_ms / plain_ms) << "x\n\n";

    std::cout << "KV cache memory (all transformer blocks, one sequence):\n";
    std::cout << "Plain (float32):    " << plain_total_bytes << " bytes (" << (plain_total_bytes / 1024.0) << " KB)\n";
    std::cout << "Quantized (int8):   " << quant_total_bytes << " bytes (" << (quant_total_bytes / 1024.0) << " KB)\n";
    std::cout << "Memory reduction:   " << (static_cast<double>(plain_total_bytes) / quant_total_bytes) << "x\n";
}
