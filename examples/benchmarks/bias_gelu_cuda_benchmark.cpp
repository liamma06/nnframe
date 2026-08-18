#include "core/tensor.h"
#include "cuda/elementwise_cuda.cuh"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

namespace {
    std::vector<scalar_t> random_buffer(size_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<scalar_t> data(n);
        for (auto& v : data) v = dist(rng);
        return data;
    }
}

int main() {
    const size_t batch_size = 64, features = 3072; // GPT-2 small MLP hidden dim
    const size_t n = batch_size * features;
    const int trials = 50;

    std::mt19937 rng(42);
    std::vector<scalar_t> h_input = random_buffer(n, rng);
    std::vector<scalar_t> h_bias = random_buffer(features, rng);

    scalar_t *d_input, *d_bias, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_bias, features * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), features * sizeof(scalar_t), cudaMemcpyHostToDevice));

    // fused: one kernel
    auto start_fused = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++) {
        bias_gelu_cuda(d_input, d_bias, d_out, batch_size, features);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_fused = std::chrono::high_resolution_clock::now();

    // unfused: broadcast bias into a full [batch_size, features] buffer, add, then gelu
    scalar_t* d_bias_full = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bias_full, n * sizeof(scalar_t)));
    std::vector<scalar_t> h_bias_full(n);
    for (size_t i = 0; i < n; i++) h_bias_full[i] = h_bias[i % features];
    CUDA_CHECK(cudaMemcpy(d_bias_full, h_bias_full.data(), n * sizeof(scalar_t), cudaMemcpyHostToDevice));

    auto start_unfused = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++) {
        add_cuda(d_input, d_bias_full, d_out, n);
        gelu_cuda(d_out, d_out, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_unfused = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_bias_full));
    CUDA_CHECK(cudaFree(d_out));

    double fused_ms = std::chrono::duration<double, std::milli>(end_fused - start_fused).count() / trials;
    double unfused_ms = std::chrono::duration<double, std::milli>(end_unfused - start_unfused).count() / trials;

    std::cout << "bias+GELU, batch=" << batch_size << " features=" << features << ", averaged over " << trials << " runs\n";
    std::cout << "Fused (bias_gelu_cuda, 1 launch): " << fused_ms << " ms\n";
    std::cout << "Unfused (add_cuda + gelu_cuda, 2 launches): " << unfused_ms << " ms\n";
    std::cout << "Speedup: " << (unfused_ms / fused_ms) << "x\n";
}
