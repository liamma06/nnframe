#include "core/tensor.h"
#include "optim/adamw.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

namespace {
    std::vector<scalar_t> random_buffer(size_t n, std::mt19937& rng, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        std::vector<scalar_t> data(n);
        for (auto& v : data) v = dist(rng);
        return data;
    }
}

int main() {
    const size_t n = 768 * 3072; // one GPT-2 small MLP weight matrix
    const scalar_t max_norm = 1.0f;
    const int trials = 200;

    std::mt19937 rng(42);
    std::vector<scalar_t> param_vals = random_buffer(n, rng, -1.0f, 1.0f);
    std::vector<scalar_t> grad_vals = random_buffer(n, rng, -10.0f, 10.0f);

    // fused: step(max_norm), scratch buffer allocated once in AdamW's constructor
    TensorPtr param_fused = Tensor::from_vector(param_vals)->to(Device::CUDA);
    param_fused->set_requires_grad(true);
    param_fused->init_grad();
    CUDA_CHECK(cudaMemcpy(param_fused->grad().mutable_device_data(), grad_vals.data(), n * sizeof(scalar_t), cudaMemcpyHostToDevice));
    AdamW opt_fused({param_fused}, 0.01f);

    auto start_fused = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++) {
        opt_fused.step(max_norm);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_fused = std::chrono::high_resolution_clock::now();

    // unfused: clip_grad_norm then plain step()
    TensorPtr param_unfused = Tensor::from_vector(param_vals)->to(Device::CUDA);
    param_unfused->set_requires_grad(true);
    param_unfused->init_grad();
    CUDA_CHECK(cudaMemcpy(param_unfused->grad().mutable_device_data(), grad_vals.data(), n * sizeof(scalar_t), cudaMemcpyHostToDevice));
    AdamW opt_unfused({param_unfused}, 0.01f);

    auto start_unfused = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++) {
        clip_grad_norm({param_unfused}, max_norm);
        opt_unfused.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_unfused = std::chrono::high_resolution_clock::now();

    double fused_ms = std::chrono::duration<double, std::milli>(end_fused - start_fused).count() / trials;
    double unfused_ms = std::chrono::duration<double, std::milli>(end_unfused - start_unfused).count() / trials;

    std::cout << "grad-clip + AdamW, n=" << n << ", averaged over " << trials << " runs\n";
    std::cout << "Fused (step(max_norm)): " << fused_ms << " ms\n";
    std::cout << "Unfused (clip_grad_norm + step()): " << unfused_ms << " ms\n";
    std::cout << "Speedup: " << (unfused_ms / fused_ms) << "x\n";
}
