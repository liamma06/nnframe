#include "core/tensor.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// direct before/after comparison for M8: same matmul, scalar loop vs AVX2 intrinsics,
// no KVCache/attention involved so this isolates the SIMD speedup itself

namespace {
    std::vector<scalar_t> random_buffer(size_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<scalar_t> data(n);
        for (auto& v : data) v = dist(rng);
        return data;
    }
}

int main() {
    const size_t M = 512, K = 512, N = 512;
    const int trials = 20;

    std::mt19937 rng(42);
    std::vector<scalar_t> a = random_buffer(M * K, rng);
    std::vector<scalar_t> b = random_buffer(K * N, rng);
    std::vector<scalar_t> out_scalar(M * N);
    std::vector<scalar_t> out_avx2(M * N);

    // correctness check first: both paths must agree before timing means anything
    Tensor::matmul_scalar(a.data(), b.data(), out_scalar.data(), M, K, N);
    Tensor::matmul_avx2(a.data(), b.data(), out_avx2.data(), M, K, N);

    bool match = true;
    for (size_t i = 0; i < M * N; i++) {
        if (std::abs(out_scalar[i] - out_avx2[i]) > 1e-3f) { match = false; break; }
    }
    std::cout << "Correctness check (scalar vs AVX2): " << (match ? "PASS" : "FAIL") << "\n\n";

    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++)
        Tensor::matmul_scalar(a.data(), b.data(), out_scalar.data(), M, K, N);
    auto end_scalar = std::chrono::high_resolution_clock::now();

    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++)
        Tensor::matmul_avx2(a.data(), b.data(), out_avx2.data(), M, K, N);
    auto end_avx2 = std::chrono::high_resolution_clock::now();

    double scalar_ms = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count() / trials;
    double avx2_ms = std::chrono::duration<double, std::milli>(end_avx2 - start_avx2).count() / trials;

    std::cout << "matmul " << M << "x" << K << " * " << K << "x" << N << ", averaged over " << trials << " runs\n";
    std::cout << "Scalar: " << scalar_ms << " ms\n";
    std::cout << "AVX2:   " << avx2_ms << " ms\n";
    std::cout << "Speedup: " << (scalar_ms / avx2_ms) << "x\n";
}
