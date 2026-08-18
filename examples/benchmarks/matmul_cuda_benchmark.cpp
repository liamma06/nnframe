#include "core/tensor.h"
#include "cuda/matmul_cuda.cuh"
#include "cuda/mempool.cuh"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// timing only (correctness already proven in tests/test_m9_cuda.cpp) --
// reports kernel-only compute time separately from full alloc+transfer+compute+transfer+free
// time, so it's visible whether GPU wall-clock is dominated by compute or by PCIe transfer.

namespace {
    std::vector<scalar_t> random_buffer(size_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<scalar_t> data(n);
        for (auto& v : data) v = dist(rng);
        return data;
    }
}

int main() {
    init_cuda_mempool();

    const size_t M = 512, K = 512, N = 512;
    const int trials = 20;

    std::mt19937 rng(42);
    std::vector<scalar_t> h_a = random_buffer(M * K, rng);
    std::vector<scalar_t> h_b = random_buffer(K * N, rng);
    std::vector<scalar_t> h_out(M * N);

    // --- full round trip: alloc + upload + compute + download + free, every trial ---
    auto start_full = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++) {
        scalar_t *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMallocAsync(&d_a, M * K * sizeof(scalar_t), 0));
        CUDA_CHECK(cudaMallocAsync(&d_b, K * N * sizeof(scalar_t), 0));
        CUDA_CHECK(cudaMallocAsync(&d_out, M * N * sizeof(scalar_t), 0));

        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(scalar_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(scalar_t), cudaMemcpyHostToDevice));

        matmul_cuda(d_a, d_b, d_out, M, K, N);

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, M * N * sizeof(scalar_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFreeAsync(d_a, 0));
        CUDA_CHECK(cudaFreeAsync(d_b, 0));
        CUDA_CHECK(cudaFreeAsync(d_out, 0));
    }
    auto end_full = std::chrono::high_resolution_clock::now();

    // --- kernel-only: upload once, time just the launches, download once at the end ---
    scalar_t *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(scalar_t)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(scalar_t), cudaMemcpyHostToDevice));

    auto start_kernel = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++) {
        matmul_cuda(d_a, d_b, d_out, M, K, N);
    }
    auto end_kernel = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, M * N * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    double full_ms = std::chrono::duration<double, std::milli>(end_full - start_full).count() / trials;
    double kernel_ms = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count() / trials;

    std::cout << "CUDA matmul " << M << "x" << K << " * " << K << "x" << N << ", averaged over " << trials << " runs\n";
    std::cout << "Kernel-only (compute, data already on GPU): " << kernel_ms << " ms\n";
    std::cout << "Full round trip (alloc+upload+compute+download+free): " << full_ms << " ms\n";
    std::cout << "Transfer/alloc overhead: " << (full_ms - kernel_ms) << " ms (" << (100.0 * (full_ms - kernel_ms) / full_ms) << "% of full round trip)\n";
}
