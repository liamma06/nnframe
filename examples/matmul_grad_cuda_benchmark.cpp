#include "core/tensor.h"
#include "cuda/matmul_cuda.cuh"
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
    const size_t M = 512, K = 512, N = 512;
    const int trials = 20;

    std::mt19937 rng(42);
    std::vector<scalar_t> h_upstream = random_buffer(M * N, rng);
    std::vector<scalar_t> h_self = random_buffer(M * K, rng);
    std::vector<scalar_t> h_other = random_buffer(K * N, rng);

    scalar_t *d_upstream, *d_self, *d_other, *d_self_grad, *d_other_grad;
    CUDA_CHECK(cudaMalloc(&d_upstream, M * N * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_self, M * K * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_other, K * N * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_self_grad, M * K * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_other_grad, K * N * sizeof(scalar_t)));

    CUDA_CHECK(cudaMemcpy(d_upstream, h_upstream.data(), M * N * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_self, h_self.data(), M * K * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_other, h_other.data(), K * N * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_self_grad, 0, M * K * sizeof(scalar_t)));
    CUDA_CHECK(cudaMemset(d_other_grad, 0, K * N * sizeof(scalar_t)));

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < trials; t++) {
        matmul_grad_cuda(d_upstream, d_self, d_other, d_self_grad, d_other_grad, M, K, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaFree(d_upstream));
    CUDA_CHECK(cudaFree(d_self));
    CUDA_CHECK(cudaFree(d_other));
    CUDA_CHECK(cudaFree(d_self_grad));
    CUDA_CHECK(cudaFree(d_other_grad));

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / trials;
    std::cout << "matmul_grad_cuda " << M << "x" << K << "x" << N << ", averaged over " << trials << " runs: " << ms << " ms\n";
}
