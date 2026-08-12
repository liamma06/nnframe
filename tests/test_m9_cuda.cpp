#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "cuda/matmul_cuda.cuh"
#include <vector>
#include <random>

TEST_CASE("CUDA matmul matches scalar reference(64x64x64)"){
    const size_t M = 64, K = 64, N = 64;

    // Create random matrices

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> h_a(M * K);
    std::vector<scalar_t> h_b(K * N);
    for(auto& val : h_a) val = dist(rng);
    for(auto& val : h_b) val = dist(rng);

    std::vector<scalar_t> out_scalar(N * M, 0.0f);
    std::vector<scalar_t> out_cuda(N * M, 0.0f);

    //scalar matmul(reference)
    Tensor::matmul_scalar(h_a.data(), h_b.data(), out_scalar.data(), M, K, N);


    scalar_t *d_a, *d_b, *d_out;

    //(address of allocated mem, size in bytes)
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(scalar_t)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(scalar_t), cudaMemcpyHostToDevice));

    //cuda matmul
    matmul_cuda(d_a, d_b, d_out, M, K, N);

    //bring back to host
    CUDA_CHECK(cudaMemcpy(out_cuda.data(), d_out, M * N * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    //comapre results
    for (size_t i = 0; i < M * N; i++) {
    CHECK(out_cuda[i] == doctest::Approx(out_scalar[i]).epsilon(1e-3f));
    }

    //free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

TEST_CASE("CUDA matmul matches scalar reference, non-multiple-of-16 (50x50x50)"){
    const size_t M = 50, K = 50, N = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> h_a(M * K);
    std::vector<scalar_t> h_b(K * N);
    for(auto& val : h_a) val = dist(rng);
    for(auto& val : h_b) val = dist(rng);

    std::vector<scalar_t> out_scalar(N * M, 0.0f);
    std::vector<scalar_t> out_cuda(N * M, 0.0f);

    //scalar matmul(reference)
    Tensor::matmul_scalar(h_a.data(), h_b.data(), out_scalar.data(), M, K, N);

    scalar_t *d_a, *d_b, *d_out;

    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(scalar_t)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(scalar_t), cudaMemcpyHostToDevice));

    //cuda matmul
    matmul_cuda(d_a, d_b, d_out, M, K, N);

    //bring back to host
    CUDA_CHECK(cudaMemcpy(out_cuda.data(), d_out, M * N * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    //compare results
    for (size_t i = 0; i < M * N; i++) {
        CHECK(out_cuda[i] == doctest::Approx(out_scalar[i]).epsilon(1e-3f));
    }

    //free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}