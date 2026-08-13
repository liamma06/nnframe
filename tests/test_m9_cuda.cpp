#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "cuda/matmul_cuda.cuh"
#include <vector>
#include <random>
#include <stdexcept>

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

TEST_CASE("Tensor::matmul on GPU-resident tensors matches CPU Tensor::matmul"){
    const size_t M = 32, K = 16, N = 24;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> a_vals(M * K);
    std::vector<scalar_t> b_vals(K * N);
    for (auto& v : a_vals) v = dist(rng);
    for (auto& v : b_vals) v = dist(rng);

    // build CPU tensors and compute the reference result via Tensor::matmul (CPU path)
    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({M, K});
    TensorPtr b_cpu = Tensor::from_vector(b_vals)->reshape({K, N});
    TensorPtr cpu_result = a_cpu->matmul(b_cpu);

    // move both operands to the GPU and run the same matmul through the CUDA dispatch branch
    TensorPtr a_gpu = a_cpu->to(Device::CUDA);
    TensorPtr b_gpu = b_cpu->to(Device::CUDA);
    TensorPtr gpu_result = a_gpu->matmul(b_gpu);

    CHECK(gpu_result->device() == Device::CUDA);

    // bring the GPU result back to host so we can compare it against the CPU reference
    TensorPtr gpu_result_host = gpu_result->to(Device::CPU);

    REQUIRE(gpu_result_host->shape() == cpu_result->shape());
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            CHECK(gpu_result_host->at({i, j}) == doctest::Approx(cpu_result->at({i, j})).epsilon(1e-3f));
        }
    }
}

TEST_CASE("Tensor::matmul rank-3 (batched) on GPU-resident tensors matches CPU Tensor::matmul"){
    const size_t L = 8, M = 12, K = 16, N = 10; // L heads, like multi-head attention

    std::mt19937 rng(13);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> a_vals(L * M * K);
    std::vector<scalar_t> b_vals(L * K * N);
    for (auto& v : a_vals) v = dist(rng);
    for (auto& v : b_vals) v = dist(rng);

    // build CPU tensors and compute the reference result via Tensor::matmul (CPU rank-3 branch)
    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({L, M, K});
    TensorPtr b_cpu = Tensor::from_vector(b_vals)->reshape({L, K, N});
    TensorPtr cpu_result = a_cpu->matmul(b_cpu);

    // move both operands to the GPU and run the same matmul through the batched CUDA dispatch branch
    TensorPtr a_gpu = a_cpu->to(Device::CUDA);
    TensorPtr b_gpu = b_cpu->to(Device::CUDA);
    TensorPtr gpu_result = a_gpu->matmul(b_gpu);

    CHECK(gpu_result->device() == Device::CUDA);

    TensorPtr gpu_result_host = gpu_result->to(Device::CPU);

    REQUIRE(gpu_result_host->shape() == cpu_result->shape());
    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                CHECK(gpu_result_host->at({l, i, j}) == doctest::Approx(cpu_result->at({l, i, j})).epsilon(1e-3f));
            }
        }
    }
}

TEST_CASE("Tensor::matmul throws when operands are on different devices"){
    TensorPtr a_cpu = Tensor::create({4, 4});
    TensorPtr b_cpu = Tensor::create({4, 4});
    TensorPtr b_gpu = b_cpu->to(Device::CUDA);

    CHECK_THROWS_AS(a_cpu->matmul(b_gpu), std::runtime_error);
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