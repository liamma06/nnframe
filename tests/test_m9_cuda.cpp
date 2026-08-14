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

TEST_CASE("Tensor::add/sub/mul on GPU-resident tensors match CPU"){
    const size_t n = 100;

    std::mt19937 rng(21);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> a_vals(n);
    std::vector<scalar_t> b_vals(n);
    for (auto& v : a_vals) v = dist(rng);
    for (auto& v : b_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals);
    TensorPtr b_cpu = Tensor::from_vector(b_vals);

    TensorPtr a_gpu = a_cpu->to(Device::CUDA);
    TensorPtr b_gpu = b_cpu->to(Device::CUDA);

    SUBCASE("add"){
        TensorPtr cpu_result = a_cpu->add(b_cpu);
        TensorPtr gpu_result = a_gpu->add(b_gpu)->to(Device::CPU);
        for (size_t i = 0; i < n; i++)
            CHECK(gpu_result->at({i}) == doctest::Approx(cpu_result->at({i})).epsilon(1e-4f));
    }

    SUBCASE("sub"){
        TensorPtr cpu_result = a_cpu->sub(b_cpu);
        TensorPtr gpu_result = a_gpu->sub(b_gpu)->to(Device::CPU);
        for (size_t i = 0; i < n; i++)
            CHECK(gpu_result->at({i}) == doctest::Approx(cpu_result->at({i})).epsilon(1e-4f));
    }

    SUBCASE("mul"){
        TensorPtr cpu_result = a_cpu->mul(b_cpu);
        TensorPtr gpu_result = a_gpu->mul(b_gpu)->to(Device::CPU);
        for (size_t i = 0; i < n; i++)
            CHECK(gpu_result->at({i}) == doctest::Approx(cpu_result->at({i})).epsilon(1e-4f));
    }
}

TEST_CASE("Tensor::add/sub/mul throw on device mismatch"){
    TensorPtr a_cpu = Tensor::create({4});
    TensorPtr b_cpu = Tensor::create({4});
    TensorPtr b_gpu = b_cpu->to(Device::CUDA);

    CHECK_THROWS_AS(a_cpu->add(b_gpu), std::runtime_error);
    CHECK_THROWS_AS(a_cpu->sub(b_gpu), std::runtime_error);
    CHECK_THROWS_AS(a_cpu->mul(b_gpu), std::runtime_error);
}

TEST_CASE("Tensor::log on GPU-resident tensor matches CPU"){
    const size_t n = 100;

    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(0.1f, 2.0f); // positive only, log(negative) is NaN

    std::vector<scalar_t> a_vals(n);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals);
    TensorPtr cpu_result = a_cpu->log();

    TensorPtr a_gpu = a_cpu->to(Device::CUDA);
    TensorPtr gpu_result = a_gpu->log()->to(Device::CPU);

    for (size_t i = 0; i < n; i++)
        CHECK(gpu_result->at({i}) == doctest::Approx(cpu_result->at({i})).epsilon(1e-4f));
}

TEST_CASE("Tensor::mean on GPU-resident tensor matches CPU (spans multiple reduction blocks)"){
    const size_t n = 1000; // > 256, exercises multiple blocks + atomicAdd combination

    std::mt19937 rng(9);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> a_vals(n);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals);
    TensorPtr cpu_result = a_cpu->mean();

    TensorPtr a_gpu = a_cpu->to(Device::CUDA);
    TensorPtr gpu_result = a_gpu->mean()->to(Device::CPU);

    CHECK(gpu_result->shape() == cpu_result->shape());
    CHECK(gpu_result->at({0}) == doctest::Approx(cpu_result->at({0})).epsilon(1e-3f));
}

TEST_CASE("Tensor::softmax rank-2 on GPU-resident tensor matches CPU (row_size > blockDim to exercise strided loop)"){
    const size_t rows = 5, row_size = 600; // > 256, forces multiple strided iterations per thread

    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<scalar_t> a_vals(rows * row_size);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({rows, row_size});
    TensorPtr cpu_result = a_cpu->softmax(1);

    TensorPtr a_gpu = a_cpu->to(Device::CUDA);
    TensorPtr gpu_result = a_gpu->softmax(1)->to(Device::CPU);

    REQUIRE(gpu_result->shape() == cpu_result->shape());
    for (size_t i = 0; i < rows; i++) {
        scalar_t row_sum = 0.0f;
        for (size_t j = 0; j < row_size; j++) {
            CHECK(gpu_result->at({i, j}) == doctest::Approx(cpu_result->at({i, j})).epsilon(1e-3f));
            row_sum += gpu_result->at({i, j});
        }
        CHECK(row_sum == doctest::Approx(1.0f).epsilon(1e-3f)); // each row must sum to 1
    }
}

TEST_CASE("Tensor::softmax rank-3 on GPU-resident tensor matches CPU"){
    const size_t L = 4, S = 6, V = 50; // e.g. [num_heads, seq_len, vocab]

    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<scalar_t> a_vals(L * S * V);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({L, S, V});
    TensorPtr cpu_result = a_cpu->softmax(2);

    TensorPtr a_gpu = a_cpu->to(Device::CUDA);
    TensorPtr gpu_result = a_gpu->softmax(2)->to(Device::CPU);

    REQUIRE(gpu_result->shape() == cpu_result->shape());
    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < S; i++) {
            for (size_t j = 0; j < V; j++) {
                CHECK(gpu_result->at({l, i, j}) == doctest::Approx(cpu_result->at({l, i, j})).epsilon(1e-3f));
            }
        }
    }
}

TEST_CASE("Tensor::add/sub/mul throw on shape mismatch when CUDA-resident (no broadcasting yet)"){
    TensorPtr a_gpu = Tensor::create({4, 4})->to(Device::CUDA);
    TensorPtr b_gpu = Tensor::create({4, 1})->to(Device::CUDA); // would broadcast on CPU, not supported on CUDA yet

    CHECK_THROWS_AS(a_gpu->add(b_gpu), std::runtime_error);
    CHECK_THROWS_AS(a_gpu->sub(b_gpu), std::runtime_error);
    CHECK_THROWS_AS(a_gpu->mul(b_gpu), std::runtime_error);
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