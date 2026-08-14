#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"
#include "cuda/matmul_cuda.cuh"
#include "cuda/embed_cuda.cuh"
#include "cuda/loss_cuda.cuh"
#include "cuda/layernorm_cuda.cuh"
#include "modules/relu.h"
#include "modules/gelu.h"
#include "modules/layernorm.h"
#include "loss/cross_entrop.h"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>

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

TEST_CASE("Tensor::relu/gelu on GPU-resident tensors match CPU"){
    const size_t n = 100;

    std::mt19937 rng(31);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<scalar_t> a_vals(n);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals);
    TensorPtr a_gpu = a_cpu->to(Device::CUDA);

    SUBCASE("relu"){
        ReLu relu;
        TensorPtr cpu_result = relu.forward(a_cpu);
        TensorPtr gpu_result = relu.forward(a_gpu)->to(Device::CPU);
        for (size_t i = 0; i < n; i++)
            CHECK(gpu_result->at({i}) == doctest::Approx(cpu_result->at({i})).epsilon(1e-4f));
    }

    SUBCASE("gelu"){
        GELU gelu;
        TensorPtr cpu_result = gelu.forward(a_cpu);
        TensorPtr gpu_result = gelu.forward(a_gpu)->to(Device::CPU);
        for (size_t i = 0; i < n; i++)
            CHECK(gpu_result->at({i}) == doctest::Approx(cpu_result->at({i})).epsilon(1e-4f));
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

TEST_CASE("CUDA embedding forward/backward match a scalar reference"){
    const size_t vocab_size = 10, embedding_dim = 8, seq_len = 6;

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(vocab_size) - 1);

    std::vector<scalar_t> h_table(vocab_size * embedding_dim);
    for (auto& v : h_table) v = val_dist(rng);

    // indices as scalar_t, some repeated on purpose to exercise the atomicAdd scatter in backward
    std::vector<scalar_t> h_indices(seq_len);
    for (auto& v : h_indices) v = static_cast<scalar_t>(idx_dist(rng));

    scalar_t *d_table, *d_indices, *d_out;
    CUDA_CHECK(cudaMalloc(&d_table, h_table.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_out, seq_len * embedding_dim * sizeof(scalar_t)));

    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));

    embed_cuda(d_table, d_indices, d_out, seq_len, embedding_dim);

    std::vector<scalar_t> out_cuda(seq_len * embedding_dim);
    CUDA_CHECK(cudaMemcpy(out_cuda.data(), d_out, out_cuda.size() * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    // scalar reference: gather
    std::vector<scalar_t> out_scalar(seq_len * embedding_dim);
    for (size_t i = 0; i < seq_len; i++) {
        size_t token = static_cast<size_t>(h_indices[i]);
        for (size_t d = 0; d < embedding_dim; d++)
            out_scalar[i * embedding_dim + d] = h_table[token * embedding_dim + d];
    }

    for (size_t i = 0; i < seq_len * embedding_dim; i++)
        CHECK(out_cuda[i] == doctest::Approx(out_scalar[i]).epsilon(1e-4f));

    // backward: random upstream gradient
    std::vector<scalar_t> h_grad_out(seq_len * embedding_dim);
    for (auto& v : h_grad_out) v = val_dist(rng);

    scalar_t *d_grad_out, *d_grad_table;
    CUDA_CHECK(cudaMalloc(&d_grad_out, h_grad_out.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_grad_table, h_table.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMemset(d_grad_table, 0, h_table.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMemcpy(d_grad_out, h_grad_out.data(), h_grad_out.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));

    embed_backward_cuda(d_indices, d_grad_out, d_grad_table, seq_len, embedding_dim);

    std::vector<scalar_t> grad_table_cuda(h_table.size());
    CUDA_CHECK(cudaMemcpy(grad_table_cuda.data(), d_grad_table, grad_table_cuda.size() * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    // scalar reference: scatter-add
    std::vector<scalar_t> grad_table_scalar(h_table.size(), 0.0f);
    for (size_t i = 0; i < seq_len; i++) {
        size_t token = static_cast<size_t>(h_indices[i]);
        for (size_t d = 0; d < embedding_dim; d++)
            grad_table_scalar[token * embedding_dim + d] += h_grad_out[i * embedding_dim + d];
    }

    for (size_t i = 0; i < h_table.size(); i++)
        CHECK(grad_table_cuda[i] == doctest::Approx(grad_table_scalar[i]).epsilon(1e-4f));

    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_grad_out));
    CUDA_CHECK(cudaFree(d_grad_table));
}

TEST_CASE("CUDA cross_entropy forward/backward match CPU CrossEntropy"){
    const size_t rows = 5, vocab_size = 12;

    std::mt19937 rng(19);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> target_dist(0, static_cast<int>(vocab_size) - 1);

    std::vector<scalar_t> h_logits(rows * vocab_size);
    for (auto& v : h_logits) v = val_dist(rng);

    std::vector<scalar_t> h_targets(rows);
    for (auto& v : h_targets) v = static_cast<scalar_t>(target_dist(rng));

    // CPU reference: build a real graph so backward() chains through softmax/log/mean correctly
    TensorPtr logits_cpu = Tensor::from_vector(h_logits)->reshape({rows, vocab_size});
    logits_cpu->set_requires_grad(true);
    TensorPtr targets_cpu = Tensor::from_vector(h_targets);

    CrossEntropy ce;
    TensorPtr loss_cpu = ce.forward(logits_cpu, targets_cpu);
    loss_cpu->backward();

    // CUDA path: raw kernel-level call, matching the loss value and the logits gradient
    scalar_t *d_logits, *d_targets, *d_loss, *d_grad_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, h_logits.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_targets, h_targets.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_grad_logits, h_logits.size() * sizeof(scalar_t)));

    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), h_targets.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));

    cross_entropy_cuda(d_logits, d_targets, d_loss, rows, vocab_size);

    scalar_t loss_cuda_val = 0.0f;
    CUDA_CHECK(cudaMemcpy(&loss_cuda_val, d_loss, sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CHECK(loss_cuda_val == doctest::Approx(loss_cpu->at({0})).epsilon(1e-3f));

    scalar_t h_upstream = 1.0f;
    scalar_t* d_upstream = nullptr;
    CUDA_CHECK(cudaMalloc(&d_upstream, sizeof(scalar_t)));
    CUDA_CHECK(cudaMemcpy(d_upstream, &h_upstream, sizeof(scalar_t), cudaMemcpyHostToDevice));

    cross_entropy_backward_cuda(d_logits, d_targets, d_grad_logits, rows, vocab_size, d_upstream);

    std::vector<scalar_t> grad_logits_cuda(h_logits.size());
    CUDA_CHECK(cudaMemcpy(grad_logits_cuda.data(), d_grad_logits, grad_logits_cuda.size() * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            CHECK(grad_logits_cuda[i * vocab_size + j] == doctest::Approx(logits_cpu->grad().at({i, j})).epsilon(1e-3f));
        }
    }

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_grad_logits));
    CUDA_CHECK(cudaFree(d_upstream));
}

TEST_CASE("CUDA layernorm matches scalar reference"){
    const size_t rows = 5, cols = 20;
    const scalar_t eps = 1e-5f;

    std::mt19937 rng(29);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);

    std::vector<scalar_t> h_input(rows * cols);
    for (auto& v : h_input) v = val_dist(rng);

    std::vector<scalar_t> h_gamma(cols);
    std::vector<scalar_t> h_beta(cols);
    for (auto& v : h_gamma) v = val_dist(rng);
    for (auto& v : h_beta) v = val_dist(rng);

    // scalar reference, mirrors LayerNorm::forward's CPU math exactly
    std::vector<scalar_t> out_scalar(rows * cols);
    for (size_t i = 0; i < rows; i++){
        scalar_t mean = 0.0f;
        for (size_t j = 0; j < cols; j++) mean += h_input[i * cols + j];
        mean /= cols;

        scalar_t variance = 0.0f;
        for (size_t j = 0; j < cols; j++){
            scalar_t diff = h_input[i * cols + j] - mean;
            variance += diff * diff;
        }
        variance /= cols;

        scalar_t inv_std = 1.0f / std::sqrt(variance + eps);
        for (size_t j = 0; j < cols; j++){
            scalar_t x_hat = (h_input[i * cols + j] - mean) * inv_std;
            out_scalar[i * cols + j] = h_gamma[j] * x_hat + h_beta[j];
        }
    }

    scalar_t *d_input, *d_gamma, *d_beta, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_gamma, h_gamma.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_beta, h_beta.size() * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(scalar_t)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));

    layernorm_cuda(d_input, d_gamma, d_beta, d_out, rows, cols, eps);

    std::vector<scalar_t> out_cuda(rows * cols);
    CUDA_CHECK(cudaMemcpy(out_cuda.data(), d_out, out_cuda.size() * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < rows * cols; i++)
        CHECK(out_cuda[i] == doctest::Approx(out_scalar[i]).epsilon(1e-3f));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_out));
}

TEST_CASE("CUDA CrossEntropy real backward() matches CPU gradient"){
    const size_t rows = 5, vocab_size = 12;

    std::mt19937 rng(41);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> target_dist(0, static_cast<int>(vocab_size) - 1);

    std::vector<scalar_t> h_logits(rows * vocab_size);
    for (auto& v : h_logits) v = val_dist(rng);

    std::vector<scalar_t> h_targets(rows);
    for (auto& v : h_targets) v = static_cast<scalar_t>(target_dist(rng));

    // CPU reference
    TensorPtr logits_cpu = Tensor::from_vector(h_logits)->reshape({rows, vocab_size});
    logits_cpu->set_requires_grad(true);
    TensorPtr targets_cpu = Tensor::from_vector(h_targets);

    CrossEntropy ce_cpu;
    TensorPtr loss_cpu = ce_cpu.forward(logits_cpu, targets_cpu);
    loss_cpu->backward();

    // GPU: through the real CrossEntropy class and real Tensor::backward(), exercising the grad_fn_ we just wired
    TensorPtr logits_gpu = Tensor::from_vector(h_logits)->reshape({rows, vocab_size})->to(Device::CUDA);
    logits_gpu->set_requires_grad(true);
    TensorPtr targets_gpu = Tensor::from_vector(h_targets)->to(Device::CUDA);

    CrossEntropy ce_gpu;
    TensorPtr loss_gpu = ce_gpu.forward(logits_gpu, targets_gpu);
    loss_gpu->backward();

    TensorPtr loss_gpu_host = loss_gpu->to(Device::CPU);
    CHECK(loss_gpu_host->at({0}) == doctest::Approx(loss_cpu->at({0})).epsilon(1e-3f));

    TensorPtr grad_gpu_host = logits_gpu->grad().to(Device::CPU);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            CHECK(grad_gpu_host->at({i, j}) == doctest::Approx(logits_cpu->grad().at({i, j})).epsilon(1e-3f));
        }
    }
}

TEST_CASE("CUDA embedding lookup backward via real Tensor::backward() matches scalar reference"){
    // Embed the Layer class always builds its table on CPU with no way to move it,
    // so this manually replicates embed.h's CUDA branch to exercise the real grad_fn_ wiring.
    const size_t vocab_size = 10, embedding_dim = 8, seq_len = 6;

    std::mt19937 rng(43);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(vocab_size) - 1);

    std::vector<scalar_t> h_table(vocab_size * embedding_dim);
    for (auto& v : h_table) v = val_dist(rng);

    std::vector<scalar_t> h_indices(seq_len);
    for (auto& v : h_indices) v = static_cast<scalar_t>(idx_dist(rng));

    TensorPtr table_gpu = Tensor::from_vector(h_table)->reshape({vocab_size, embedding_dim})->to(Device::CUDA);
    table_gpu->set_requires_grad(true);
    TensorPtr indices_gpu = Tensor::from_vector(h_indices)->to(Device::CUDA);

    scalar_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, seq_len * embedding_dim * sizeof(scalar_t)));
    embed_cuda(table_gpu->device_data(), indices_gpu->device_data(), d_out, seq_len, embedding_dim);
    TensorPtr output_gpu = Tensor::from_device_ptr(d_out, std::vector<size_t>{seq_len, embedding_dim}, std::vector<size_t>{embedding_dim, 1});

    output_gpu->set_requires_grad(table_gpu->requires_grad());
    output_gpu->set_inputs(std::vector<TensorPtr>{table_gpu});
    output_gpu->set_grad_fn([table_gpu, indices_gpu, seq_len, embedding_dim](const Tensor& upstream){
        if (table_gpu->requires_grad()){
            table_gpu->init_grad();
            embed_backward_cuda(indices_gpu->device_data(), upstream.device_data(), table_gpu->grad().mutable_device_data(), seq_len, embedding_dim);
        }
    });

    output_gpu->backward(); // root grad defaults to all-ones, matching output_gpu's shape

    TensorPtr grad_host = table_gpu->grad().to(Device::CPU);

    // scalar reference: upstream is all-ones (matches backward()'s default root gradient), so this
    // reduces to "how many times did each (token, dim) get touched" - still exercises the atomicAdd
    // scatter for repeated tokens, just with 1.0 contributions instead of random ones.
    std::vector<scalar_t> grad_table_scalar(h_table.size(), 0.0f);
    for (size_t i = 0; i < seq_len; i++) {
        size_t token = static_cast<size_t>(h_indices[i]);
        for (size_t d = 0; d < embedding_dim; d++)
            grad_table_scalar[token * embedding_dim + d] += 1.0f;
    }

    for (size_t i = 0; i < h_table.size(); i++)
        CHECK(grad_host->data()[i] == doctest::Approx(grad_table_scalar[i]).epsilon(1e-4f));
}

TEST_CASE("CUDA add/sub/mul backward via real Tensor::backward() matches CPU"){
    const size_t n = 50;

    std::mt19937 rng(51);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<scalar_t> a_vals(n), b_vals(n);
    for (auto& v : a_vals) v = dist(rng);
    for (auto& v : b_vals) v = dist(rng);

    SUBCASE("add"){
        TensorPtr a_cpu = Tensor::from_vector(a_vals); a_cpu->set_requires_grad(true);
        TensorPtr b_cpu = Tensor::from_vector(b_vals); b_cpu->set_requires_grad(true);
        TensorPtr c_cpu = a_cpu->add(b_cpu);
        c_cpu->backward();

        TensorPtr a_gpu = Tensor::from_vector(a_vals)->to(Device::CUDA); a_gpu->set_requires_grad(true);
        TensorPtr b_gpu = Tensor::from_vector(b_vals)->to(Device::CUDA); b_gpu->set_requires_grad(true);
        TensorPtr c_gpu = a_gpu->add(b_gpu);
        c_gpu->backward();

        TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
        TensorPtr b_grad_host = b_gpu->grad().to(Device::CPU);
        for (size_t i = 0; i < n; i++){
            CHECK(a_grad_host->at({i}) == doctest::Approx(a_cpu->grad().at({i})).epsilon(1e-4f));
            CHECK(b_grad_host->at({i}) == doctest::Approx(b_cpu->grad().at({i})).epsilon(1e-4f));
        }
    }

    SUBCASE("sub"){
        TensorPtr a_cpu = Tensor::from_vector(a_vals); a_cpu->set_requires_grad(true);
        TensorPtr b_cpu = Tensor::from_vector(b_vals); b_cpu->set_requires_grad(true);
        TensorPtr c_cpu = a_cpu->sub(b_cpu);
        c_cpu->backward();

        TensorPtr a_gpu = Tensor::from_vector(a_vals)->to(Device::CUDA); a_gpu->set_requires_grad(true);
        TensorPtr b_gpu = Tensor::from_vector(b_vals)->to(Device::CUDA); b_gpu->set_requires_grad(true);
        TensorPtr c_gpu = a_gpu->sub(b_gpu);
        c_gpu->backward();

        TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
        TensorPtr b_grad_host = b_gpu->grad().to(Device::CPU);
        for (size_t i = 0; i < n; i++){
            CHECK(a_grad_host->at({i}) == doctest::Approx(a_cpu->grad().at({i})).epsilon(1e-4f));
            CHECK(b_grad_host->at({i}) == doctest::Approx(b_cpu->grad().at({i})).epsilon(1e-4f));
        }
    }

    SUBCASE("mul"){
        TensorPtr a_cpu = Tensor::from_vector(a_vals); a_cpu->set_requires_grad(true);
        TensorPtr b_cpu = Tensor::from_vector(b_vals); b_cpu->set_requires_grad(true);
        TensorPtr c_cpu = a_cpu->mul(b_cpu);
        c_cpu->backward();

        TensorPtr a_gpu = Tensor::from_vector(a_vals)->to(Device::CUDA); a_gpu->set_requires_grad(true);
        TensorPtr b_gpu = Tensor::from_vector(b_vals)->to(Device::CUDA); b_gpu->set_requires_grad(true);
        TensorPtr c_gpu = a_gpu->mul(b_gpu);
        c_gpu->backward();

        TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
        TensorPtr b_grad_host = b_gpu->grad().to(Device::CPU);
        for (size_t i = 0; i < n; i++){
            CHECK(a_grad_host->at({i}) == doctest::Approx(a_cpu->grad().at({i})).epsilon(1e-4f));
            CHECK(b_grad_host->at({i}) == doctest::Approx(b_cpu->grad().at({i})).epsilon(1e-4f));
        }
    }
}

TEST_CASE("CUDA log backward via real Tensor::backward() matches CPU"){
    const size_t n = 50;

    std::mt19937 rng(53);
    std::uniform_real_distribution<float> dist(0.1f, 2.0f); // positive only

    std::vector<scalar_t> a_vals(n);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals); a_cpu->set_requires_grad(true);
    TensorPtr c_cpu = a_cpu->log();
    c_cpu->backward();

    TensorPtr a_gpu = Tensor::from_vector(a_vals)->to(Device::CUDA); a_gpu->set_requires_grad(true);
    TensorPtr c_gpu = a_gpu->log();
    c_gpu->backward();

    TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
    for (size_t i = 0; i < n; i++)
        CHECK(a_grad_host->at({i}) == doctest::Approx(a_cpu->grad().at({i})).epsilon(1e-4f));
}

TEST_CASE("CUDA mean backward via real Tensor::backward() matches CPU"){
    const size_t n = 1000; // spans multiple reduction blocks

    std::mt19937 rng(57);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<scalar_t> a_vals(n);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals); a_cpu->set_requires_grad(true);
    TensorPtr c_cpu = a_cpu->mean();
    c_cpu->backward();

    TensorPtr a_gpu = Tensor::from_vector(a_vals)->to(Device::CUDA); a_gpu->set_requires_grad(true);
    TensorPtr c_gpu = a_gpu->mean();
    c_gpu->backward();

    TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
    for (size_t i = 0; i < n; i++)
        CHECK(a_grad_host->at({i}) == doctest::Approx(a_cpu->grad().at({i})).epsilon(1e-4f));
}

TEST_CASE("CUDA relu/gelu backward via real Tensor::backward() matches CPU"){
    const size_t n = 50;

    std::mt19937 rng(59);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<scalar_t> a_vals(n);
    for (auto& v : a_vals) v = dist(rng);

    SUBCASE("relu"){
        TensorPtr a_cpu = Tensor::from_vector(a_vals); a_cpu->set_requires_grad(true);
        ReLu relu_cpu;
        TensorPtr c_cpu = relu_cpu.forward(a_cpu);
        c_cpu->backward();

        TensorPtr a_gpu = Tensor::from_vector(a_vals)->to(Device::CUDA); a_gpu->set_requires_grad(true);
        ReLu relu_gpu;
        TensorPtr c_gpu = relu_gpu.forward(a_gpu);
        c_gpu->backward();

        TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
        for (size_t i = 0; i < n; i++)
            CHECK(a_grad_host->at({i}) == doctest::Approx(a_cpu->grad().at({i})).epsilon(1e-4f));
    }

    SUBCASE("gelu"){
        TensorPtr a_cpu = Tensor::from_vector(a_vals); a_cpu->set_requires_grad(true);
        GELU gelu_cpu;
        TensorPtr c_cpu = gelu_cpu.forward(a_cpu);
        c_cpu->backward();

        TensorPtr a_gpu = Tensor::from_vector(a_vals)->to(Device::CUDA); a_gpu->set_requires_grad(true);
        GELU gelu_gpu;
        TensorPtr c_gpu = gelu_gpu.forward(a_gpu);
        c_gpu->backward();

        TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
        for (size_t i = 0; i < n; i++)
            CHECK(a_grad_host->at({i}) == doctest::Approx(a_cpu->grad().at({i})).epsilon(1e-4f));
    }
}

TEST_CASE("CUDA matmul rank-2 backward via real Tensor::backward() matches CPU"){
    const size_t M = 12, K = 8, N = 10;

    std::mt19937 rng(61);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> a_vals(M * K), b_vals(K * N);
    for (auto& v : a_vals) v = dist(rng);
    for (auto& v : b_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({M, K}); a_cpu->set_requires_grad(true);
    TensorPtr b_cpu = Tensor::from_vector(b_vals)->reshape({K, N}); b_cpu->set_requires_grad(true);
    TensorPtr c_cpu = a_cpu->matmul(b_cpu);
    c_cpu->backward();

    TensorPtr a_gpu = Tensor::from_vector(a_vals)->reshape({M, K})->to(Device::CUDA); a_gpu->set_requires_grad(true);
    TensorPtr b_gpu = Tensor::from_vector(b_vals)->reshape({K, N})->to(Device::CUDA); b_gpu->set_requires_grad(true);
    TensorPtr c_gpu = a_gpu->matmul(b_gpu);
    c_gpu->backward();

    TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
    TensorPtr b_grad_host = b_gpu->grad().to(Device::CPU);

    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            CHECK(a_grad_host->at({i, k}) == doctest::Approx(a_cpu->grad().at({i, k})).epsilon(1e-3f));

    for (size_t k = 0; k < K; k++)
        for (size_t j = 0; j < N; j++)
            CHECK(b_grad_host->at({k, j}) == doctest::Approx(b_cpu->grad().at({k, j})).epsilon(1e-3f));
}

TEST_CASE("CUDA matmul rank-3 (batched) backward via real Tensor::backward() matches CPU"){
    const size_t L = 4, M = 6, K = 5, N = 7;

    std::mt19937 rng(63);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<scalar_t> a_vals(L * M * K), b_vals(L * K * N);
    for (auto& v : a_vals) v = dist(rng);
    for (auto& v : b_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({L, M, K}); a_cpu->set_requires_grad(true);
    TensorPtr b_cpu = Tensor::from_vector(b_vals)->reshape({L, K, N}); b_cpu->set_requires_grad(true);
    TensorPtr c_cpu = a_cpu->matmul(b_cpu);
    c_cpu->backward();

    TensorPtr a_gpu = Tensor::from_vector(a_vals)->reshape({L, M, K})->to(Device::CUDA); a_gpu->set_requires_grad(true);
    TensorPtr b_gpu = Tensor::from_vector(b_vals)->reshape({L, K, N})->to(Device::CUDA); b_gpu->set_requires_grad(true);
    TensorPtr c_gpu = a_gpu->matmul(b_gpu);
    c_gpu->backward();

    TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
    TensorPtr b_grad_host = b_gpu->grad().to(Device::CPU);

    for (size_t l = 0; l < L; l++)
        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++)
                CHECK(a_grad_host->at({l, i, k}) == doctest::Approx(a_cpu->grad().at({l, i, k})).epsilon(1e-3f));

    for (size_t l = 0; l < L; l++)
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
                CHECK(b_grad_host->at({l, k, j}) == doctest::Approx(b_cpu->grad().at({l, k, j})).epsilon(1e-3f));
}

TEST_CASE("CUDA softmax rank-2 backward via real Tensor::backward() matches CPU"){
    const size_t rows = 6, row_size = 20;

    std::mt19937 rng(67);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<scalar_t> a_vals(rows * row_size);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({rows, row_size}); a_cpu->set_requires_grad(true);
    TensorPtr c_cpu = a_cpu->softmax(1);
    c_cpu->backward();

    TensorPtr a_gpu = Tensor::from_vector(a_vals)->reshape({rows, row_size})->to(Device::CUDA); a_gpu->set_requires_grad(true);
    TensorPtr c_gpu = a_gpu->softmax(1);
    c_gpu->backward();

    TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < row_size; j++)
            CHECK(a_grad_host->at({i, j}) == doctest::Approx(a_cpu->grad().at({i, j})).epsilon(1e-3f));
}

TEST_CASE("CUDA softmax rank-3 backward via real Tensor::backward() matches CPU"){
    const size_t L = 3, S = 5, V = 15;

    std::mt19937 rng(71);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<scalar_t> a_vals(L * S * V);
    for (auto& v : a_vals) v = dist(rng);

    TensorPtr a_cpu = Tensor::from_vector(a_vals)->reshape({L, S, V}); a_cpu->set_requires_grad(true);
    TensorPtr c_cpu = a_cpu->softmax(2);
    c_cpu->backward();

    TensorPtr a_gpu = Tensor::from_vector(a_vals)->reshape({L, S, V})->to(Device::CUDA); a_gpu->set_requires_grad(true);
    TensorPtr c_gpu = a_gpu->softmax(2);
    c_gpu->backward();

    TensorPtr a_grad_host = a_gpu->grad().to(Device::CPU);
    for (size_t l = 0; l < L; l++)
        for (size_t i = 0; i < S; i++)
            for (size_t j = 0; j < V; j++)
                CHECK(a_grad_host->at({l, i, j}) == doctest::Approx(a_cpu->grad().at({l, i, j})).epsilon(1e-3f));
}

TEST_CASE("CUDA LayerNorm backward via real Tensor::backward() matches CPU"){
    // LayerNorm's gamma/beta are always deterministically initialized (1.0 / 0.0, no RNG),
    // so a fresh CPU LayerNorm and a manually-built GPU graph start from identical parameters.
    const size_t seq_len = 6, embed_dim = 10;

    std::mt19937 rng(73);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<scalar_t> input_vals(seq_len * embed_dim);
    for (auto& v : input_vals) v = dist(rng);

    // CPU reference, via the real LayerNorm class
    LayerNorm ln_cpu(embed_dim);
    TensorPtr input_cpu = Tensor::from_vector(input_vals)->reshape({seq_len, embed_dim});
    input_cpu->set_requires_grad(true);
    TensorPtr out_cpu = ln_cpu.forward(input_cpu);
    out_cpu->backward();

    TensorPtr gamma_grad_cpu = ln_cpu.parameters()[0]->grad().to(Device::CPU);
    TensorPtr beta_grad_cpu = ln_cpu.parameters()[1]->grad().to(Device::CPU);

    // GPU: LayerNorm's internal gamma_/beta_ can't be moved to CUDA (no setter), so this
    // manually replicates LayerNorm::forward's CUDA branch, same as the embedding backward test.
    TensorPtr gamma_gpu = Tensor::create({embed_dim}, 1.0f)->to(Device::CUDA);
    gamma_gpu->set_requires_grad(true);
    TensorPtr beta_gpu = Tensor::zeros({embed_dim})->to(Device::CUDA);
    beta_gpu->set_requires_grad(true);
    TensorPtr input_gpu = Tensor::from_vector(input_vals)->reshape({seq_len, embed_dim})->to(Device::CUDA);
    input_gpu->set_requires_grad(true);

    scalar_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, seq_len * embed_dim * sizeof(scalar_t)));
    layernorm_cuda(input_gpu->device_data(), gamma_gpu->device_data(), beta_gpu->device_data(), d_out, seq_len, embed_dim, 1e-5f);
    TensorPtr out_gpu = Tensor::from_device_ptr(d_out, std::vector<size_t>{seq_len, embed_dim}, std::vector<size_t>{embed_dim, 1});

    out_gpu->set_requires_grad(true);
    out_gpu->set_inputs(std::vector<TensorPtr>{input_gpu, gamma_gpu, beta_gpu});
    out_gpu->set_grad_fn([input_gpu, gamma_gpu, beta_gpu, seq_len, embed_dim](const Tensor& upstream){
        input_gpu->init_grad();
        gamma_gpu->init_grad();
        beta_gpu->init_grad();
        layernorm_grad_cuda(upstream.device_data(), input_gpu->device_data(), gamma_gpu->device_data(),
                             input_gpu->grad().mutable_device_data(), gamma_gpu->grad().mutable_device_data(), beta_gpu->grad().mutable_device_data(),
                             seq_len, embed_dim, 1e-5f);
    });

    out_gpu->backward();

    TensorPtr input_grad_host = input_gpu->grad().to(Device::CPU);
    TensorPtr gamma_grad_host = gamma_gpu->grad().to(Device::CPU);
    TensorPtr beta_grad_host = beta_gpu->grad().to(Device::CPU);

    for (size_t i = 0; i < seq_len; i++)
        for (size_t j = 0; j < embed_dim; j++)
            CHECK(input_grad_host->at({i, j}) == doctest::Approx(input_cpu->grad().at({i, j})).epsilon(1e-3f));

    for (size_t j = 0; j < embed_dim; j++){
        CHECK(gamma_grad_host->at({j}) == doctest::Approx(gamma_grad_cpu->at({j})).epsilon(1e-3f));
        CHECK(beta_grad_host->at({j}) == doctest::Approx(beta_grad_cpu->at({j})).epsilon(1e-3f));
    }
}