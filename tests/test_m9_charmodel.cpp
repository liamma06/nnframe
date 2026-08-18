#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "cuda/mempool.cuh"
#include "core/tensor.h"
#include "cuda/matmul_cuda.cuh"
#include "cuda/embed_cuda.cuh"
#include "cuda/pos_embed_cuda.cuh"
#include "cuda/layernorm_cuda.cuh"
#include "cuda/attention_cuda.cuh"
#include "cuda/linear_cuda.cuh"
#include "modules/char_model.h"
#include "modules/gelu.h"
#include <vector>
#include <random>
#include <cmath>



namespace {
    void check_charmodel_matches_cpu(size_t vocab_size, size_t embedding_dim, size_t num_heads, size_t num_blocks, size_t seq_len, int seed) {
        CharModel model(vocab_size, embedding_dim, num_heads, num_blocks);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> tok_dist(0, static_cast<int>(vocab_size) - 1);
        std::vector<scalar_t> token_ids(seq_len);
        for (auto& t : token_ids) t = static_cast<scalar_t>(tok_dist(rng));
        TensorPtr input_cpu = Tensor::from_vector(token_ids); // token ids don't require grad

        TensorPtr output_cpu = model.forward(input_cpu);
        output_cpu->backward();

        // capture the CPU weight tensors (with their now-computed grads) before to() replaces them
        std::vector<TensorPtr> params_cpu = model.parameters();

        model.to(Device::CUDA); // moves every weight in place; forward()/backward() need no changes

        TensorPtr input_gpu = input_cpu->to(Device::CUDA);
        TensorPtr output_gpu = model.forward(input_gpu);
        output_gpu->backward();

        std::vector<TensorPtr> params_gpu = model.parameters();

        TensorPtr output_gpu_host = output_gpu->to(Device::CPU);
        for (size_t i = 0; i < seq_len; i++)
            for (size_t j = 0; j < vocab_size; j++)
                CHECK(output_gpu_host->at({i, j}) == doctest::Approx(output_cpu->at({i, j})).epsilon(1e-3f));

        REQUIRE(params_cpu.size() == params_gpu.size());
        for (size_t p = 0; p < params_cpu.size(); p++) {
            TensorPtr grad_host = params_gpu[p]->grad().to(Device::CPU);
            for (size_t i = 0; i < params_cpu[p]->numel(); i++)
                CHECK(grad_host->data()[i] == doctest::Approx(params_cpu[p]->grad().data()[i]).epsilon(1e-3f));
        }
    }
}

TEST_CASE("CUDA CharModel (1 transformer block) forward+backward matches CPU") {
    check_charmodel_matches_cpu(/*vocab_size=*/10, /*embedding_dim=*/8, /*num_heads=*/2, /*num_blocks=*/1, /*seq_len=*/4, /*seed=*/19);
}

TEST_CASE("CUDA CharModel (multiple stacked transformer blocks) forward+backward matches CPU") {
    check_charmodel_matches_cpu(/*vocab_size=*/10, /*embedding_dim=*/8, /*num_heads=*/2, /*num_blocks=*/3, /*seq_len=*/4, /*seed=*/23);
}

int main(int argc, char** argv) {
    init_cuda_mempool();

    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
