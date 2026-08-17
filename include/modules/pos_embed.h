#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"
#include "modules/layer.h"
#include "cmath"

#ifdef NNFRAME_WITH_CUDA
#include "cuda/pos_embed_cuda.cuh"
#endif


class PositionalEmbed : public Layer{
    public:
        TensorPtr forward(const TensorPtr& input) override{
            /*
                input: [seq_length, embed_dim]
                output: [seq_length, embed_dim]
                Positional encoding is added to the input embeddings to provide information about the position of each token in the sequence.
                The encoding is based on sine and cosine functions of different frequencies.
                https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/
            */
            size_t seq_length = input->shape()[0];
            size_t embed_dim = input->shape()[1];

            #ifdef NNFRAME_WITH_CUDA
                if (input->device() == Device::CUDA){
                    scalar_t* d_pos = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_pos, seq_length * embed_dim * sizeof(scalar_t)));
                    pos_embed_cuda(d_pos, seq_length, embed_dim, 0);
                    TensorPtr pos_table = Tensor::from_device_ptr(d_pos, std::vector<size_t>{seq_length, embed_dim}, std::vector<size_t>{embed_dim, 1});
                    return input->add(pos_table);
                }
            #endif

            auto output_tensor = Tensor::create({seq_length, embed_dim});

            for (size_t pos = 0; pos < seq_length; pos++){
                for (size_t i = 0; i < embed_dim; i++){
                    if (i % 2 == 0){
                        output_tensor->at({pos, i}) = std::sin(pos / std::pow(10000.0f, static_cast<float>(i) / embed_dim));
                    } else {
                        output_tensor->at({pos, i}) = std::cos(pos / std::pow(10000.0f, static_cast<float>(i - 1) / embed_dim));
                    }
                }
            }

            return input->add(output_tensor);
        }

        std::vector<TensorPtr> parameters() const override { return {}; }

        /*
            problem is that during decode for infer it 1 token at a time
            This mean with start_pos every token is just at 0
            we need to include how many prior tokens to get the right
            positional encoding for the new token.
        */
        TensorPtr forward(const TensorPtr& input, size_t start_pos){
            size_t seq_length = input->shape()[0];
            size_t embed_dim = input->shape()[1];

            #ifdef NNFRAME_WITH_CUDA
                if (input->device() == Device::CUDA){
                    scalar_t* d_pos = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_pos, seq_length * embed_dim * sizeof(scalar_t)));
                    pos_embed_cuda(d_pos, seq_length, embed_dim, start_pos);
                    TensorPtr pos_table = Tensor::from_device_ptr(d_pos, std::vector<size_t>{seq_length, embed_dim}, std::vector<size_t>{embed_dim, 1});
                    return input->add(pos_table);
                }
            #endif

            auto output_tensor = Tensor::create({seq_length, embed_dim});

            for (size_t pos = 0; pos < seq_length; pos++){
                for (size_t i = 0; i < embed_dim; i++){
                    size_t actual_pos = start_pos + pos;
                    if (i % 2 == 0){
                        output_tensor->at({pos, i}) = std::sin(actual_pos / std::pow(10000.0f, static_cast<float>(i) / embed_dim));
                    } else {
                        output_tensor->at({pos, i}) = std::cos(actual_pos / std::pow(10000.0f, static_cast<float>(i - 1) / embed_dim));
                    }
                }
            }

            return input->add(output_tensor);
        }
};