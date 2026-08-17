#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"
#include "modules/layer.h"
#include <random>
#include <cassert>

#ifdef NNFRAME_WITH_CUDA
#include "cuda/embed_cuda.cuh"
#endif



class Embed: public Layer{
    private:
        size_t vocab_size_;
        size_t embedding_dim_;
        TensorPtr embedding_matrix_;

    public:
        Embed(size_t vocab_size, size_t embedding_dim){
            vocab_size_ = vocab_size;
            embedding_dim_ = embedding_dim;

            embedding_matrix_ = Tensor::create({vocab_size_, embedding_dim_});
            embedding_matrix_->set_requires_grad(true);

            //random initialize (Xavier-style scale)
            std::mt19937 rng(42);
            float scale = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
            std::uniform_real_distribution<float> dist(-scale, scale);
            for (size_t i = 0; i < embedding_matrix_->numel(); i++){
                embedding_matrix_->mutable_data()[i] = dist(rng); 
            }


        }

        std::vector<TensorPtr> parameters() const override{
            return {embedding_matrix_};
        }

        void set_parameters(const std::vector<TensorPtr>& params) override{
            embedding_matrix_ = params[0];
        }

        TensorPtr forward(const TensorPtr& input) override{
            /*
                input: [seq_length] we only have 1D (batch size 1) for now
                output: [seq_length, embedding_dim]
            */
            if (input->device() != embedding_matrix_->device()){
                throw std::runtime_error("Embedding matrix and input indices must be on the same device");
            }

            #ifdef NNFRAME_WITH_CUDA
                if (input->device() == Device::CUDA){
                    size_t seq_len = input->numel();
                    scalar_t* d_out = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_out, seq_len * embedding_dim_ * sizeof(scalar_t)));
                    embed_cuda(embedding_matrix_->device_data(), input->device_data(), d_out, seq_len, embedding_dim_);

                    return Tensor::from_device_ptr(d_out, {seq_len, embedding_dim_}, {embedding_dim_, 1});
                }
            #endif

            std::vector<size_t> new_input_shape = input->shape();
            auto output_tensor = Tensor::create({new_input_shape[0], embedding_dim_});

            for (size_t i = 0; i < new_input_shape[0]; i++){
                size_t idx = static_cast<size_t>(input->at({i}));
                assert(idx < vocab_size_ && "Index out of bounds for embedding layer");
                for (size_t j = 0; j < embedding_dim_; j++){
                    output_tensor->mutable_data()[i * embedding_dim_ + j] = embedding_matrix_->at({idx, j});
                }
            }

            output_tensor->set_requires_grad(embedding_matrix_->requires_grad());

            //autograd 
            auto self = std::const_pointer_cast<Tensor>(embedding_matrix_->shared_from_this());
            output_tensor->set_inputs(std::vector<TensorPtr>{self});
            output_tensor->set_grad_fn([self, input](const Tensor& upstream){
                /*
                    upstream gradient -> into same row of embed table 
                */
                if (self->requires_grad()){
                    self->init_grad();

                    #ifdef NNFRAME_WITH_CUDA
                        if (self->device() == Device::CUDA){
                            size_t seq_len = input->numel();
                            embed_backward_cuda(input->device_data(), upstream.device_data(), self->grad().mutable_device_data(), seq_len, self->shape()[1]);
                            return;
                        }
                    #endif

                    for (size_t i = 0; i < input->shape()[0]; i++){
                        size_t idx = static_cast<size_t>(input->at({i}));
                        for (size_t j = 0; j < self->shape()[1]; j++){
                            self->grad().mutable_data()[idx * self->shape()[1] + j] += upstream.data()[i * self->shape()[1] + j];
                        }
                    }
                }
           });

            return output_tensor;
        };
        
};