#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"
#include "modules/layer.h"
#include <random>
#include <cassert>



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

            //random initialize
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < embedding_matrix_->numel(); i++){
                embedding_matrix_->mutable_data()[i] = dist(rng); 
            }


        }

        std::vector<TensorPtr> parameters() const override{
            return {embedding_matrix_};
        }

        TensorPtr forward(const TensorPtr& input) override{
            /*
                input: [seq_length] we only have 1D (batch size 1) for now 
                output: [seq_length, embedding_dim]
            */
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