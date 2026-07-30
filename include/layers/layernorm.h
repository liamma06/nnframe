#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"
#include "layers/layer.h"

class LayerNorm: public Layer{

    private:
        size_t normalized_shape_;
        TensorPtr gamma_;
        TensorPtr beta_;
    
    public:
        LayerNorm(size_t normalized_shape){
            normalized_shape_ = normalized_shape;

            gamma_ = std::make_shared<Tensor>(std::vector<size_t>{normalized_shape_}, 1.0f); 
            beta_ = std::make_shared<Tensor>(std::vector<size_t>{normalized_shape_}, 0.0f);
            
            gamma_->set_requires_grad(true);
            beta_->set_requires_grad(true);
        };

        std::vector<TensorPtr> parameters() const override{
            return {gamma_, beta_};
        };

        TensorPtr forward(const TensorPtr& input) override{
            /*
                input: embedding tensor
                output: normalized tensor
            */
            size_t seq_len = input->shape()[0];
            size_t embed_dim = input->shape()[1];

            auto output_tensor = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim}, 0.0f);

            for (size_t i = 0; i < seq_len; i++){
                scalar_t mean = 0.0f; 
                for (size_t j = 0; j < embed_dim; j++){
                    mean += input->at({i, j});
                }
                mean /= embed_dim;

                scalar_t variance = 0.0f;
                for (size_t j = 0; j < embed_dim; j++){
                    variance += std::pow(input->at({i, j}) - mean, 2);
                }
                variance /= embed_dim;

                for (size_t j = 0; j < embed_dim; j++){
                    output_tensor->at({i,j}) = gamma_->at({j}) * (input->at({i,j}) - mean) / std::sqrt(variance + 1e-5f) + beta_->at({j});
                }
            }

            //autograd
            auto gamma = gamma_;
            auto beta = beta_;

            auto self = std::const_pointer_cast<Tensor>(input->shared_from_this());
            output_tensor->set_inputs(std::vector<TensorPtr>{self, gamma_, beta_});
            output_tensor->set_grad_fn([self, gamma, beta](const Tensor& upstream){
                //OMG this is so confusing
                
                size_t seq_len = self->shape()[0];
                size_t embed_dim = self->shape()[1];

                if (gamma->requires_grad()) gamma->init_grad();
                if (beta->requires_grad()) beta->init_grad();
                if (self->requires_grad()) self->init_grad();

                for (size_t i = 0; i < seq_len; i++){
                    scalar_t mean = 0.0f; 
                    for (size_t j = 0; j < embed_dim; j++){
                        mean += self->at({i, j});
                    }
                    mean /= embed_dim;

                    scalar_t variance = 0.0f;
                    for (size_t j = 0; j < embed_dim; j++){
                        variance += std::pow(self->at({i, j}) - mean, 2);
                    }
                    variance /= embed_dim;

                    scalar_t inv_std = 1.0f / std::sqrt(variance + 1e-5f);


                    for (size_t j = 0; j < embed_dim; j++){
                        scalar_t x_hat = (self->at({i,j}) - mean) * inv_std;

                        if (gamma->requires_grad())
                            gamma->grad().mutable_data()[j] += upstream.at({i,j}) * x_hat;

                        if (beta->requires_grad())
                            beta->grad().mutable_data()[j] += upstream.at({i,j});
                    }

                    if (self->requires_grad()){
                        scalar_t sum_upstream = 0.0f;
                        scalar_t sum_upstream_xhat = 0.0f;
                        for (size_t j = 0; j < embed_dim; j++){
                            scalar_t x_hat = (self->at({i,j}) - mean) * inv_std;
                            sum_upstream += upstream.at({i,j}) * gamma->at({j});
                            sum_upstream_xhat += upstream.at({i,j}) * gamma->at({j}) * x_hat;
                        }
                        for (size_t j = 0; j < embed_dim; j++){
                            scalar_t x_hat = (self->at({i,j}) - mean) * inv_std;
                            self->grad().mutable_data()[i * embed_dim + j] +=
                                inv_std / embed_dim * (embed_dim * upstream.at({i,j}) * gamma->at({j}) - sum_upstream - x_hat * sum_upstream_xhat);
                        }
                    }
                }
            });

            return output_tensor;

        }



};