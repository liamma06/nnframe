#pragma once
#include <vector>
#include <random>
#include <cmath>
#include "core/tensor.h"
#include "modules/layer.h"

#ifdef NNFRAME_WITH_CUDA
#include "cuda/linear_cuda.cuh"
#endif

// Linear followed by GELU, fused: matmul stays a separate op (reuses Tensor::matmul),
// but bias-add + GELU run as one kernel instead of two.
class LinearGELU : public Layer{
    private:
        TensorPtr weights_;
        TensorPtr bias_;

    public:
        LinearGELU(size_t in_features, size_t out_features){
            weights_ = Tensor::create({in_features, out_features});
            bias_ = Tensor::create({out_features});
            weights_->set_requires_grad(true);
            bias_->set_requires_grad(true);

            float scale = 1.0f / std::sqrt(static_cast<float>(in_features));
            std::uniform_real_distribution<float> dist(-scale, scale);
            std::mt19937 rng(42);
            for (size_t i = 0; i < weights_->numel(); i++){
                weights_->mutable_data()[i] = dist(rng);
            }
        }

        std::vector<TensorPtr> parameters() const override{
            return {weights_, bias_};
        }

        TensorPtr forward(const TensorPtr& input) override{
            TensorPtr output_xW = input->matmul(weights_); // [batch_size, out_features]

            size_t batch_size = output_xW->shape()[0];
            size_t features = output_xW->shape()[1];

            #ifdef NNFRAME_WITH_CUDA
                if (output_xW->device() == Device::CUDA){
                    size_t n = output_xW->numel();
                    scalar_t* d_out = nullptr;
                    CUDA_CHECK(cudaMallocAsync(&d_out, n * sizeof(scalar_t), 0));

                    bias_gelu_cuda(output_xW->device_data(), bias_->device_data(), d_out, batch_size, features);

                    auto output_tensor = Tensor::from_device_ptr(d_out, output_xW->shape(), output_xW->strides());

                    auto self = output_xW;
                    auto bias = bias_;
                    output_tensor->set_requires_grad(self->requires_grad() || bias->requires_grad());
                    output_tensor->set_inputs(std::vector<TensorPtr>{self, bias});
                    output_tensor->set_grad_fn([self, bias, batch_size, features](const Tensor& upstream){
                        if (self->requires_grad() || bias->requires_grad()){
                            self->init_grad();
                            bias->init_grad();
                            bias_gelu_grad_cuda(upstream.device_data(), self->device_data(), bias->device_data(),
                                                 self->grad().mutable_device_data(), bias->grad().mutable_device_data(),
                                                 batch_size, features);
                        }
                    });

                    return output_tensor;
                }
            #endif

            auto output_tensor = Tensor::create(output_xW->shape());
            for (size_t i = 0; i < output_xW->numel(); i++){
                size_t feature_idx = i % features;
                scalar_t x = output_xW->mutable_data()[i] + bias_->mutable_data()[feature_idx];
                scalar_t sig = 1.0f / (1.0f + std::exp(-1.702f * x));
                output_tensor->mutable_data()[i] = x * sig;
            }

            auto self = output_xW;
            auto bias = bias_;
            output_tensor->set_requires_grad(self->requires_grad() || bias->requires_grad());
            output_tensor->set_inputs(std::vector<TensorPtr>{self, bias});
            output_tensor->set_grad_fn([self, bias, batch_size, features](const Tensor& upstream){
                if (self->requires_grad() || bias->requires_grad()){
                    self->init_grad();
                    bias->init_grad();
                    for (size_t i = 0; i < self->numel(); i++){
                        size_t feature_idx = i % features;
                        scalar_t x = self->mutable_data()[i] + bias->mutable_data()[feature_idx];
                        scalar_t sig = 1.0f / (1.0f + std::exp(-1.702f * x));
                        scalar_t grad = sig + x * sig * (1.0f - sig) * 1.702f;
                        self->grad().mutable_data()[i] += upstream.data()[i] * grad;
                        bias->grad().mutable_data()[feature_idx] += upstream.data()[i] * grad;
                    }
                }
            });

            return output_tensor;
        }
};
