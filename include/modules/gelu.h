#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"
#include "modules/layer.h"

#ifdef NNFRAME_WITH_CUDA
#include "cuda/elementwise_cuda.cuh"
#endif

class GELU : public Layer{
    public:
    TensorPtr forward(const TensorPtr& input) override{
        #ifdef NNFRAME_WITH_CUDA
            if (input->device() == Device::CUDA){
                size_t n = input->numel();
                scalar_t* d_out = nullptr;
                CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(scalar_t)));

                gelu_cuda(input->device_data(), d_out, n);

                return Tensor::from_device_ptr(d_out, input->shape(), input->strides());
            }
        #endif

        std::vector<size_t> new_input_shape = input->shape();
        auto output_tensor = Tensor::create(new_input_shape);

        for(size_t i = 0; i < input->numel(); i++){
            scalar_t x = input->mutable_data()[i];
            scalar_t sig = 1.0f / (1.0f + std::exp(-1.702f * x)); // sigmoid(1.702 * x)
            output_tensor->mutable_data()[i] = x * sig;
        }

        auto self = std::const_pointer_cast<Tensor>(input->shared_from_this());
        output_tensor->set_inputs(std::vector<TensorPtr>{self});

        output_tensor->set_grad_fn([self](const Tensor& upstream){
            if (self->requires_grad()){
                self->init_grad();

                for (size_t i = 0; i < self->numel(); i++){
                    scalar_t x = self->mutable_data()[i];
                    scalar_t sig = 1.0f / (1.0f + std::exp(-1.702f * x));
                    // product rule: d/dx[x * sig] = sig + x * sig * (1 - sig) * 1.702
                    scalar_t grad = sig + x * sig * (1.0f - sig) * 1.702f;
                    self->grad().mutable_data()[i] += upstream.data()[i] * grad;
                }
            }
        });
        output_tensor->set_requires_grad(input->requires_grad());
        return output_tensor;
    }
};
