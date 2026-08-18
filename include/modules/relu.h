#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"
#include "modules/layer.h"

#ifdef NNFRAME_WITH_CUDA
#include "cuda/elementwise_cuda.cuh"
#endif

class ReLu: public Layer{
    public:
    TensorPtr forward(const TensorPtr& input) override{
        #ifdef NNFRAME_WITH_CUDA
            if (input->device() == Device::CUDA){
                size_t n = input->numel();
                scalar_t* d_out = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_out, n * sizeof(scalar_t), 0));

                relu_cuda(input->device_data(), d_out, n);

                auto output_tensor = Tensor::from_device_ptr(d_out, input->shape(), input->strides());

                auto self = std::const_pointer_cast<Tensor>(input->shared_from_this());
                output_tensor->set_requires_grad(input->requires_grad());
                output_tensor->set_inputs(std::vector<TensorPtr>{self});
                output_tensor->set_grad_fn([self, n](const Tensor& upstream){
                    if (self->requires_grad()){
                        self->init_grad();
                        relu_grad_cuda(upstream.device_data(), self->device_data(), self->grad().mutable_device_data(), n);
                    }
                });

                return output_tensor;
            }
        #endif

        std::vector<size_t> new_input_shape = input->shape();
        auto output_tensor = Tensor::create(new_input_shape);

        for(size_t i = 0; i < input->numel(); i++){
            if(input->mutable_data()[i] < 0){
                output_tensor->mutable_data()[i] = 0.0f;
            }
            else{
                output_tensor->mutable_data()[i] = input->mutable_data()[i];
            }
        }

        /*
            need gradient calc here too!

            Took me a while but we don't update params but they still change
            so we want gradient to pass through the RELU layer too

            Challenges:
            - had to make many public functions to edit the grad_ and grad_fn_ of the input
            - public vs private

        */

        auto self = std::const_pointer_cast<Tensor>(input->shared_from_this());
        output_tensor->set_inputs(std::vector<TensorPtr>{self});

        output_tensor->set_grad_fn([self](const Tensor& upstream){
            if (self->requires_grad()){

                self->init_grad();

                //simple relu gradient: if input > 0, grad passes through, else blocked
                for (size_t i = 0; i < self->numel(); i++){
                    if (self->mutable_data()[i] > 0){
                        self->grad().mutable_data()[i] += upstream.data()[i];
                    }
                }
            }
        });
        output_tensor->set_requires_grad(input->requires_grad());
        return output_tensor;
    }
};
