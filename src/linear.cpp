#include "modules/linear.h"
#include <random>
#include <cmath>

#if defined(NNFRAME_WITH_CUDA)
    #include "cuda/linear_cuda.cuh"
#endif

std::mt19937 rng(42);

Linear::Linear(size_t in_features, size_t out_features){
    weights_ = Tensor::create({in_features, out_features});
    bias_ = Tensor::create({out_features});
    weights_->set_requires_grad(true);
    bias_->set_requires_grad(true);

    // Xavier-style scale
    float scale = 1.0f / std::sqrt(static_cast<float>(in_features));
    std::uniform_real_distribution<float> dist(-scale, scale);

    //random (seed) initialzie
    for (size_t i = 0; i < weights_->numel(); i++){
        weights_->mutable_data()[i] = dist(rng);
    }
}

std::vector<TensorPtr> Linear::parameters() const{
    return {weights_, bias_}; 
}

TensorPtr Linear::forward(const TensorPtr& input){
    /*
        input: [batch_size, in_features]
        weights: [in_features, out_features]
        output: [batch_size, out_features]
        y=xW + b
    */


    TensorPtr output_xW = input->matmul(weights_);

    #if defined(NNFRAME_WITH_CUDA)
        if (output_xW->device() == Device::CUDA){
            size_t batch_size = output_xW->shape()[0];
            size_t out_features = output_xW->shape()[1];
            size_t n = output_xW->numel();

            scalar_t* d_out = nullptr;
            CUDA_CHECK(cudaMalloc(&d_out, n*sizeof(scalar_t)));

            add_bias_cuda(output_xW->device_data(), bias_->device_data(), d_out, batch_size, out_features);

            TensorPtr output = Tensor::from_device_ptr(d_out, output_xW->shape(), output_xW->strides());

            //doesnt auto set like CPU add 
            auto self = output_xW;
            auto bias = bias_;
            output->set_requires_grad(self->requires_grad() || bias->requires_grad());
            output->set_inputs(std::vector<TensorPtr>{self, bias});
            output->set_grad_fn([self, bias, batch_size, out_features](const Tensor& upstream){
                if (self->requires_grad() || bias->requires_grad()){
                    self->init_grad();
                    bias->init_grad();
                    add_bias_grad_cuda(upstream.device_data(), self->grad().mutable_device_data(), bias->grad().mutable_device_data(), batch_size, out_features);
                }
            });

            return output;
        }
    #endif

    TensorPtr output = output_xW->add(bias_);
    return output;
}