#include "layers/linear.h"
#include <random>

std::mt19937 rng(42);
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

Linear::Linear(size_t in_features, size_t out_features){
    weights_ = Tensor::create({in_features, out_features});
    bias_ = Tensor::create({out_features});
    weights_->set_requires_grad(true);
    bias_->set_requires_grad(true);

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
    TensorPtr output = output_xW->add(bias_);

    return output;

}