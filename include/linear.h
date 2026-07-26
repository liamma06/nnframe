#pragma once 
#include <vector>
#include <memory> 
#include "tensor.h"
#include "layer.h"

class Linear : public Layer{
    private: 
        TensorPtr weights_; 
        TensorPtr bias_; 

    public:
        Linear(size_t in_features, size_t out_features);
        std::vector<TensorPtr> parameters() const override; //both wieght and bias
        
        //input-> [batch_size, in_features] output -> [batch_size, out_features]
        TensorPtr forward(const TensorPtr& input) override; 
};