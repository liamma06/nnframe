#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"
#include "loss/loss.h"

class MSE: public Loss{
    public: 
        TensorPtr forward(const TensorPtr& predictions, const TensorPtr& targets) override{
            auto diff = predictions->sub(targets);
            auto squared_diff = diff->mul(diff); 
            auto mean_squared_error = squared_diff->mean(); 

            return mean_squared_error;
        };

};