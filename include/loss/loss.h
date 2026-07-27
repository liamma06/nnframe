#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"

class Loss{
    public:
        virtual TensorPtr forward(const TensorPtr& predictions, const TensorPtr& targets) = 0; 
        virtual ~Loss() = default;
};