#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"

class Layer{
    public:
        virtual TensorPtr forward(const TensorPtr& input) = 0; // for subclass to define
        virtual ~Layer() = default;

        virtual std::vector<TensorPtr> parameters() const{
            return {};
        };
};
