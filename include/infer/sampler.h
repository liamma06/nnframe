#include <random>
#include "core/tensor.h"
#pragma once

class Sampler{
    private: 
        std::mt19937 rng_;
    
    public:
        Sampler(unsigned seed = 42);
        size_t sample(const TensorPtr& logits, float temperature, size_t top_k);

};
