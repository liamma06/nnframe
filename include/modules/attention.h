#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"
#include "modules/layer.h"
#include <cassert>

class SelfAttention: public Layer{
    private:
        size_t embed_dim_;
        size_t num_heads_;
        size_t head_dim_;
        TensorPtr W_q_;
        TensorPtr W_k_;
        TensorPtr W_v_;

        /*
            Wo confused me but in multihead where we have multiple heads each with its own weight matrix.
            We need a way to concatenate outputs of all heads and Wo is where we mix 
            in the contributions into one final output applied to the embedding.
        */
        TensorPtr W_o_; 

    public:
        SelfAttention(size_t embed_dim, size_t num_heads = 1);
        std::vector<TensorPtr> parameters() const override;
        TensorPtr forward(const TensorPtr& input) override;
} ;