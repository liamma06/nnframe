#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"

class KVCache{
    private: 
        /*
            to fit into our attention module:
            - K_head: [num_heads, seq_len, head_dim]
            - V_head: [num_heads, seq_len, head_dim]
            need ot match to plug into. 
        */
        TensorPtr cache_k_;
        TensorPtr cache_v_;

        static TensorPtr grow(const TensorPtr& old_tensor, const TensorPtr& new_tensor);

    public:
        void append(const TensorPtr& k, const TensorPtr& v);
        TensorPtr get_k() const;
        TensorPtr get_v() const;

        //would need eviction maybe when GPU added 
        

};