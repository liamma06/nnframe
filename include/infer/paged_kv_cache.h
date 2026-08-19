#pragma once
#include "core/tensor.h"

class PagedKVCache {
    private: 
        TensorPtr cache_k_;
        TensorPtr cache_v_;

        size_t num_heads_;
        size_t head_dim_;
        size_t block_size_; //# of tokens per block
        size_t capacity_seq_len_;
        size_t curr_len_ = 0;
        Device device_;

    public:
            PagedKVCache(size_t num_heads, size_t head_dim, size_t max_seq_len, size_t block_size = 16, Device device = Device::CPU);

            void append(const TensorPtr& new_k, const TensorPtr& new_v);
            TensorPtr get_k() const;
            TensorPtr get_v() const;
};