#include "infer/paged_kv_cache.h"
#include <cassert>
#include <stdexcept>

PagedKVCache::PagedKVCache(size_t num_heads, size_t head_dim, size_t max_seq_len, size_t block_size){
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    block_size_ = block_size;

    //to the nearest block size
    capacity_seq_len_ = ((max_seq_len + block_size - 1) / block_size) * block_size; 
    cache_k_ = Tensor::create({num_heads_, capacity_seq_len_, head_dim_});
    cache_v_ = Tensor::create({num_heads_, capacity_seq_len_, head_dim_});
}

 void PagedKVCache::append(const TensorPtr& new_k, const TensorPtr& new_v){
    size_t new_seq_len = new_k->shape()[1]; //# of tokens being added 

    size_t new_total_len = curr_len_ + new_seq_len;

    if (new_total_len > capacity_seq_len_) {
        throw std::runtime_error("eviction not done yet");
    }

    //copy new
    for (size_t h = 0; h < num_heads_; h++){
        for (size_t i = 0; i < new_seq_len; i++){
            for (size_t j = 0; j < head_dim_; j++){
                cache_k_->at({h, curr_len_ + i, j}) = new_k->at({h, i, j});
                cache_v_->at({h, curr_len_ + i, j}) = new_v->at({h, i, j});
            }
        }
    }

    curr_len_ = new_total_len;

}

TensorPtr PagedKVCache::get_k() const {
    auto valid_k = Tensor::create({num_heads_, curr_len_, head_dim_});
    for (size_t h = 0; h < num_heads_; h++){
        for (size_t i = 0; i < curr_len_; i++){
            for (size_t j = 0; j < head_dim_; j++){
                valid_k->at({h, i, j}) = cache_k_->at({h, i, j});
            }
        }
    }
    return valid_k;
}

TensorPtr PagedKVCache::get_v() const {
    auto valid_v = Tensor::create({num_heads_, curr_len_, head_dim_});
    for (size_t h = 0; h < num_heads_; h++){
        for (size_t i = 0; i < curr_len_; i++){
            for (size_t j = 0; j < head_dim_; j++){
                valid_v->at({h, i, j}) = cache_v_->at({h, i, j});
            }
        }
    }
    return valid_v;
}

