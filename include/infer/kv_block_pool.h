#pragma once
#include "core/tensor.h"
#include "infer/int8_quant.h"
#include <unordered_map>
#include <list>

class KVBlockPool {
    private:
        TensorPtr pool_k_;
        TensorPtr pool_v_;

        // CPU storage:
        std::vector<int8_t> pool_k_i8_;
        std::vector<int8_t> pool_v_i8_;
        std::vector<float> token_scales_k_;
        std::vector<float> token_scales_v_;
        // CUDA storage (device pointers, same layout):
        int8_t* pool_k_i8_dev_ = nullptr;
        int8_t* pool_v_i8_dev_ = nullptr;
        float* token_scales_k_dev_ = nullptr;
        float* token_scales_v_dev_ = nullptr;

        size_t num_heads_;
        size_t head_dim_;
        size_t block_size_;
        size_t max_blocks_;
        Device device_;
        bool use_quantization_;

        std::vector<size_t> free_blocks_;

    //each seq has it own state of how many blocks and tokens 
    struct SequenceState {
        std::vector<size_t> block_table;
        size_t tokens_in_last_block = 0;
    };
    std::unordered_map<size_t, SequenceState> sequences_;

    /*
        eviction:
        -LRU list of sequence ids
        -remove from related owner (block -> sequence id mapping)
    */
    std::list<size_t> lru_list_; 
    std::unordered_map<size_t, std::list<size_t>::iterator> lru_iters_; //quick look up 
    std::unordered_map<size_t, size_t> block_owner_; 

    void touch_block(size_t block_idx);
    size_t evict_block();


    public:

    KVBlockPool(size_t num_heads, size_t head_dim, size_t max_blocks, size_t block_size = 16, Device device = Device::CPU, bool use_quantization = false);

    void append(size_t sequence_id, const TensorPtr& k, const TensorPtr& v);
    TensorPtr get_k(size_t sequence_id) const;
    TensorPtr get_v(size_t sequence_id) const;
};