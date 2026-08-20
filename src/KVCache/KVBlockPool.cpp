#include "infer/kv_block_pool.h"
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <vector>

#if defined(NNFRAME_WITH_CUDA)
    #include "cuda/matmul_cuda.cuh"
    #include "cuda/int8_quant_cuda.cuh"
#endif

KVBlockPool::KVBlockPool(size_t num_heads, size_t head_dim, size_t max_blocks, size_t block_size, Device device, bool use_quantization){
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    block_size_ = block_size;
    max_blocks_ = max_blocks;
    device_ = device;
    use_quantization_ = use_quantization;

    for(size_t i = 0; i < max_blocks_; i++){
        free_blocks_.push_back(i);
    }

    #ifdef NNFRAME_WITH_CUDA
        if (device_ == Device::CUDA) {
            if (use_quantization_) {
                size_t total_slots = num_heads_ * max_blocks_ * block_size_;

                CUDA_CHECK(cudaMallocAsync(&pool_k_i8_dev_, total_slots * head_dim_ * sizeof(int8_t), 0));
                CUDA_CHECK(cudaMallocAsync(&pool_v_i8_dev_, total_slots * head_dim_ * sizeof(int8_t), 0));
                CUDA_CHECK(cudaMallocAsync(&token_scales_k_dev_, total_slots * sizeof(float), 0));
                CUDA_CHECK(cudaMallocAsync(&token_scales_v_dev_, total_slots * sizeof(float), 0));

                return;
            }

            scalar_t* d_k = nullptr;
            scalar_t* d_v = nullptr;

            size_t size_kv = num_heads_ * max_blocks_ * block_size_ * head_dim_ * sizeof(scalar_t);

            CUDA_CHECK(cudaMallocAsync(&d_k, size_kv, 0));
            CUDA_CHECK(cudaMallocAsync(&d_v, size_kv, 0));

            std::vector<size_t> shape = {num_heads_, max_blocks_ * block_size_, head_dim_};
            std::vector<size_t> strides = {max_blocks_ * block_size_ * head_dim_, head_dim_, 1};

            pool_k_ = Tensor::from_device_ptr(d_k, shape, strides);
            pool_v_ = Tensor::from_device_ptr(d_v, shape, strides);

            return;
        }
    #endif

    if (use_quantization_) {
        /*
            slot -> since token split amoung heads -> every token's head part 
            pool -> slot itself is how many vectors -> giving it the dimension
            token_scales -> one per token head part -> spread across. 
        */

        size_t total_slots = num_heads_ * max_blocks_ * block_size_; 
        
        
        pool_k_i8_.assign(total_slots * head_dim_, 0);
        pool_v_i8_.assign(total_slots * head_dim_, 0);
        token_scales_k_.assign(total_slots, 1.0f);
        token_scales_v_.assign(total_slots, 1.0f);
        return;
    }

    pool_k_ = Tensor::create({num_heads_, max_blocks_ * block_size_, head_dim_});
    pool_v_ = Tensor::create({num_heads_, max_blocks_ * block_size_, head_dim_});
}

//evict 
void KVBlockPool::touch_block(size_t block_idx){
    auto it = lru_iters_.find(block_idx);
    if (it != lru_iters_.end()) {
        lru_list_.erase(it->second);
    }
    //update to front of list
    lru_list_.push_front(block_idx);
    lru_iters_[block_idx] = lru_list_.begin();
}

size_t KVBlockPool::evict_block(){
    if (lru_list_.empty()) {
        throw std::runtime_error("No blocks to evict");
    }

    //remove from list 
    size_t victim = lru_list_.back();
    lru_list_.pop_back();
    lru_iters_.erase(victim);

    //remove from owner sequence
    size_t owner_id = block_owner_.at(victim);
    SequenceState& owner_state = sequences_.at(owner_id);
    auto& table = owner_state.block_table;
    table.erase(std::remove(table.begin(), table.end(), victim), table.end());

    block_owner_.erase(victim);
    return victim;
}
    

void KVBlockPool::append(size_t sequence_id, const TensorPtr& k, const TensorPtr& v){
    SequenceState& state = sequences_[sequence_id]; //get or create 
    size_t new_seq_len = k -> shape()[1]; //num of new tokens to add 
    size_t src_pos = 0; 

    while (src_pos < new_seq_len){
        //grab fresh block or curr block full
        if (state.block_table.empty() || state.tokens_in_last_block == block_size_){
            size_t new_block;
            if (!free_blocks_.empty()){
                new_block = free_blocks_.back();
                free_blocks_.pop_back();
            } else {
                new_block = evict_block();
            }
            state.block_table.push_back(new_block);
            block_owner_[new_block] = sequence_id;
            state.tokens_in_last_block = 0;
        }

        size_t block_idx = state.block_table.back();
        touch_block(block_idx); // mark as recently use
        size_t room = block_size_ - state.tokens_in_last_block;
        size_t tokens_to_copy = std::min(room, new_seq_len - src_pos); //what is possible

        #ifdef NNFRAME_WITH_CUDA
            if (device_ == Device::CUDA){
                for (size_t head = 0; head < num_heads_; head++){
                    size_t src_offset = head * new_seq_len * head_dim_ + src_pos * head_dim_;

                    if (use_quantization_) {
                        size_t slot = block_idx * block_size_ + state.tokens_in_last_block;
                        size_t scale_idx = head * max_blocks_ * block_size_ + slot;
                        size_t data_offset = scale_idx * head_dim_;

                        quantize_cuda(k->device_data() + src_offset, pool_k_i8_dev_ + data_offset, token_scales_k_dev_ + scale_idx, tokens_to_copy, head_dim_);
                        quantize_cuda(v->device_data() + src_offset, pool_v_i8_dev_ + data_offset, token_scales_v_dev_ + scale_idx, tokens_to_copy, head_dim_);
                        continue;
                    }

                    size_t dst_offset = head * max_blocks_ * block_size_ * head_dim_ + block_idx * block_size_ * head_dim_ + state.tokens_in_last_block * head_dim_;

                    CUDA_CHECK(cudaMemcpyAsync(pool_k_->mutable_device_data() + dst_offset, k->device_data() + src_offset, tokens_to_copy * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpyAsync(pool_v_->mutable_device_data() + dst_offset, v->device_data() + src_offset, tokens_to_copy * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                }

                state.tokens_in_last_block += tokens_to_copy;
                src_pos += tokens_to_copy;
                continue;
            }
        #endif

        if (use_quantization_) {
            std::vector<float> k_slice(head_dim_), v_slice(head_dim_);

            for (size_t head = 0; head < num_heads_; head++){
                for (size_t token = 0; token < tokens_to_copy; token++){
                    size_t slot = block_idx * block_size_ + state.tokens_in_last_block + token;
                    size_t scale_idx = head * max_blocks_ * block_size_ + slot;
                    size_t data_offset = scale_idx * head_dim_;

                    for (size_t dim = 0; dim < head_dim_; dim++){
                        k_slice[dim] = k -> at({head, src_pos + token, dim});
                        v_slice[dim] = v -> at({head, src_pos + token, dim});
                    }

                    token_scales_k_[scale_idx] = quantize(k_slice.data(), &pool_k_i8_[data_offset], head_dim_);
                    token_scales_v_[scale_idx] = quantize(v_slice.data(), &pool_v_i8_[data_offset], head_dim_);
                }
            }
        } else {
            for (size_t head = 0; head < num_heads_; head++){
                for (size_t token = 0; token < tokens_to_copy; token++){
                    for (size_t dim = 0; dim < head_dim_; dim++){
                        pool_k_ -> at({head, block_idx * block_size_ + state.tokens_in_last_block + token, dim}) = k -> at({head, src_pos + token, dim});
                        pool_v_ -> at({head, block_idx * block_size_ + state.tokens_in_last_block + token, dim}) = v -> at({head, src_pos + token, dim});
                    }
                }
            }
        }

        state.tokens_in_last_block += tokens_to_copy;
        src_pos += tokens_to_copy;

    }
}

TensorPtr KVBlockPool::get_k(size_t sequence_id) const {
    const SequenceState& state = sequences_.at(sequence_id);

    size_t total_tokens = 0;
    if (!state.block_table.empty()) {
        total_tokens = (state.block_table.size() - 1) * block_size_ + state.tokens_in_last_block;
    }

    #ifdef NNFRAME_WITH_CUDA
        if (device_ == Device::CUDA) {
            scalar_t* d_out = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_out, num_heads_ * total_tokens * head_dim_ * sizeof(scalar_t), 0));

            size_t dst_pos = 0;
            for (size_t b = 0; b < state.block_table.size(); b++) {
                size_t block_idx = state.block_table[b];
                size_t tokens_in_block = (b == state.block_table.size() - 1) ? state.tokens_in_last_block : block_size_;

                for (size_t head = 0; head < num_heads_; head++) {
                    size_t dst_offset = head * total_tokens * head_dim_ + dst_pos * head_dim_;

                    if (use_quantization_) {
                        size_t slot = block_idx * block_size_;
                        size_t scale_idx = head * max_blocks_ * block_size_ + slot;
                        size_t data_offset = scale_idx * head_dim_;
                        dequantize_cuda(pool_k_i8_dev_ + data_offset, d_out + dst_offset, token_scales_k_dev_ + scale_idx, tokens_in_block, head_dim_);
                        continue;
                    }

                    size_t src_offset = head * max_blocks_ * block_size_ * head_dim_ + block_idx * block_size_ * head_dim_;
                    CUDA_CHECK(cudaMemcpyAsync(d_out + dst_offset, pool_k_->device_data() + src_offset,tokens_in_block * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                }
                dst_pos += tokens_in_block;
            }

            std::vector<size_t> shape = {num_heads_, total_tokens, head_dim_};
            std::vector<size_t> strides = {total_tokens * head_dim_, head_dim_, 1};
            return Tensor::from_device_ptr(d_out, shape, strides);
        }
    #endif

    auto out = Tensor::create({num_heads_, total_tokens, head_dim_});

    size_t dst_pos = 0;
    for (size_t b = 0; b < state.block_table.size(); b++) {
        size_t block_idx = state.block_table[b];
        size_t tokens_in_block = (b == state.block_table.size() - 1) ? state.tokens_in_last_block : block_size_;

        if (use_quantization_) {
            std::vector<float> k_slice(head_dim_);
            for (size_t head = 0; head < num_heads_; head++) {
                for (size_t token = 0; token < tokens_in_block; token++) {
                    size_t slot = block_idx * block_size_ + token;
                    size_t scale_idx = head * max_blocks_ * block_size_ + slot;
                    size_t data_offset = scale_idx * head_dim_;

                    dequantize(&pool_k_i8_[data_offset], k_slice.data(), head_dim_, token_scales_k_[scale_idx]);
                    for (size_t dim = 0; dim < head_dim_; dim++) {
                        out->at({head, dst_pos + token, dim}) = k_slice[dim];
                    }
                }
            }
        } else {
            for (size_t head = 0; head < num_heads_; head++) {
                for (size_t token = 0; token < tokens_in_block; token++) {
                    for (size_t dim = 0; dim < head_dim_; dim++) {
                        out->at({head, dst_pos + token, dim}) = pool_k_->at({head, block_idx * block_size_ + token, dim});
                    }
                }
            }
        }
        dst_pos += tokens_in_block;
    }

    return out;
}

TensorPtr KVBlockPool::get_v(size_t sequence_id) const {
    const SequenceState& state = sequences_.at(sequence_id);

    size_t total_tokens = 0;
    if (!state.block_table.empty()) {
        total_tokens = (state.block_table.size() - 1) * block_size_ + state.tokens_in_last_block;
    }

    #ifdef NNFRAME_WITH_CUDA
        if (device_ == Device::CUDA) {
            scalar_t* d_out = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_out, num_heads_ * total_tokens * head_dim_ * sizeof(scalar_t), 0));

            size_t dst_pos = 0;
            for (size_t b = 0; b < state.block_table.size(); b++) {
                size_t block_idx = state.block_table[b];
                size_t tokens_in_block = (b == state.block_table.size() - 1) ? state.tokens_in_last_block : block_size_;

                for (size_t head = 0; head < num_heads_; head++) {
                    size_t dst_offset = head * total_tokens * head_dim_ + dst_pos * head_dim_;

                    if (use_quantization_) {
                        size_t slot = block_idx * block_size_;
                        size_t scale_idx = head * max_blocks_ * block_size_ + slot;
                        size_t data_offset = scale_idx * head_dim_;
                        dequantize_cuda(pool_v_i8_dev_ + data_offset, d_out + dst_offset, token_scales_v_dev_ + scale_idx, tokens_in_block, head_dim_);
                        continue;
                    }

                    size_t src_offset = head * max_blocks_ * block_size_ * head_dim_ + block_idx * block_size_ * head_dim_;
                    CUDA_CHECK(cudaMemcpyAsync(d_out + dst_offset, pool_v_->device_data() + src_offset, tokens_in_block * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                }
                dst_pos += tokens_in_block;
            }

            std::vector<size_t> shape = {num_heads_, total_tokens, head_dim_};
            std::vector<size_t> strides = {total_tokens * head_dim_, head_dim_, 1};
            return Tensor::from_device_ptr(d_out, shape, strides);
        }
    #endif

    auto out = Tensor::create({num_heads_, total_tokens, head_dim_});

    size_t dst_pos = 0;
    for (size_t b = 0; b < state.block_table.size(); b++) {
        size_t block_idx = state.block_table[b];
        size_t tokens_in_block = (b == state.block_table.size() - 1) ? state.tokens_in_last_block : block_size_;

        if (use_quantization_) {
            std::vector<float> v_slice(head_dim_);
            for (size_t head = 0; head < num_heads_; head++) {
                for (size_t token = 0; token < tokens_in_block; token++) {
                    size_t slot = block_idx * block_size_ + token;
                    size_t scale_idx = head * max_blocks_ * block_size_ + slot;
                    size_t data_offset = scale_idx * head_dim_;

                    dequantize(&pool_v_i8_[data_offset], v_slice.data(), head_dim_, token_scales_v_[scale_idx]);
                    for (size_t dim = 0; dim < head_dim_; dim++) {
                        out->at({head, dst_pos + token, dim}) = v_slice[dim];
                    }
                }
            }
        } else {
            for (size_t head = 0; head < num_heads_; head++) {
                for (size_t token = 0; token < tokens_in_block; token++) {
                    for (size_t dim = 0; dim < head_dim_; dim++) {
                        out->at({head, dst_pos + token, dim}) = pool_v_->at({head, block_idx * block_size_ + token, dim});
                    }
                }
            }
        }
        dst_pos += tokens_in_block;
    }

    return out;
}