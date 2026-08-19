#include "infer/paged_kv_cache.h"
#include <cassert>
#include <stdexcept>

#if defined(NNFRAME_WITH_CUDA)
    #include "cuda/matmul_cuda.cuh"
#endif

PagedKVCache::PagedKVCache(size_t num_heads, size_t head_dim, size_t max_seq_len, size_t block_size, Device device){
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    block_size_ = block_size;
    device_ = device;

    //to the nearest block size
    capacity_seq_len_ = ((max_seq_len + block_size - 1) / block_size) * block_size; 

    #ifdef NNFRAME_WITH_CUDA
    if (device_ == Device::CUDA){
        scalar_t* d_k = nullptr;
        scalar_t* d_v = nullptr;

        size_t size_kv = num_heads_ * capacity_seq_len_ * head_dim_ * sizeof(scalar_t);

        CUDA_CHECK(cudaMallocAsync(&d_k, size_kv, 0));
        CUDA_CHECK(cudaMallocAsync(&d_v, size_kv, 0));

        std::vector<size_t> shape = {num_heads_, capacity_seq_len_, head_dim_};
        std::vector<size_t> strides = {capacity_seq_len_ * head_dim_, head_dim_, 1};

        cache_k_ = Tensor::from_device_ptr(d_k, shape, strides);
        cache_v_ = Tensor::from_device_ptr(d_v, shape, strides);

        return;
    }
    #endif

    cache_k_ = Tensor::create({num_heads_, capacity_seq_len_, head_dim_});
    cache_v_ = Tensor::create({num_heads_, capacity_seq_len_, head_dim_});
}

 void PagedKVCache::append(const TensorPtr& new_k, const TensorPtr& new_v){
    size_t new_seq_len = new_k->shape()[1]; //# of tokens being added 

    size_t new_total_len = curr_len_ + new_seq_len;

    if (new_total_len > capacity_seq_len_) {
        throw std::runtime_error("eviction not done yet");
    }

    #ifdef NNFRAME_WITH_CUDA
        //similar to the normal KVcahce imp but instead of copy old data just input 
        if (device_ == Device::CUDA){
            for (size_t h = 0; h < num_heads_; h++){
                size_t new_data_offset_location = h * new_seq_len * head_dim_; // offset from input
                size_t new_data_offset_landing = h * capacity_seq_len_ * head_dim_ + curr_len_ * head_dim_;

                CUDA_CHECK(cudaMemcpyAsync(cache_k_->mutable_device_data() + new_data_offset_landing, new_k->device_data() + new_data_offset_location, new_seq_len * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(cache_v_->mutable_device_data() + new_data_offset_landing, new_v->device_data() + new_data_offset_location, new_seq_len * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
            }
            
            curr_len_ = new_total_len;
            return;
        }
    #endif

    //copy new into the empty space 
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
    #ifdef NNFRAME_WITH_CUDA
        if (device_ == Device::CUDA){
            scalar_t* d_out = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_out, num_heads_ * curr_len_ * head_dim_ * sizeof(scalar_t), 0));

            for (size_t h = 0; h < num_heads_; h++){
                size_t src_offset = h * capacity_seq_len_ * head_dim_;
                size_t dst_offset = h * curr_len_ * head_dim_;
                CUDA_CHECK(cudaMemcpyAsync(d_out + dst_offset, cache_k_->device_data() + src_offset, curr_len_ * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
            }

            std::vector<size_t> shape = {num_heads_, curr_len_, head_dim_};
            std::vector<size_t> strides = {curr_len_ * head_dim_, head_dim_, 1};
            return Tensor::from_device_ptr(d_out, shape, strides);
        }
    #endif

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
    #ifdef NNFRAME_WITH_CUDA
        if (device_ == Device::CUDA){
            scalar_t* d_out = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_out, num_heads_ * curr_len_ * head_dim_ * sizeof(scalar_t), 0));

            for (size_t h = 0; h < num_heads_; h++){
                size_t src_offset = h * capacity_seq_len_ * head_dim_;
                size_t dst_offset = h * curr_len_ * head_dim_;
                CUDA_CHECK(cudaMemcpyAsync(d_out + dst_offset, cache_v_->device_data() + src_offset,
                                            curr_len_ * head_dim_ * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
            }

            std::vector<size_t> shape = {num_heads_, curr_len_, head_dim_};
            std::vector<size_t> strides = {curr_len_ * head_dim_, head_dim_, 1};
            return Tensor::from_device_ptr(d_out, shape, strides);
        }
    #endif

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

