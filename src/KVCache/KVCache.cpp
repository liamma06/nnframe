#include "infer/kv_cache.h"
#include <cassert>

#if defined(NNFRAME_WITH_CUDA)
    #include "cuda/matmul_cuda.cuh"
#endif

TensorPtr KVCache::get_k() const {
    return cache_k_;
}

TensorPtr KVCache::get_v() const {
    return cache_v_;
}

TensorPtr KVCache::grow(const TensorPtr& old_tensor, const TensorPtr& new_tensor) {
    assert(old_tensor->rank() == 3 && new_tensor->rank() == 3 && "Both tensors must be rank 3");
    assert(old_tensor->shape()[0] == new_tensor->shape()[0] && old_tensor->shape()[2] == new_tensor->shape()[2] && "Tensors must have the same num_heads and head_dim");

    std::vector<size_t> new_shape = old_tensor->shape();
    new_shape[1] += new_tensor->shape()[1]; // increase seq_len

    auto grown_tensor = Tensor::create(new_shape);

    #ifdef NNFRAME_WITH_CUDA
        if(old_tensor->device() == Device::CUDA ){
            scalar_t* d_grown = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_grown, grown_tensor->numel()*sizeof(scalar_t), 0));

            size_t heads = old_tensor->shape()[0];

            for (size_t i = 0; i < heads; i++){
                size_t old_seq_len = old_tensor->shape()[1];
                size_t delta = new_tensor->shape()[1]; //how much we adding
                size_t total_seq_len = old_seq_len + delta;
                size_t head_dim = old_tensor->shape()[2];

                size_t old_data_offset_location = i * old_seq_len * head_dim;
                size_t old_data_offset_landing = i * total_seq_len * head_dim;

                size_t new_data_offset_location = i * delta * head_dim;
                size_t new_data_offset_landing = i * total_seq_len * head_dim + old_seq_len * head_dim;

                CUDA_CHECK(cudaMemcpyAsync(d_grown +old_data_offset_landing, old_tensor->device_data() + old_data_offset_location, old_seq_len * head_dim * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(d_grown +new_data_offset_landing, new_tensor->device_data() + new_data_offset_location, delta * head_dim * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
            }

            // row-major contiguous strides for new_shape
            std::vector<size_t> new_strides(new_shape.size());
            size_t stride = 1;
            for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; i--){
                new_strides[i] = stride;
                stride *= new_shape[i];
            }

            return Tensor::from_device_ptr(d_grown, new_shape, new_strides);
        }
    #endif
    
    // Copy old data
    for (size_t i = 0; i < old_tensor->shape()[0]; ++i) {
        for (size_t j = 0; j < old_tensor->shape()[1]; ++j) {
            for (size_t k = 0; k < old_tensor->shape()[2]; ++k) {
                grown_tensor->at({i, j, k}) = old_tensor->at({i, j, k});
            }
        }
    }

    // Copy new data into
    for (size_t i = 0; i < new_tensor->shape()[0]; ++i) {
        for (size_t j = 0; j < new_tensor->shape()[1]; ++j) {
            for (size_t k = 0; k < new_tensor->shape()[2]; ++k) {
                grown_tensor->at({i, j + old_tensor->shape()[1], k}) = new_tensor->at({i, j, k});
            }
        }
    }

    return grown_tensor;
}

void KVCache::append(const TensorPtr& k, const TensorPtr& v){
    /*
        append into caches 
        k: [num_heads, seq_len, head_dim]
        v: [num_heads, seq_len, head_dim]
        concat(stuff there already) or store directly(first item)
    */
    assert(k->shape() == v->shape() && "K and V must have the same shape");
    assert(k->rank() == 3 && v->rank() == 3 && "K and V must be rank 3 tensors");

    if (!cache_k_ && !cache_v_){
        cache_k_ = k;
        cache_v_ = v;
    }
    else{
        cache_k_ = grow(cache_k_, k);
        cache_v_ = grow(cache_v_, v);
    }

}