#include "cuda/attention_cuda.cuh"

__global__ void causal_mask_kernel(const scalar_t* d_scores, scalar_t* d_out, size_t heads, size_t seq_len_q, size_t seq_len_k) {
    /*
        (head,j, k) -> (head,j,k) if k <= j else -inf
    */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < heads * seq_len_q * seq_len_k) {
        size_t j = (idx / seq_len_k) % seq_len_q;
        size_t k = idx % seq_len_k;
        if (k > j) {
            d_out[idx] = -INFINITY; 
        } else {
            d_out[idx] = d_scores[idx];
        }
    }
}

void causal_mask_cuda(const scalar_t* d_scores, scalar_t* d_out, size_t heads, size_t seq_len_q, size_t seq_len_k) {
    //thread per entry
    size_t total_threads = heads * seq_len_q * seq_len_k;
    size_t blockDim = 256;
    size_t gridDim = (total_threads + blockDim - 1) / blockDim;
    causal_mask_kernel<<<gridDim, blockDim>>>(d_scores, d_out, heads, seq_len_q, seq_len_k);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void causal_mask_grad_kernel(const scalar_t* d_upstream, scalar_t* d_self_grad, size_t heads, size_t seq_len_q, size_t seq_len_k){
    /*
        (head,j,k) - > (head,j, k) -> (head,j,k) if k <= j else dont add upstream 
    */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < heads * seq_len_q * seq_len_k) {
        size_t j = (idx / seq_len_k) % seq_len_q;
        size_t k = idx % seq_len_k;
        if (j >= k) {
            d_self_grad[idx] += d_upstream[idx];
        } 
    }
}

void causal_mask_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, size_t heads, size_t seq_len_q, size_t seq_len_k) {
    size_t total_threads = heads * seq_len_q * seq_len_k;
    size_t blockDim = 256;
    size_t gridDim = (total_threads + blockDim - 1) / blockDim;
    causal_mask_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_grad, heads, seq_len_q, seq_len_k);
    CUDA_CHECK(cudaGetLastError());
}