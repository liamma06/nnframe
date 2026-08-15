#include "cuda/embed_cuda.cuh"

__global__ void embed_kernel(const scalar_t* d_table, const scalar_t* d_indices, scalar_t* d_out, size_t seq_len, size_t embedding_dim){
    // EACH dimension gets its own thread 
    //goal: look up indexes in the d_table and copy into d_out.

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
    size_t total = seq_len * embedding_dim;

    if (idx < total){
        size_t pos = idx / embedding_dim; //pos
        size_t d = idx % embedding_dim; //dimension IN that position's embedding vecotr 

        size_t token = (size_t)d_indices[pos]; 
        d_out[idx] = d_table[token * embedding_dim + d]; 
    }
}

void embed_cuda(const scalar_t* d_table, const scalar_t* d_indices, scalar_t* d_out, size_t seq_len, size_t embedding_dim) {
    size_t total = seq_len * embedding_dim;
    dim3 blockDim(256);
    size_t gridDim = (total + blockDim.x - 1) / blockDim.x;
    embed_kernel<<<gridDim, blockDim>>>(d_table, d_indices, d_out, seq_len, embedding_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void embed_backward_kernel(const scalar_t* d_indices, const scalar_t* d_grad_out, scalar_t* d_grad_table, size_t seq_len, size_t embedding_dim){
    //goal: for each index in d_indices, add the corresponding gradient from d_grad_out
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * embedding_dim;

    if (idx < total){
        size_t pos = idx / embedding_dim; 
        size_t d = idx % embedding_dim; 

        size_t token = (size_t)d_indices[pos]; 

        //since multiple threads may write to same token, atomic add to avoid problems
        //but is slower 
        atomicAdd(&d_grad_table[token * embedding_dim + d], d_grad_out[idx]); 
    }
}

void embed_backward_cuda(const scalar_t* d_indices, const scalar_t* d_grad_out, scalar_t* d_grad_table, size_t seq_len, size_t embedding_dim) {
    size_t total = seq_len * embedding_dim;
    dim3 blockDim(256);
    size_t gridDim = (total + blockDim.x - 1) / blockDim.x;
    embed_backward_kernel<<<gridDim, blockDim>>>(d_indices, d_grad_out, d_grad_table, seq_len, embedding_dim);
    CUDA_CHECK(cudaGetLastError());
}