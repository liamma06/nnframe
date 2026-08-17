#include "cuda/pos_embed_cuda.cuh"

__global__ void pos_embed_kernel(scalar_t* out, size_t seq_length, size_t embed_dim, size_t start_pos) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_length * embed_dim) {
        size_t pos = idx / embed_dim;
        size_t i = idx % embed_dim;
        size_t actual_pos = start_pos + pos;

        scalar_t angle = actual_pos / powf(10000.0f, (scalar_t)(i - (i % 2)) / embed_dim);
        out[idx] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

void pos_embed_cuda(scalar_t* d_out, size_t seq_length, size_t embed_dim, size_t start_pos) {
    size_t blockDim = 256;
    size_t gridDim = (seq_length * embed_dim + blockDim - 1) / blockDim;
    pos_embed_kernel<<<gridDim, blockDim>>>(d_out, seq_length, embed_dim, start_pos);
    CUDA_CHECK(cudaGetLastError());
}
