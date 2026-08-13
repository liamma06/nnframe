#include "cuda/elementwise_cuda.cuh"

/*
    elementwise ops -> one thread per element
*/

__global__ void add_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void sub_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void mul_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

void add_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    add_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sub_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    sub_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mul_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    mul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
