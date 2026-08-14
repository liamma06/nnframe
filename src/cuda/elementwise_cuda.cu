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

__global__ void log_kernel(const scalar_t* in, scalar_t* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(in[i]); 
}

void log_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    log_kernel<<<gridDim, blockDim>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void sum_reduce_kernel(const scalar_t* in, scalar_t* out, size_t n ){
    /*
        -reserve fast shared mem per block
        -each thread loads one element from global to shared mem
        -wait till all threads finish loading
        -do reduction in shared mem
        -final result add to global mem using atomicAdd
    */

    __shared__ scalar_t sdata[256];

    size_t tid = threadIdx.x; //where on hte shared memory for this block
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    //sum into global mem out
    if (tid == 0) {
        atomicAdd(out, sdata[0]); //end of reduction everything in sdata[0]
    }

}

void sum_reduce_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n) {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(scalar_t))); 

    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    sum_reduce_kernel<<<gridDim, blockDim>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void scale_scalar_kernel(scalar_t* val, scalar_t factor) {
    val[0] *= factor;
}

void scale_scalar_cuda(scalar_t* d_val, scalar_t factor) {
    scale_scalar_kernel<<<1, 1>>>(d_val, factor); // single value
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}