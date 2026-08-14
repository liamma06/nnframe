#include "cuda/layernorm_cuda.cuh"

__global__ void layernorm_kernel(const scalar_t* d_input, const scalar_t* d_gamma, const scalar_t* d_beta, scalar_t* d_output, size_t rows, size_t cols, scalar_t eps) {
    /*
        One block per row 
    */

    size_t row = blockIdx.x; 
    size_t tid = threadIdx.x;

    const scalar_t* row_in = d_input + row * cols;
    scalar_t* row_out = d_output + row * cols;

    //row mean 
    scalar_t local_sum = 0.0f;
    for (size_t i = tid; i < cols; i += blockDim.x) {
        local_sum += row_in[i];
    }

    __shared__ scalar_t sdata[256];
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] = sdata[tid] + sdata[tid + s]; //sum everything togther 
        }
        __syncthreads();
    }

    //now sum in sdata[0] 
    scalar_t mean = sdata[0] / (scalar_t) cols;
    __syncthreads();

    //row variance
    local_sum = 0.0f;
    for (size_t i = tid; i < cols; i += blockDim.x) {
        scalar_t diff = row_in[i] - mean;
        local_sum += diff * diff;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] = sdata[tid] + sdata[tid + s]; 
        }
        __syncthreads();
    }

    scalar_t variance = sdata[0] / (scalar_t) cols;

    //normalize 
    scalar_t inv_std = 1.0f / sqrtf(variance + eps);
    for (size_t i = tid; i < cols; i += blockDim.x) {
        scalar_t x_hat = (row_in[i] - mean) * inv_std;
        row_out[i] = d_gamma[i] * x_hat + d_beta[i];
    }

}

void layernorm_cuda(const scalar_t* d_input, const scalar_t* d_gamma, const scalar_t* d_beta, scalar_t* d_output, size_t rows, size_t cols, scalar_t eps) {
    dim3 block(256);
    dim3 grid(rows);
    layernorm_kernel<<<grid, block>>>(d_input, d_gamma, d_beta, d_output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}