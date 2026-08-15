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
}

__global__ void layernorm_grad_kernel(const scalar_t* upstream, const scalar_t* input, const scalar_t* gamma,
                                      scalar_t* input_grad, scalar_t* gamma_grad, scalar_t* beta_grad,
                                      size_t rows, size_t cols, scalar_t eps) {
    /*
        one block per row:
    */

    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    const scalar_t* row_in = input + row * cols;
    const scalar_t* row_upstream = upstream + row * cols;
    scalar_t* row_grad = input_grad + row * cols;

    __shared__ scalar_t sdata[256];

    // recompute mean
    scalar_t local_sum = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x) local_sum += row_in[j];
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    scalar_t mean = sdata[0] / (scalar_t)cols;
    __syncthreads();

    // recompute variance
    local_sum = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x) {
        scalar_t diff = row_in[j] - mean;
        local_sum += diff * diff;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    scalar_t variance = sdata[0] / (scalar_t)cols;
    __syncthreads();

    scalar_t inv_std = 1.0f / sqrtf(variance + eps);

    // gamma/beta gradients -> accumulated across every row, so atomicAdd across blocks
    for (size_t j = tid; j < cols; j += blockDim.x) {
        scalar_t x_hat = (row_in[j] - mean) * inv_std;
        atomicAdd(&gamma_grad[j], row_upstream[j] * x_hat);
        atomicAdd(&beta_grad[j], row_upstream[j]);
    }

    scalar_t local_sum_upstream = 0.0f;
    scalar_t local_sum_upstream_xhat = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x) {
        scalar_t x_hat = (row_in[j] - mean) * inv_std;
        local_sum_upstream += row_upstream[j] * gamma[j];
        local_sum_upstream_xhat += row_upstream[j] * gamma[j] * x_hat;
    }

    sdata[tid] = local_sum_upstream;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    scalar_t sum_upstream = sdata[0];
    __syncthreads();

    sdata[tid] = local_sum_upstream_xhat;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    scalar_t sum_upstream_xhat = sdata[0];
    __syncthreads();

    // final 
    for (size_t j = tid; j < cols; j += blockDim.x) {
        scalar_t x_hat = (row_in[j] - mean) * inv_std;
        row_grad[j] += inv_std / (scalar_t)cols * ((scalar_t)cols * row_upstream[j] * gamma[j] - sum_upstream - x_hat * sum_upstream_xhat);
    }
}

void layernorm_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_input, const scalar_t* d_gamma,
                          scalar_t* d_input_grad, scalar_t* d_gamma_grad, scalar_t* d_beta_grad,
                          size_t rows, size_t cols, scalar_t eps) {
    dim3 block(256);
    dim3 grid(rows);
    layernorm_grad_kernel<<<grid, block>>>(d_upstream, d_input, d_gamma, d_input_grad, d_gamma_grad, d_beta_grad, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}