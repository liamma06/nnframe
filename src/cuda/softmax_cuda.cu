#include "cuda/softmax_cuda.cuh"

__global__ void softmax_kernel(const scalar_t* d_in, scalar_t* d_out, size_t rows, size_t row_size){
    /*
        one block per row. parallel across rows instead of just the same threads doing the computations
        - find max (each thread handle more than one element)
        - sharememory and load max into shared memory
        - compute exp(x - max) for each element 
        - sum the exponentials
        - divide each exponential by the sum

    */
    
    size_t row = blockIdx.x; 
    size_t tid = threadIdx.x;

    const scalar_t* row_in = d_in + row * row_size;
    scalar_t* row_out = d_out + row * row_size;

    scalar_t local_max = -INFINITY;
    for (size_t j = tid; j < row_size; j += blockDim.x){
        if (row_in[j] > local_max){
            local_max = row_in[j];
        }
    }

    __shared__ scalar_t sdata[256];
    sdata[tid] = local_max;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    scalar_t row_max = sdata[0];
    __syncthreads();


    scalar_t local_sum = 0.0f;
    for (size_t j = tid; j < row_size; j += blockDim.x) {
        scalar_t val = expf(row_in[j] - row_max);
        row_out[j] = val;
        local_sum += val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
        __syncthreads();
    }

    scalar_t row_sum = sdata[0];

    for (size_t j = tid; j < row_size; j += blockDim.x) {
        row_out[j] /= row_sum;
    }
}

void softmax_cuda(const scalar_t* d_in, scalar_t* d_out, size_t rows, size_t row_size) {
    size_t blockDim = 256;
    size_t gridDim = rows; // one block per row
    softmax_kernel<<<gridDim, blockDim>>>(d_in, d_out, rows, row_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void softmax_grad_kernel(const scalar_t* upstream, const scalar_t* output, scalar_t* grad, size_t rows, size_t row_size) {
    /*
        one block per row, same shape as forward:
        - dot_product = sum_j upstream[j] * output[j]  
        - grad[j] += output[j] * (upstream[j] - dot_product)
    */

    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    const scalar_t* row_upstream = upstream + row * row_size;
    const scalar_t* row_output = output + row * row_size;
    scalar_t* row_grad = grad + row * row_size;

    scalar_t local_sum = 0.0f;
    for (size_t j = tid; j < row_size; j += blockDim.x) {
        local_sum += row_upstream[j] * row_output[j];
    }

    __shared__ scalar_t sdata[256];
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    scalar_t dot_product = sdata[0];

    for (size_t j = tid; j < row_size; j += blockDim.x) {
        row_grad[j] += row_output[j] * (row_upstream[j] - dot_product);
    }
}

void softmax_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_output, scalar_t* d_grad, size_t rows, size_t row_size) {
    size_t blockDim = 256;
    size_t gridDim = rows;
    softmax_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_output, d_grad, rows, row_size);
    CUDA_CHECK(cudaGetLastError());
}