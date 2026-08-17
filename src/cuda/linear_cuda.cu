#include "cuda/linear_cuda.cuh"

// TODO: add_bias_kernel / add_bias_cuda
// TODO: add_bias_grad_kernel / add_bias_grad_cuda

//fusion
__global__ void bias_gelu_kernel(const scalar_t* input, const scalar_t* bias, scalar_t* out, size_t batch_size, size_t features) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < batch_size * features) {
            size_t feature_idx = i % features; //since bias in 1D need to "loop back"
            scalar_t x = input[i] + bias[feature_idx];
            scalar_t sig = 1.0f / (1.0f + expf(-1.702f * x));
            out[i] = x * sig;
        }
    }

void bias_gelu_cuda(const scalar_t* d_input, const scalar_t* d_bias, scalar_t* d_out, size_t batch_size, size_t features){
    size_t blockDim = 256;
    size_t gridDim = (batch_size * features + blockDim - 1) / blockDim;
    bias_gelu_kernel<<<gridDim, blockDim>>>(d_input, d_bias, d_out, batch_size, features);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void bias_gelu_grad_kernel(const scalar_t* upstream, const scalar_t* input, const scalar_t* bias, scalar_t* input_grad, scalar_t* bias_grad, size_t batch_size, size_t features) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size * features) {
        size_t feature_idx = i % features;
        scalar_t x = input[i] + bias[feature_idx]; // recompute pre-activation sinc never stored from forward
        scalar_t sig = 1.0f / (1.0f + expf(-1.702f * x));
        scalar_t grad = sig + x * sig * (1.0f - sig) * 1.702f;

        input_grad[i] += upstream[i] * grad;
        atomicAdd(&bias_grad[feature_idx], upstream[i] * grad); // rows shared -> atomic add
    }
}

void bias_gelu_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_input, const scalar_t* d_bias, scalar_t* d_input_grad, scalar_t* d_bias_grad, size_t batch_size, size_t features) {
    size_t blockDim = 256;
    size_t gridDim = (batch_size * features + blockDim - 1) / blockDim;
    bias_gelu_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_input, d_bias, d_input_grad, d_bias_grad, batch_size, features);
    CUDA_CHECK(cudaGetLastError());
}
