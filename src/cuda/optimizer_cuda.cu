#include "cuda/optimizer_cuda.cuh"
#include <vector>

__global__ void adamw_kernel(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2, scalar_t inv_scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < param_size){
        scalar_t grad = d_grad[idx] * inv_scale;
        scalar_t m = d_m[idx];
        scalar_t v = d_v[idx];

        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * grad * grad;

        d_m[idx] = m;
        d_v[idx] = v;

        scalar_t m_hat = m / bias_correction1;
        scalar_t v_hat = v / bias_correction2;

        d_param[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * d_param[idx]);
    }
}


void adamw_cuda(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2, scalar_t inv_scale) {
    dim3 blockDim(256);

    //atleast each param gets one thread
    size_t gridDim = (param_size + blockDim.x - 1) / blockDim.x;
    adamw_kernel<<<gridDim, blockDim>>>(d_param, d_grad, d_m, d_v, param_size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, inv_scale);
    CUDA_CHECK(cudaGetLastError());

}


//fusion: grad clip -> adanw update 
__global__ void adamw_clipped_kernel(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2, const scalar_t* d_sum_sq, scalar_t max_norm) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < param_size){
        scalar_t norm = sqrtf(*d_sum_sq);
        scalar_t scale = (norm > max_norm) ? (max_norm / norm) : 1.0f;
        scalar_t grad = d_grad[idx] * scale; // read from d_grad only once

        scalar_t m = d_m[idx];
        scalar_t v = d_v[idx];

        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * grad * grad;

        d_m[idx] = m;
        d_v[idx] = v;

        scalar_t m_hat = m / bias_correction1;
        scalar_t v_hat = v / bias_correction2;

        d_param[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * d_param[idx]);
    }
}

void adamw_clipped_cuda(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2, const scalar_t* d_sum_sq, scalar_t max_norm) {
    dim3 blockDim(256);
    size_t gridDim = (param_size + blockDim.x - 1) / blockDim.x;
    adamw_clipped_kernel<<<gridDim, blockDim>>>(d_param, d_grad, d_m, d_v, param_size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, d_sum_sq, max_norm);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sum_of_squares_kernel(const scalar_t* d_input, scalar_t* d_output, size_t input_size) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    __shared__ scalar_t sdata[256];

    scalar_t g = (idx < input_size) ? d_input[idx] : 0.0f;
    sdata[tid] = g * g;     
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) { //half threads each iteration
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_output, sdata[0]);
    }
}

void sum_of_squares_cuda(const scalar_t* d_input, scalar_t* d_output, size_t input_size) {
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(scalar_t)));

    dim3 blockDim(256);
    size_t gridDim = (input_size + blockDim.x - 1) / blockDim.x;
    sum_of_squares_kernel<<<gridDim, blockDim>>>(d_input, d_output, input_size);
    CUDA_CHECK(cudaGetLastError());
}

// same as sum_of_squares_cuda but doesn't zero d_output first
void sum_of_squares_accumulate_cuda(const scalar_t* d_input, scalar_t* d_output, size_t input_size) {
    dim3 blockDim(256);
    size_t gridDim = (input_size + blockDim.x - 1) / blockDim.x;
    sum_of_squares_kernel<<<gridDim, blockDim>>>(d_input, d_output, input_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void grad_clip_kernel(scalar_t* d_grad, const scalar_t* d_sum_sq, size_t n, scalar_t max_norm) {
    /*
        goal:
        - compute norm -> "how big whole gradinet"
        -if norm > max norm, scale down each element 
        - why? gradients can explode -> bad     
    */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t norm = sqrtf(*d_sum_sq);

    if (idx < n && norm > max_norm) {
        scalar_t scale = max_norm / norm;
        d_grad[idx] *= scale;
    }
}

void grad_clip_cuda(scalar_t* d_grad, const scalar_t* d_sum_sq, size_t n, scalar_t max_norm) {
    dim3 blockDim(256);
    size_t gridDim = (n + blockDim.x - 1) / blockDim.x;
    grad_clip_kernel<<<gridDim, blockDim>>>(d_grad, d_sum_sq, n, max_norm);
    CUDA_CHECK(cudaGetLastError());
}

void clip_grad_norm_cuda(std::vector<scalar_t*> d_grads, std::vector<size_t> sizes, scalar_t max_norm) {
    scalar_t* d_sum_sq = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_sum_sq, sizeof(scalar_t), 0));
    CUDA_CHECK(cudaMemset(d_sum_sq, 0, sizeof(scalar_t))); 

    dim3 blockDim(256);

    for (size_t i = 0; i < d_grads.size(); i++) {
        size_t gridDim = (sizes[i] + blockDim.x - 1) / blockDim.x;
        sum_of_squares_kernel<<<gridDim, blockDim>>>(d_grads[i], d_sum_sq, sizes[i]);
    }
    CUDA_CHECK(cudaGetLastError());

    for (size_t i = 0; i < d_grads.size(); i++) {
        size_t gridDim = (sizes[i] + blockDim.x - 1) / blockDim.x;
        grad_clip_kernel<<<gridDim, blockDim>>>(d_grads[i], d_sum_sq, sizes[i], max_norm);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_sum_sq, 0));
}