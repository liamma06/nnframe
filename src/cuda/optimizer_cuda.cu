#include "cuda/optimizer_cuda.cuh"

__global__ void adamw_kernel(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < param_size){
        scalar_t grad = d_grad[idx];
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


void adamw_cuda(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2) {
    dim3 blockDim(256);

    //atleast each param gets one thread 
    size_t gridDim = (param_size + blockDim.x - 1) / blockDim.x;
    adamw_kernel<<<gridDim, blockDim>>>(d_param, d_grad, d_m, d_v, param_size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); 

}