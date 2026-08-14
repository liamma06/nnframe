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

__global__ void relu_kernel(const scalar_t* in, scalar_t* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void relu_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    relu_kernel<<<gridDim, blockDim>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void gelu_kernel(const scalar_t* in, scalar_t* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        scalar_t x = in[i];
        scalar_t sig = 1.0f / (1.0f + expf(-1.702f * x));
        out[i] = x * sig;
    }
}

void gelu_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    gelu_kernel<<<gridDim, blockDim>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/*
    backward kernels -> same one-thread-per-element shape as forward
*/

__global__ void add_grad_kernel(const scalar_t* upstream, scalar_t* self_grad, scalar_t* other_grad, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        self_grad[i] += upstream[i];
        other_grad[i] += upstream[i];
    }
}

void add_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    add_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_grad, d_other_grad, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void sub_grad_kernel(const scalar_t* upstream, scalar_t* self_grad, scalar_t* other_grad, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        self_grad[i] += upstream[i];
        other_grad[i] -= upstream[i];
    }
}

void sub_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    sub_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_grad, d_other_grad, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void mul_grad_kernel(const scalar_t* upstream, const scalar_t* self_data, const scalar_t* other_data, scalar_t* self_grad, scalar_t* other_grad, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        self_grad[i] += upstream[i] * other_data[i];
        other_grad[i] += upstream[i] * self_data[i];
    }
}

void mul_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, const scalar_t* d_other_data, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    mul_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_data, d_other_data, d_self_grad, d_other_grad, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void log_grad_kernel(const scalar_t* upstream, const scalar_t* self_data, scalar_t* self_grad, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) self_grad[i] += upstream[i] / self_data[i];
}

void log_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, scalar_t* d_self_grad, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    log_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_data, d_self_grad, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void relu_grad_kernel(const scalar_t* upstream, const scalar_t* self_data, scalar_t* self_grad, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && self_data[i] > 0.0f) self_grad[i] += upstream[i];
}

void relu_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, scalar_t* d_self_grad, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    relu_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_data, d_self_grad, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void gelu_grad_kernel(const scalar_t* upstream, const scalar_t* self_data, scalar_t* self_grad, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        scalar_t x = self_data[i];
        scalar_t sig = 1.0f / (1.0f + expf(-1.702f * x));
        scalar_t grad = sig + x * sig * (1.0f - sig) * 1.702f;
        self_grad[i] += upstream[i] * grad;
    }
}

void gelu_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, scalar_t* d_self_grad, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    gelu_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_data, d_self_grad, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void mean_grad_kernel(const scalar_t* upstream, scalar_t* self_grad, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) self_grad[i] += (*upstream) / static_cast<scalar_t>(n);
}

void mean_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    mean_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_grad, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}