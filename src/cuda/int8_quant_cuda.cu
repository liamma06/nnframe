#include "cuda/int8_quant_cuda.cuh"

__global__ void quantize_kernel(const scalar_t* d_in, int8_t* d_out, float* d_scales, size_t num_slots, size_t head_dim) {
    // one thread per slot
    size_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= num_slots) return;

    const scalar_t* row_in = d_in + slot * head_dim;
    int8_t* row_out = d_out + slot * head_dim;

    float max_abs = 0.0f;
    for (size_t i = 0; i < head_dim; i++) {
        float v = fabsf(row_in[i]);
        if (v > max_abs) max_abs = v;
    }

    float scale = max_abs / 127.0f;
    if (scale == 0.0f) {
        for (size_t i = 0; i < head_dim; i++) row_out[i] = 0;
        d_scales[slot] = 1.0f;
        return;
    }

    for (size_t i = 0; i < head_dim; i++) {
        float rounded = roundf(row_in[i] / scale);
        if (rounded > 127.0f) rounded = 127.0f;
        if (rounded < -127.0f) rounded = -127.0f;
        row_out[i] = static_cast<int8_t>(rounded);
    }

    d_scales[slot] = scale;
}

void quantize_cuda(const scalar_t* d_in, int8_t* d_out, float* d_scales, size_t num_slots, size_t head_dim) {
    dim3 block(256);
    dim3 grid((num_slots + block.x - 1) / block.x);
    quantize_kernel<<<grid, block>>>(d_in, d_out, d_scales, num_slots, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void dequantize_kernel(const int8_t* d_in, scalar_t* d_out, const float* d_scales, size_t num_slots, size_t head_dim) {
    size_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= num_slots) return;

    const int8_t* row_in = d_in + slot * head_dim;
    scalar_t* row_out = d_out + slot * head_dim;
    float scale = d_scales[slot];

    for (size_t i = 0; i < head_dim; i++) {
        row_out[i] = static_cast<scalar_t>(row_in[i]) * scale;
    }
}

void dequantize_cuda(const int8_t* d_in, scalar_t* d_out, const float* d_scales, size_t num_slots, size_t head_dim) {
    dim3 block(256);
    dim3 grid((num_slots + block.x - 1) / block.x);
    dequantize_kernel<<<grid, block>>>(d_in, d_out, d_scales, num_slots, head_dim);
    CUDA_CHECK(cudaGetLastError());
}
