#pragma once
#include "cuda/matmul_cuda.cuh"


void quantize_cuda(const scalar_t* d_in, int8_t* d_out, float* d_scales, size_t num_slots, size_t head_dim);

void dequantize_cuda(const int8_t* d_in, scalar_t* d_out, const float* d_scales, size_t num_slots, size_t head_dim);
