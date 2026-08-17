#pragma once
#include "cuda/matmul_cuda.cuh"

void add_bias_cuda(const scalar_t* d_input, const scalar_t* d_bias, scalar_t* d_out, size_t batch_size, size_t features);
void add_bias_grad_cuda(const scalar_t* d_upstream, scalar_t* d_input_grad, scalar_t* d_bias_grad, size_t batch_size, size_t features);

//reasoning:  froward flow: matmul -> bias -> gelu (bias+gelu)
void bias_gelu_cuda(const scalar_t* d_input, const scalar_t* d_bias, scalar_t* d_out, size_t batch_size, size_t features);
void bias_gelu_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_input, const scalar_t* d_bias, scalar_t* d_input_grad, scalar_t* d_bias_grad, size_t batch_size, size_t features);
