#pragma once
#include "cuda/matmul_cuda.cuh" 

void add_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n);
void sub_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n);
void mul_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n);
void log_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n);

void sum_reduce_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n);
void scale_scalar_cuda(scalar_t* d_val, scalar_t factor);

void relu_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n);
void gelu_cuda(const scalar_t* d_in, scalar_t* d_out, size_t n);

void scale_tensor_cuda(const scalar_t* d_in, scalar_t* d_out, scalar_t factor, size_t n);
void scale_tensor_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, scalar_t factor, size_t n);

void add_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t n);
void sub_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t n);
void mul_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, const scalar_t* d_other_data, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t n);
void log_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, scalar_t* d_self_grad, size_t n);
void relu_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, scalar_t* d_self_grad, size_t n);
void gelu_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self_data, scalar_t* d_self_grad, size_t n);
void mean_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, size_t n);

void accumulate_cuda(scalar_t* d_dst, const scalar_t* d_src, size_t n);

//reasoning:  froward flow: matmul -> bias -> gelu (bias+gelu)
void bias_gelu_cuda(const scalar_t* d_input, const scalar_t* d_bias, scalar_t* d_out, size_t batch_size, size_t features);
void bias_gelu_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_input, const scalar_t* d_bias, scalar_t* d_input_grad, scalar_t* d_bias_grad, size_t batch_size, size_t features);
