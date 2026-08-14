#pragma once
#include "cuda/matmul_cuda.cuh" 

void layernorm_cuda(const scalar_t* d_input, const scalar_t* d_gamma, const scalar_t* d_beta, scalar_t* d_output, size_t rows, size_t cols, scalar_t eps);
void layernorm_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_input, const scalar_t* d_gamma,
                         scalar_t* d_input_grad, scalar_t* d_gamma_grad, scalar_t* d_beta_grad,
                         size_t rows, size_t cols, scalar_t eps);