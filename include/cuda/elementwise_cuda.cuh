#pragma once
#include "cuda/matmul_cuda.cuh" 

void add_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n);
void sub_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n);
void mul_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t n);
