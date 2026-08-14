#pragma once
#include "cuda/matmul_cuda.cuh" 

void softmax_cuda(const scalar_t* d_in, scalar_t* d_out, size_t rows, size_t row_size);