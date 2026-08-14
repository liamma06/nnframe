#pragma once
#include "cuda/matmul_cuda.cuh" 

void embed_cuda(const scalar_t* d_table, const scalar_t* d_indices, scalar_t* d_out, size_t seq_len, size_t embedding_dim);
void embed_backward_cuda( const scalar_t* d_indices, const scalar_t* d_grad_out, scalar_t* d_grad_table, size_t seq_len, size_t embedding_dim);