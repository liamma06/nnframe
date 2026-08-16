#pragma once
#include "cuda/matmul_cuda.cuh" 

void causal_mask_cuda(const scalar_t* d_scores, scalar_t* d_out, size_t heads, size_t seq_len_q, size_t seq_len_k);
void causal_mask_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, size_t heads, size_t seq_len_q, size_t seq_len_k);