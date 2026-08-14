#pragma once
#include "cuda/softmax_cuda.cuh"
#include "cuda/elementwise_cuda.cuh"

void cross_entropy_cuda(const scalar_t* d_logits, const scalar_t* d_targets, scalar_t* d_loss, size_t rows, size_t vocab_size);
void cross_entropy_backward_cuda(const scalar_t* d_logits, const scalar_t* d_targets, scalar_t* d_grad_logits, size_t rows, size_t vocab_size, scalar_t upstream);
