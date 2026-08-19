#pragma once
#include "cuda/matmul_cuda.cuh"
#include <vector>

void adamw_cuda(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2, scalar_t inv_scale = 1.0f); // inv_scale unscales loss-scaled gradients (1/loss_scale) before the update
void adamw_clipped_cuda(scalar_t* d_param, const scalar_t* d_grad, scalar_t* d_m, scalar_t* d_v, size_t param_size, scalar_t lr, scalar_t beta1, scalar_t beta2, scalar_t eps, scalar_t weight_decay, scalar_t bias_correction1, scalar_t bias_correction2, const scalar_t* d_sum_sq, scalar_t max_norm);

void sum_of_squares_cuda(const scalar_t* d_input, scalar_t* d_output, size_t input_size);
void sum_of_squares_accumulate_cuda(const scalar_t* d_input, scalar_t* d_output, size_t input_size);

void grad_clip_cuda(scalar_t* d_grad, const scalar_t* d_sum_sq, size_t n, scalar_t max_norm);

void clip_grad_norm_cuda(std::vector<scalar_t*> d_grads, std::vector<size_t> sizes, scalar_t max_norm);