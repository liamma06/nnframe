#pragma once 
#include <cuda_runtime.h>
#include "core/tensor.h" 

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void matmul_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N);

void matmul_batched(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N, size_t batch_size);

void transpose_cuda(const scalar_t* d_in, scalar_t* d_out, size_t rows, size_t cols);
void transpose_batched_cuda(const scalar_t* d_in, scalar_t* d_out, size_t rows, size_t cols, size_t batch_size);

void matmul_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self, const scalar_t* d_other, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t M, size_t K, size_t N);
void matmul_grad_batched_cuda(const scalar_t* d_upstream, const scalar_t* d_self, const scalar_t* d_other, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t M, size_t K, size_t N, size_t batch_size);

void matmul_accumulate_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N);
void matmul_batched_accumulate_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N, size_t batch_size);