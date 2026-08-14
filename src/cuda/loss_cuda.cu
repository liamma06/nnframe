#include "cuda/loss_cuda.cuh"

/*
    reuses softmax_cuda 
    forward gathers the target-class probability per row and takes -log of it
*/

__global__ void nll_gather_kernel(const scalar_t* probs, const scalar_t* targets, scalar_t* row_loss, size_t rows, size_t vocab_size) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        size_t target = (size_t)targets[row];
        scalar_t p = fmaxf(probs[row * vocab_size + target], 1e-7f);
        row_loss[row] = -logf(p);
    }
}

void cross_entropy_cuda(const scalar_t* d_logits, const scalar_t* d_targets, scalar_t* d_loss, size_t rows, size_t vocab_size) {
    scalar_t* d_probs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_probs, rows * vocab_size * sizeof(scalar_t)));
    softmax_cuda(d_logits, d_probs, rows, vocab_size);

    scalar_t* d_row_loss = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_loss, rows * sizeof(scalar_t)));

    size_t blockDim = 256;
    size_t gridDim = (rows + blockDim - 1) / blockDim;
    nll_gather_kernel<<<gridDim, blockDim>>>(d_probs, d_targets, d_row_loss, rows, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    sum_reduce_cuda(d_row_loss, d_loss, rows);
    scale_scalar_cuda(d_loss, 1.0f / static_cast<scalar_t>(rows));

    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_row_loss));
}

__global__ void cross_entropy_backward_kernel(const scalar_t* probs, const scalar_t* targets, scalar_t* grad_logits, size_t rows, size_t vocab_size, scalar_t upstream) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * vocab_size;

    if (idx < total) {
        size_t row = idx / vocab_size;
        size_t cls = idx % vocab_size;
        size_t target = (size_t)targets[row];

        scalar_t one_hot = (cls == target) ? 1.0f : 0.0f;
        grad_logits[idx] = (probs[idx] - one_hot) * upstream / static_cast<scalar_t>(rows);
    }
}

void cross_entropy_backward_cuda(const scalar_t* d_logits, const scalar_t* d_targets, scalar_t* d_grad_logits, size_t rows, size_t vocab_size, scalar_t upstream) {
    scalar_t* d_probs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_probs, rows * vocab_size * sizeof(scalar_t)));
    softmax_cuda(d_logits, d_probs, rows, vocab_size);

    size_t total = rows * vocab_size;
    size_t blockDim = 256;
    size_t gridDim = (total + blockDim - 1) / blockDim;
    cross_entropy_backward_kernel<<<gridDim, blockDim>>>(d_probs, d_targets, d_grad_logits, rows, vocab_size, upstream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_probs));
}
