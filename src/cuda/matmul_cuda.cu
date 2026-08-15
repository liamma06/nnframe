#include "cuda/matmul_cuda.cuh"
#include "cuda/elementwise_cuda.cuh" 


__global__ void matmul_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N) {
    /*
        each thread -> one element of output matrix (threadIdx)
        each block -> 16x16 threads (blockIdx)
        
        helpful:
        - blockIdx.{x,y}  which block the thread is in 
        - blockDim.{x,y}  how many thread per block in that dimension (16) 
        - threadIdx.{x,y}  which thread in the block (0-15)
        so by blockIdx.x * blockDim.x -> gives us an offset for the thread
    */

    //element thread responsible for 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N){
        scalar_t sum = 0;
        for (int i = 0; i < K; i++){
            sum += a[row * K + i ] * b[i * N + col];
        }
        out[row * N + col] = sum; 
    }
}

void matmul_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N) {
    /*
        helper function to launch kernel
        blockdim -> 16 x 16 threads per 

        OUPUT is a M x N matrix which run in the grid:
        Grid -> (M + 15)/16, (N + 15)/16 blocks per grid 
        +15 because we want round up to hold any remaining dim 

        TLDR:
        - block is 16x16 of the output 
        - grid is enough of those blocks to cover the entire output matrix
    */
    
    dim3 blockDim(16,16); 
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_out, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    
}

__global__ void matmul_batched_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N, size_t batch_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (row < M && col < N && batch < batch_size){
        scalar_t sum = 0;
        for (int i = 0; i < K; i++){
            sum += a[batch * M * K + row * K + i ] * b[batch * K * N + i * N + col];
        }
        out[batch * M * N + row * N + col] = sum;
    }

}

void matmul_batched(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N, size_t batch_size) {
    /*
        launch all batches in one kernel
    */

    dim3 blockDim(16,16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16, batch_size);
    matmul_batched_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_out, M, K, N, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void transpose_kernel(const scalar_t* in, scalar_t* out, size_t rows, size_t cols) {
    // in is [rows, cols], out is [cols, rows], one thread per element
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;

    if (idx < total) {
        size_t i = idx / cols; 
        size_t j = idx % cols; 
        out[j * rows + i] = in[idx];
    }
}

void transpose_cuda(const scalar_t* d_in, scalar_t* d_out, size_t rows, size_t cols) {
    size_t total = rows * cols;
    size_t blockDim = 256;
    size_t gridDim = (total + blockDim - 1) / blockDim;
    transpose_kernel<<<gridDim, blockDim>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void transpose_batched_kernel(const scalar_t* in, scalar_t* out, size_t rows, size_t cols, size_t batch_size) {
    // each batch's [rows, cols] slice is transposed separate
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t per_batch = rows * cols;
    size_t total = per_batch * batch_size;

    if (idx < total) {
        size_t batch = idx / per_batch;
        size_t local = idx % per_batch;
        size_t i = local / cols;
        size_t j = local % cols;
        out[batch * per_batch + j * rows + i] = in[idx];
    }
}

void transpose_batched_cuda(const scalar_t* d_in, scalar_t* d_out, size_t rows, size_t cols, size_t batch_size) {
    size_t total = rows * cols * batch_size;
    size_t blockDim = 256;
    size_t gridDim = (total + blockDim - 1) / blockDim;
    transpose_batched_kernel<<<gridDim, blockDim>>>(d_in, d_out, rows, cols, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_grad_cuda(const scalar_t* d_upstream, const scalar_t* d_self, const scalar_t* d_other, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t M, size_t K, size_t N) {
    /*
        - d_upstream is [M, N], d_self is [M, K], d_other is [K, N]
        - d_self_grad += d_upstream @ d_other^T
        - d_other_grad += d_self^T @ d_upstream
        - matmul_cuda overwrites its output -> product goes into a temp -> accumlate to grad
    */
    scalar_t* d_other_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_other_T, N * K * sizeof(scalar_t)));
    transpose_cuda(d_other, d_other_T, K, N);

    scalar_t* temp1 = nullptr;
    CUDA_CHECK(cudaMalloc(&temp1, M * K * sizeof(scalar_t)));
    matmul_cuda(d_upstream, d_other_T, temp1, M, N, K);
    accumulate_cuda(d_self_grad, temp1, M * K);

    CUDA_CHECK(cudaFree(d_other_T));
    CUDA_CHECK(cudaFree(temp1));

    scalar_t* d_self_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_self_T, K * M * sizeof(scalar_t)));
    transpose_cuda(d_self, d_self_T, M, K);

    scalar_t* temp2 = nullptr;
    CUDA_CHECK(cudaMalloc(&temp2, K * N * sizeof(scalar_t)));
    matmul_cuda(d_self_T, d_upstream, temp2, K, M, N);
    accumulate_cuda(d_other_grad, temp2, K * N);

    CUDA_CHECK(cudaFree(d_self_T));
    CUDA_CHECK(cudaFree(temp2));
}

void matmul_grad_batched_cuda(const scalar_t* d_upstream, const scalar_t* d_self, const scalar_t* d_other, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t M, size_t K, size_t N, size_t batch_size) {
    scalar_t* d_other_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_other_T, batch_size * N * K * sizeof(scalar_t)));
    transpose_batched_cuda(d_other, d_other_T, K, N, batch_size);

    scalar_t* temp1 = nullptr;
    CUDA_CHECK(cudaMalloc(&temp1, batch_size * M * K * sizeof(scalar_t)));
    matmul_batched(d_upstream, d_other_T, temp1, M, N, K, batch_size);
    accumulate_cuda(d_self_grad, temp1, batch_size * M * K);

    CUDA_CHECK(cudaFree(d_other_T));
    CUDA_CHECK(cudaFree(temp1));

    scalar_t* d_self_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_self_T, batch_size * K * M * sizeof(scalar_t)));
    transpose_batched_cuda(d_self, d_self_T, M, K, batch_size);

    scalar_t* temp2 = nullptr;
    CUDA_CHECK(cudaMalloc(&temp2, batch_size * K * N * sizeof(scalar_t)));
    matmul_batched(d_self_T, d_upstream, temp2, K, M, N, batch_size);
    accumulate_cuda(d_other_grad, temp2, batch_size * K * N);

    CUDA_CHECK(cudaFree(d_self_T));
    CUDA_CHECK(cudaFree(temp2));
}