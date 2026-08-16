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

__global__ void matmul_batched_accumulate_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N, size_t batch_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (row < M && col < N && batch < batch_size){
        scalar_t sum = 0;
        for (int i = 0; i < K; i++){
            sum += a[batch * M * K + row * K + i ] * b[batch * K * N + i * N + col];
        }
        out[batch * M * N + row * N + col] += sum;
    }
}

void matmul_batched_accumulate_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N, size_t batch_size) {
    dim3 blockDim(16,16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16, batch_size);
    matmul_batched_accumulate_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_out, M, K, N, batch_size);
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

/*
    contiguous(): rank<=3 strided gather (forward) / scatter (backward).
*/
__global__ void contiguous_kernel(const scalar_t* in, scalar_t* out,
                                    size_t shape0, size_t shape1, size_t shape2,
                                    size_t stride0, size_t stride1, size_t stride2,
                                    size_t offset, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t remaining = i;
        size_t idx2 = remaining % shape2; remaining /= shape2;
        size_t idx1 = remaining % shape1; remaining /= shape1;
        size_t idx0 = remaining;

        size_t pos = offset + idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
        out[i] = in[pos];
    }
}

void contiguous_cuda(const scalar_t* d_in, scalar_t* d_out, size_t shape0, size_t shape1, size_t shape2, size_t stride0, size_t stride1, size_t stride2, size_t offset, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    contiguous_kernel<<<gridDim, blockDim>>>(d_in, d_out, shape0, shape1, shape2, stride0, stride1, stride2, offset, n);
    CUDA_CHECK(cudaGetLastError());
}

/*
    backward for permute(axes): self_grad[i] += upstream[out_idx]
*/
__global__ void permute_grad_kernel(const scalar_t* upstream, scalar_t* self_grad,
                                     size_t shape0, size_t shape1, size_t shape2,
                                     size_t axis0, size_t axis1, size_t axis2,
                                     size_t up_stride0, size_t up_stride1, size_t up_stride2,
                                     size_t up_offset, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t remaining = i;
        size_t self_idx2 = remaining % shape2; remaining /= shape2;
        size_t self_idx1 = remaining % shape1; remaining /= shape1;
        size_t self_idx0 = remaining;
        size_t self_idx[3] = {self_idx0, self_idx1, self_idx2};

        size_t axes_arr[3] = {axis0, axis1, axis2};
        size_t up_strides[3] = {up_stride0, up_stride1, up_stride2};

        size_t pos = up_offset;
        for (size_t j = 0; j < 3; j++) pos += self_idx[axes_arr[j]] * up_strides[j];

        self_grad[i] += upstream[pos];
    }
}

void permute_grad_cuda(const scalar_t* d_upstream, scalar_t* d_self_grad, size_t shape0, size_t shape1, size_t shape2, size_t axis0, size_t axis1, size_t axis2, size_t up_stride0, size_t up_stride1, size_t up_stride2, size_t up_offset, size_t n) {
    size_t blockDim = 256;
    size_t gridDim = (n + blockDim - 1) / blockDim;
    permute_grad_kernel<<<gridDim, blockDim>>>(d_upstream, d_self_grad, shape0, shape1, shape2, axis0, axis1, axis2, up_stride0, up_stride1, up_stride2, up_offset, n);
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
        - matmul_accumulate_cuda writes the product straight into the grad buffer with +=, no temp needed
    */
    scalar_t* d_other_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_other_T, N * K * sizeof(scalar_t)));
    transpose_cuda(d_other, d_other_T, K, N);

    matmul_accumulate_cuda(d_upstream, d_other_T, d_self_grad, M, N, K);

    CUDA_CHECK(cudaFree(d_other_T));

    scalar_t* d_self_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_self_T, K * M * sizeof(scalar_t)));
    transpose_cuda(d_self, d_self_T, M, K);

    matmul_accumulate_cuda(d_self_T, d_upstream, d_other_grad, K, M, N);

    CUDA_CHECK(cudaFree(d_self_T));
}

void matmul_grad_batched_cuda(const scalar_t* d_upstream, const scalar_t* d_self, const scalar_t* d_other, scalar_t* d_self_grad, scalar_t* d_other_grad, size_t M, size_t K, size_t N, size_t batch_size) {
    scalar_t* d_other_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_other_T, batch_size * N * K * sizeof(scalar_t)));
    transpose_batched_cuda(d_other, d_other_T, K, N, batch_size);

    matmul_batched_accumulate_cuda(d_upstream, d_other_T, d_self_grad, M, N, K, batch_size);

    CUDA_CHECK(cudaFree(d_other_T));

    scalar_t* d_self_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_self_T, batch_size * K * M * sizeof(scalar_t)));
    transpose_batched_cuda(d_self, d_self_T, M, K, batch_size);

    matmul_batched_accumulate_cuda(d_self_T, d_upstream, d_other_grad, K, M, N, batch_size);

    CUDA_CHECK(cudaFree(d_self_T));
}


//fused accumlate kernel for backward pass of matmul 
__global__ void matmul_accumulate_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N){
        scalar_t sum = 0;
        for (int i = 0; i < K; i++){
            sum += a[row * K + i ] * b[i * N + col];
        }
        out[row * N + col] += sum; 
    }
}

void matmul_accumulate_cuda(const scalar_t* d_a, const scalar_t* d_b, scalar_t* d_out, size_t M, size_t K, size_t N) {
    dim3 blockDim(16,16); 
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    matmul_accumulate_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_out, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}