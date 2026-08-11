#include "cuda/matmul_cuda.cuh"


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
    CUDA_CHECK(cudaDeviceSynchronize());
    
}