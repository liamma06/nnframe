#include "cuda/mempool.cuh"
#include "cuda/matmul_cuda.cuh"


//since we just defualt stream, 0 for all malloc/free calls
void init_cuda_mempool(){
    cudaMemPool_t mempool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, 0));
    uint64_t threshold = UINT64_MAX; // for now idk if it best since it kinda maintins at the largest
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
}