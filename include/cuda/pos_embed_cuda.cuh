#pragma once
#include "cuda/matmul_cuda.cuh"

void pos_embed_cuda(scalar_t* d_out, size_t seq_length, size_t embed_dim, size_t start_pos);
