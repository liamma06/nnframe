#pragma once
#include <vector>
#include <cmath>
#include "core/tensor.h"

#ifdef NNFRAME_WITH_CUDA
    #include "cuda/optimizer_cuda.cuh"
#endif

/*
    important note: 
        - clip grad applies accross all parameters gradients
        - current implementation assumes all params either CPU or CUDA
*/

inline void clip_grad_norm(std::vector<TensorPtr> params, scalar_t max_norm) {
    #ifdef NNFRAME_WITH_CUDA
        std::vector<scalar_t*> d_grads;
        std::vector<size_t> d_sizes;
    #endif

    scalar_t sum_sq = 0.0f;

    for (auto& param : params) {
        if (!param->requires_grad()) continue;

        #ifdef NNFRAME_WITH_CUDA
            if (param->device() == Device::CUDA) {
                d_grads.push_back(param->grad().mutable_device_data());
                d_sizes.push_back(param->numel());
                continue;
            }
        #endif

        auto& grad_data = param->grad().data();
        for (size_t i = 0; i < grad_data.size(); i++) {
            sum_sq += grad_data[i] * grad_data[i];
        }
    }

    scalar_t norm = std::sqrt(sum_sq);
    if (norm > max_norm) {
        scalar_t scale = max_norm / norm;
        for (auto& param : params) {
            if (!param->requires_grad()) continue;

            #ifdef NNFRAME_WITH_CUDA
                if (param->device() == Device::CUDA) continue;
            #endif

            auto& grad_data = param->grad().mutable_data();
            for (size_t i = 0; i < grad_data.size(); i++) {
                grad_data[i] *= scale;
            }
        }
    }

    #ifdef NNFRAME_WITH_CUDA
        if (!d_grads.empty()) {
            clip_grad_norm_cuda(d_grads, d_sizes, max_norm);
        }
    #endif
}
