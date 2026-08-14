#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"
#include "loss/loss.h"
#include <cassert>
#include <stdexcept>

#ifdef NNFRAME_WITH_CUDA
#include "cuda/loss_cuda.cuh"
#endif

class CrossEntropy: public Loss{
    public:
        TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override{
            /*
                logits: [batch_size, num_classes]
                targets: [batch_size]
                output: scalar tensor

                Best for classifiction

                Note: important to remeber that target isn't the val it self but index of the token in the vocab.
            */

            if (logits->device() != targets->device()){
                throw std::runtime_error("logits and targets must be on the same device");
            }

            size_t seq_len = logits->shape()[0];
            size_t vocab_size = logits->shape()[1];

            #ifdef NNFRAME_WITH_CUDA
                if (logits->device() == Device::CUDA){
                    scalar_t* d_loss = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(scalar_t)));

                    cross_entropy_cuda(logits->device_data(), targets->device_data(), d_loss, seq_len, vocab_size);

                    return Tensor::from_device_ptr(d_loss, std::vector<size_t>{1}, std::vector<size_t>{1});
                }
            #endif

            auto softmax_probs = logits->softmax(1);

            
            auto correct_probs = Tensor::create({seq_len, 1});
            for (size_t i = 0; i < seq_len; i++){
                size_t target_class = static_cast<size_t>(targets->at({i}));
                assert(target_class < vocab_size && "Target class index out of bounds");
        
                correct_probs->at({i, 0}) = std::max(softmax_probs->at({i, target_class}), 1e-7f);
            }

            //autograd (BRUH)
            correct_probs->set_requires_grad(softmax_probs->requires_grad());
            correct_probs->set_inputs({softmax_probs});
            correct_probs->set_grad_fn([softmax_probs, targets, seq_len, vocab_size](const Tensor& upstream){
                if (softmax_probs->requires_grad()){
                    softmax_probs->init_grad();
                    for (size_t i = 0; i < seq_len; i++){
                        size_t target_class = static_cast<size_t>(targets->at({i}));
                        softmax_probs->grad().mutable_data()[i * vocab_size + target_class] += upstream.at({i, 0});
                    }
                }
            });

            
            return correct_probs->log()->mul(Tensor::create({1}, -1.0f))->mean();
        }
};