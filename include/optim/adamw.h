#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"

#ifdef NNFRAME_WITH_CUDA
    #include "cuda/optimizer_cuda.cuh"
#endif
#include "optim/grad_clip.h"

//bru hthis so confusing 

class AdamW{
    private:
        std::vector<TensorPtr> params_;
        float lr_;
        float beta1_;
        float beta2_;
        float epsilon_;
        float weight_decay_;

        std::vector<std::vector<scalar_t>> m_; //first moment
        std::vector<std::vector<scalar_t>> v_; //second moment

        //CUDA
        std::vector<scalar_t*> d_m_;
        std::vector<scalar_t*> d_v_;

        #ifdef NNFRAME_WITH_CUDA
            scalar_t* d_sum_sq_ = nullptr; 
        #endif

        size_t t_; //time step
    
    public:
        AdamW(std::vector<TensorPtr> params, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.01f)
            : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {
            m_.resize(params.size());
            v_.resize(params.size());
            d_m_.resize(params.size());
            d_v_.resize(params.size());

            for (size_t i = 0; i < params.size(); ++i) {
                if(params[i]->device() == Device::CPU){
                    m_[i].resize(params[i]->numel(), 0.0f);
                    v_[i].resize(params[i]->numel(), 0.0f);
                }

                #ifdef NNFRAME_WITH_CUDA
                    else if(params[i]->device() == Device::CUDA){
                        size_t n = params[i]->numel();
                        CUDA_CHECK(cudaMallocAsync(&d_m_[i], n * sizeof(scalar_t), 0));
                        CUDA_CHECK(cudaMallocAsync(&d_v_[i], n * sizeof(scalar_t), 0));
                        CUDA_CHECK(cudaMemset(d_m_[i], 0, n * sizeof(scalar_t)));
                        CUDA_CHECK(cudaMemset(d_v_[i], 0, n * sizeof(scalar_t)));
                    }
                #endif


            }

            #ifdef NNFRAME_WITH_CUDA
                for (size_t i = 0; i < params.size(); ++i) {
                    if (params[i]->device() == Device::CUDA) {
                        CUDA_CHECK(cudaMallocAsync(&d_sum_sq_, sizeof(scalar_t), 0));
                        break;
                    }
                }
            #endif
        }

        ~AdamW(){
            #ifdef NNFRAME_WITH_CUDA
                for (size_t i = 0; i < params_.size(); ++i) {
                    if(params_[i]->device() == Device::CUDA){
                        CUDA_CHECK(cudaFreeAsync(d_m_[i], 0));
                        CUDA_CHECK(cudaFreeAsync(d_v_[i], 0));
                    }
                }
                if (d_sum_sq_) CUDA_CHECK(cudaFreeAsync(d_sum_sq_, 0));
            #endif
        }

        void step() {
            t_++;
            scalar_t bias_correction1 = 1.0f - std::pow(beta1_, static_cast<scalar_t>(t_));
            scalar_t bias_correction2 = 1.0f - std::pow(beta2_, static_cast<scalar_t>(t_));

            for (size_t i = 0; i < params_.size(); ++i) {
                if (!params_[i]->requires_grad()) continue;

                #ifdef NNFRAME_WITH_CUDA
                    if (params_[i]->device() == Device::CUDA) {
                        adamw_cuda(params_[i]->mutable_device_data(), params_[i]->grad().device_data(),
                                    d_m_[i], d_v_[i], params_[i]->numel(),
                                   lr_, beta1_, beta2_, epsilon_, weight_decay_, bias_correction1, bias_correction2);
                        continue;
                    }
                #endif

                auto& param_data = params_[i]->mutable_data();
                auto& param_grad = params_[i]->grad().mutable_data();
                for (size_t j = 0; j < param_data.size(); ++j) {
                    // Update biased first moment estimate
                    m_[i][j] = beta1_ * m_[i][j] + (1 - beta1_) * param_grad[j];
                    // Update biased second first moment estimate
                    v_[i][j] = beta2_ * v_[i][j] + (1 - beta2_) * (param_grad[j] * param_grad[j]);
                    // Compute bias correct first moment estimate
                    scalar_t m_hat = m_[i][j] / bias_correction1;
                    // Compute bias correct second first moment estimate
                    scalar_t v_hat = v_[i][j] / bias_correction2;
                    // Update parameters ->  decay
                    param_data[j] -= lr_ * (m_hat / (std::sqrt(v_hat) + epsilon_) + weight_decay_ * param_data[j]);
                }
            }
        }

        // grad-clip fused: AdamW reads the raw grad once and applies the clip-scale
        void step(scalar_t max_norm) {
            t_++;
            scalar_t bias_correction1 = 1.0f - std::pow(beta1_, static_cast<scalar_t>(t_));
            scalar_t bias_correction2 = 1.0f - std::pow(beta2_, static_cast<scalar_t>(t_));

            #ifdef NNFRAME_WITH_CUDA
                if (d_sum_sq_) {
                    CUDA_CHECK(cudaMemset(d_sum_sq_, 0, sizeof(scalar_t)));

                    for (auto& param : params_) {
                        if (param->requires_grad() && param->device() == Device::CUDA) {
                            sum_of_squares_accumulate_cuda(param->grad().device_data(), d_sum_sq_, param->numel());
                        }
                    }
                }
            #endif

            // CPU params: clip -> then run the normal update 
            std::vector<TensorPtr> cpu_params;
            for (auto& param : params_) {
                if (param->requires_grad() && param->device() == Device::CPU) cpu_params.push_back(param);
            }
            if (!cpu_params.empty()) clip_grad_norm(cpu_params, max_norm);

            for (size_t i = 0; i < params_.size(); ++i) {
                if (!params_[i]->requires_grad()) continue;

                #ifdef NNFRAME_WITH_CUDA
                    if (params_[i]->device() == Device::CUDA) {
                        adamw_clipped_cuda(params_[i]->mutable_device_data(), params_[i]->grad().device_data(),
                                    d_m_[i], d_v_[i], params_[i]->numel(),
                                   lr_, beta1_, beta2_, epsilon_, weight_decay_, bias_correction1, bias_correction2,
                                   d_sum_sq_, max_norm);
                        continue;
                    }
                #endif

                auto& param_data = params_[i]->mutable_data();
                auto& param_grad = params_[i]->grad().mutable_data();
                for (size_t j = 0; j < param_data.size(); ++j) {
                    m_[i][j] = beta1_ * m_[i][j] + (1 - beta1_) * param_grad[j];
                    v_[i][j] = beta2_ * v_[i][j] + (1 - beta2_) * (param_grad[j] * param_grad[j]);
                    scalar_t m_hat = m_[i][j] / bias_correction1;
                    scalar_t v_hat = v_[i][j] / bias_correction2;
                    param_data[j] -= lr_ * (m_hat / (std::sqrt(v_hat) + epsilon_) + weight_decay_ * param_data[j]);
                }
            }
        }

        void zero_grad() {
            for (auto& param : params_) {
                if (!param->requires_grad()) continue;

                param->init_grad();

                #ifdef NNFRAME_WITH_CUDA
                    if (param->device() == Device::CUDA) {
                        CUDA_CHECK(cudaMemset(param->grad().mutable_device_data(), 0, param->numel() * sizeof(scalar_t)));
                        continue;
                    }
                #endif

                for (size_t i = 0; i < param->numel(); i++) {
                    param->grad().mutable_data()[i] = 0.0f;
                }
            }
        };

};