#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"

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
        size_t t_; //time step
    
    public:
        AdamW(std::vector<TensorPtr> params, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.01f)
            : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {
            m_.resize(params.size());
            v_.resize(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                m_[i].resize(params[i]->numel(), 0.0f);
                v_[i].resize(params[i]->numel(), 0.0f);
            }
        }

        void step() {
            t_++;
            for (size_t i = 0; i < params_.size(); ++i) {
                if (params_[i]->requires_grad()) {
                    auto& param_data = params_[i]->mutable_data();
                    auto& param_grad = params_[i]->grad().mutable_data();
                    for (size_t j = 0; j < param_data.size(); ++j) {
                        // Update biased first moment estimate
                        m_[i][j] = beta1_ * m_[i][j] + (1 - beta1_) * param_grad[j];
                        // Update biased second first moment estimate
                        v_[i][j] = beta2_ * v_[i][j] + (1 - beta2_) * (param_grad[j] * param_grad[j]);
                        // Compute bias correct first moment estimate
                        scalar_t m_hat = m_[i][j] / (1 - std::pow(beta1_, t_));
                        // Compute bias correct second first moment estimate
                        scalar_t v_hat = v_[i][j] / (1 - std::pow(beta2_, t_));
                        // Update parameters ->  decay
                        param_data[j] -= lr_ * (m_hat / (std::sqrt(v_hat) + epsilon_) + weight_decay_ * param_data[j]);
                    }
                }
            }
        }

        void zero_grad() {
            for (auto& param : params_) {
                if (param->requires_grad()) {
                    param->init_grad();
                    for (size_t i = 0; i < param->numel(); i++) {
                        param->grad().mutable_data()[i] = 0.0f;
                    }
                }
            }
        };

};