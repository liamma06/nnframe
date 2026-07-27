#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"

class SGD{

    private:
        //each linear has (weights and bais) those are the params 
        std::vector<TensorPtr> params_;
        float lr_; 

    public:

        SGD(std::vector<TensorPtr> params, float lr){
            params_ = params; 
            lr_ = lr; 
        }

        //update params with gradient descent step
        void step(){
            for (auto& param : params_){
                if (param->requires_grad()){
                    for (size_t i = 0; i < param->numel(); i++){
                        param->mutable_data()[i] -= lr_ * param->grad().mutable_data()[i]; 
                    }
                }
            }
        }

        //reset gradients to zero after each step/epoch
        void zero_grad(){
            for (auto& param : params_){
                if (param->requires_grad()){
                    param->init_grad(); 
                    for (size_t i = 0; i < param->numel(); i++){
                        param->grad().mutable_data()[i] = 0.0f;
                    }
                }
            }
        }
};