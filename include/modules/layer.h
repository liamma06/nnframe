#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"

class Layer{
    public:
        virtual TensorPtr forward(const TensorPtr& input) = 0; // for subclass to define
        virtual ~Layer() = default;

        virtual std::vector<TensorPtr> parameters() const{
            return {};
        };

        virtual void set_parameters(const std::vector<TensorPtr>& params) {
            
        };
        
        void to(Device device){
            std::vector<TensorPtr> params = parameters();
            std::vector<TensorPtr> new_params_moved;
            for (auto& param : params){
                TensorPtr moved_param = param->to(device);
                moved_param->set_requires_grad(param->requires_grad());
                new_params_moved.push_back(moved_param);
            }
            //new tensor so need to realign 
            set_parameters(new_params_moved);
        }
};
