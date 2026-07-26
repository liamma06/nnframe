#pragma once 
#include <vector>
#include <memory> 
#include "tensor.h"
#include "layer.h"

class Sequential : public Layer{
    private: 
        /*
            did this prevously but we want to store different data types
            by having a vector of pointer it is same but point to different data types
            (found this interesting) 
        */
        std::vector<std::shared_ptr<Layer>> layers_;

    public:
        Sequential(std::vector<std::shared_ptr<Layer>> layers){
            layers_ = layers; 
        }

        TensorPtr forward(const TensorPtr& input) override{
            auto current = input; //point to same input tensor 

            for (const auto& layer : layers_){
                current = layer->forward(current);
            }
            return current;
        }

        std::vector<TensorPtr> parameters() const override{
            //store the tensor pointers of the wieghts and bias of each layer
            std::vector<TensorPtr> params; 

            for (const auto& layer : layers_){
                auto layer_params = layer->parameters(); 
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            } 

            return params;

        }


};