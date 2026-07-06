#pragma once 
#include <vector>
#include <memory> 

using scalar_t = float; //might change later 

class Tensor {
    private: 
        std::shared_ptr<std::vector<scalar_t>> _data; //pointer to tensor data 
        std::vector<size_t> _shape; 
        std::vector<size_t> _strides; 
        size_t offset; 
};