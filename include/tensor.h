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

    public: 
        Tensor(std::vector<size_t> shape, scalar_t fill_value = 0.0f); 
        Tensor (std::vector<size_t> shape, std::vector<scalar_t> data);

        const std::vector<size_t>& shape() const; //returns reference to shape vector read only (const)
        const std::vector<size_t>& strides() const;
        size_t rank()   const;
        size_t numel()  const;
        bool   is_contiguous() const;

        // Access
        scalar_t& at(std::initializer_list<size_t> indices);
        scalar_t  at(std::initializer_list<size_t> indices) const;
        const std::vector<scalar_t>& data() const;

        // Shape ops note it would still point to the same data but with different shape and strides
        Tensor reshape(std::vector<size_t> new_shape) const; 
        Tensor transpose() const;

        // Math ops(completely new tensor is returned )
        Tensor add(const Tensor& other) const;
        Tensor sub(const Tensor& other) const;
        Tensor mul(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;

        //to make it easier to use math symbols they just link to the tensor functions 
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;

        bool allclose(const Tensor& other, scalar_t eps = 1e-5f) const; //comparison testing 
};