#pragma once 
#include <vector>
#include <memory> 
#include <functional>

using scalar_t = float; //might change later
class Tensor; // forward declaration so TensorPtr can reference it
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor {
    private: 
        std::shared_ptr<std::vector<scalar_t>> data_; //pointer to tensor data
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_;

        //autograd 
        bool requires_grad_ = false;
        std::shared_ptr<Tensor> grad_; 
        std::function<void(const Tensor&)> grad_fn_; //store function to compute gradient
        std::vector<std::shared_ptr<Tensor>> inputs_; //store parents for backpropagation



        size_t compute_strides();

        Tensor(std::shared_ptr<std::vector<scalar_t>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset); // takes in data ptr so no need to recopy/buffer

    public: 
        Tensor(std::vector<size_t> shape, scalar_t fill_value = 0.0f); 
        Tensor (std::vector<size_t> shape, std::vector<scalar_t> data);

        const std::vector<size_t>& shape() const; //returns reference to shape vector read only (const)
        const std::vector<size_t>& strides() const;
        size_t rank()   const;
        size_t numel()  const;
        bool   is_contiguous() const;

        // Access
        scalar_t& at(std::initializer_list<size_t> indices); //return reference to value so it can be modified
        scalar_t  at(std::initializer_list<size_t> indices) const; //return copy read version
        scalar_t& at(const std::vector<size_t>& indices); 
        scalar_t  at(const std::vector<size_t>& indices) const;

        const std::vector<scalar_t>& data() const; //read only access to data vector 

        // Shape ops note it would still point to the same data but with different shape and strides
        TensorPtr reshape(std::vector<size_t> new_shape) const;
        TensorPtr transpose() const;
        TensorPtr permute(std::vector<size_t> axes) const;

        // Math ops(completely new tensor is returned )
        TensorPtr add(const TensorPtr& other) const;
        TensorPtr sub(const TensorPtr& other) const;
        TensorPtr mul(const TensorPtr& other) const;
        TensorPtr matmul(const TensorPtr& other) const;

        //to make it easier to use math symbols they just link to the tensor functions
        TensorPtr operator+(const TensorPtr& other) const;
        TensorPtr operator-(const TensorPtr& other) const;
        TensorPtr operator*(const TensorPtr& other) const;

        bool allclose(const Tensor& other, scalar_t eps = 1e-5f) const; //comparison testing 

        //autograd
        bool requires_grad() const;
        Tensor& grad();
        void set_requires_grad(bool val);
        void backward();
};