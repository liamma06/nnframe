#include "tensor.h"
#include <cassert>
#include <numeric>
#include <cmath>

Tensor::Tensor(std::vector<size_t> shape, scalar_t fill){
    shape_ = shape;
    offset_ = 0;
    size_t stride = compute_strides(); //shape - > stride 

    //allocate memory 
    data_ = std::make_shared<std::vector<scalar_t>>(stride, fill); //ptr to vector fill
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<scalar_t> data){
    shape_ = shape;
    offset_ = 0;
    size_t stride = compute_strides();

    //verify 
    assert(data.size() == stride && "Data size does not match shape size");

    data_ = std::make_shared<std::vector<scalar_t>>(data); //direct input data
}

Tensor::Tensor(std::shared_ptr<std::vector<scalar_t>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) {
    data_ = data;
    shape_ = shape;
    strides_ = strides;
    offset_ = offset;
}

//hlper stride func 
size_t Tensor::compute_strides() {
    strides_.resize(shape_.size()); //match shape size
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        strides_[i] = stride; //save stride_ {..., ...} 
        stride *= shape_[i];
    }
    return stride;  //total
}


const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

size_t Tensor::rank() const {
    return shape_.size();
}

size_t Tensor:: numel() const{
    return data_->size(); //pointer access 
}

bool Tensor::is_contiguous() const {
    //compute "new" shape stride and see if matches old stride 
    size_t expected = 1;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

scalar_t& Tensor::at(std::initializer_list<size_t> indices) {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    size_t i = 0;
    for (auto index: indices){
        assert(index < shape_[i] && "Index so of bounds");
        pos += index * strides_[i];
        i++;
    }
    return (*data_)[pos]; //dereference pointer and return reference to specific val 
}

scalar_t Tensor::at(std::initializer_list<size_t> indices) const {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    size_t i = 0;
    for (auto index : indices) {
        assert(index < shape_[i] && "Index out of bounds");
        pos += index * strides_[i];
        i++;
    }
    return (*data_)[pos]; //only copy val return because of const 
}

scalar_t& Tensor::at(const std::vector<size_t>& indices) {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    for (size_t i = 0; i < indices.size(); i++) {
        assert(indices[i] < shape_[i] && "Index out of bounds");
        pos += indices[i] * strides_[i];
    }
    return (*data_)[pos];
}

scalar_t Tensor::at(const std::vector<size_t>& indices) const {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    for (size_t i = 0; i < indices.size(); i++) {
        assert(indices[i] < shape_[i] && "Index out of bounds");
        pos += indices[i] * strides_[i];
    }
    return (*data_)[pos];
}

 const std::vector<scalar_t>&  Tensor::data() const{
    return *data_;
 }

TensorPtr Tensor::reshape(std::vector<size_t> new_shape) const {

    assert(is_contiguous() && "Tensor must be contiguous to reshape");

    size_t new_numel = 1;
    for (size_t dim : new_shape) {
        new_numel *= dim;
    }
    assert(new_numel == numel() && "New shape must have the same number of elements as the original tensor");

    // compute new strides locally don't replace the old ones because we want to keep the original tensor intact
    std::vector<size_t> new_strides(new_shape.size());
    size_t stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; i--) {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }

    return TensorPtr(new Tensor(data_, new_shape, new_strides, offset_));
}

TensorPtr Tensor::permute(std::vector<size_t> axes) const{
    assert(axes.size() == shape_.size() && "Axes length must match tensor rank");

    for (size_t a: axes){
        assert(a < shape_.size() && "Axis index out of bounds");
    }

    std::vector<bool> seen(shape_.size(), false);
    for (size_t a: axes){
        assert(!seen[a] && "Axes must be unique");
        seen[a] = true;
    }

    std::vector<size_t> new_shape(shape_.size());
    std::vector<size_t> new_strides(strides_.size());

    //swapping shape and strides based on axes
    for (size_t i = 0; i <axes.size(); i++){
        new_shape[i] = shape_[axes[i]];
        new_strides[i] = strides_[axes[i]];
    }
    return TensorPtr(new Tensor(data_, new_shape, new_strides, offset_));
}

TensorPtr Tensor::transpose() const {
    assert(rank() == 2 && "transpose is for 2D tensors, use permute for higher ranks");
    return permute({1, 0});
}

TensorPtr Tensor::add(const TensorPtr& other) const{
    size_t out_rank = std::max(rank(), other->rank()); //take the larger rank/size of the 2 tensors
    std::vector<size_t> new_shape(out_rank);

    for (size_t i = 0; i < out_rank; i ++){
        //basically if the shape don't match or is 1 then bad
        size_t a = ( i < rank()) ? shape_[i] : 1;
        size_t b = ( i < other->rank()) ? other->shape()[i] : 1;

        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for addition");

        new_shape[i] = std::max(a, b); //take the larger shape for the new tensor
    }

    auto output_tensor = std::make_shared<Tensor>(new_shape, 0.0f);

    //we don't know the rank so we cant just nest for loops
    //odometer

    std::vector<size_t> cur_idx(out_rank, 0);  //match dim

    for (size_t i = 0; i < output_tensor->numel(); i++){
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());
        /*
            compute indicies for values to add based on curr index.
            if shape 1 then just use 0 otherwise use the right index because that is what is impacting
            [1,2,3] only the 2nd and 3rd dim are impacting the output so we just use those indicies for the first tensor
        */

        for (size_t j =0; j < rank(); j++){
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];
        }

        for (size_t j = 0; j < other->rank(); j++){
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];
        }

        output_tensor->at(cur_idx) = at(a_idx) + other->at(b_idx);

        //does the odometer incrementing thing to get the next index for the output tensor
        for (int k = out_rank - 1; k >= 0; k--) {
            if (++cur_idx[k] < new_shape[k]) break;
            cur_idx[k] = 0;
        }
    }

    
    return output_tensor;
}

TensorPtr Tensor::sub(const TensorPtr& other) const {
    size_t out_rank = std::max(rank(), other->rank());
    std::vector<size_t> new_shape(out_rank);

    for (size_t i = 0; i < out_rank; i++) {
        size_t a = (i < rank()) ? shape_[i] : 1;
        size_t b = (i < other->rank()) ? other->shape()[i] : 1;
        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for subtraction");
        new_shape[i] = std::max(a, b);
    }

    auto output_tensor = std::make_shared<Tensor>(new_shape, 0.0f);
    std::vector<size_t> cur_idx(out_rank, 0);

    for (size_t i = 0; i < output_tensor->numel(); i++) {
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());

        for (size_t j = 0; j < rank(); j++)
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];

        for (size_t j = 0; j < other->rank(); j++)
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];

        output_tensor->at(cur_idx) = at(a_idx) - other->at(b_idx);

        for (int k = out_rank - 1; k >= 0; k--) {
            if (++cur_idx[k] < new_shape[k]) break;
            cur_idx[k] = 0;
        }
    }

    return output_tensor;
}

TensorPtr Tensor::mul(const TensorPtr& other) const {
    size_t out_rank = std::max(rank(), other->rank());
    std::vector<size_t> new_shape(out_rank);

    for (size_t i = 0; i < out_rank; i++) {
        size_t a = (i < rank()) ? shape_[i] : 1;
        size_t b = (i < other->rank()) ? other->shape()[i] : 1;
        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for multiplication");
        new_shape[i] = std::max(a, b);
    }

    auto output_tensor = std::make_shared<Tensor>(new_shape, 0.0f);
    std::vector<size_t> cur_idx(out_rank, 0);

    for (size_t i = 0; i < output_tensor->numel(); i++) {
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());

        for (size_t j = 0; j < rank(); j++)
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];

        for (size_t j = 0; j < other->rank(); j++)
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];

        output_tensor->at(cur_idx) = at(a_idx) * other->at(b_idx);

        for (int k = out_rank - 1; k >= 0; k--) {
            if (++cur_idx[k] < new_shape[k]) break;
            cur_idx[k] = 0;
        }
    }

    return output_tensor;
}

bool Tensor::allclose(const Tensor& other, scalar_t eps) const {
    assert(shape_ == other.shape_ && "Shapes must match for allclose");
    for (size_t i = 0; i < data_->size(); i++) {
        if (std::abs((*data_)[i] - (*other.data_)[i]) > eps) return false;
    }
    return true;
}

TensorPtr Tensor::operator+(const TensorPtr& other) const { return add(other); }
TensorPtr Tensor::operator-(const TensorPtr& other) const { return sub(other); }
TensorPtr Tensor::operator*(const TensorPtr& other) const { return mul(other); }

TensorPtr Tensor::matmul(const TensorPtr& other) const {
    assert(rank() == 2 && other->rank() == 2 && "Both tensors must be 2D for matrix multiplication");
    assert(shape_[1] == other->shape()[0] && "Inner dimensions must match for matrix multiplication");

    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = other->shape()[1];

    auto output_tensor = std::make_shared<Tensor>(std::vector<size_t>{M, N}, 0.0f);

    /*
        dot product for each element in the output tensor

        For every sample(i) loop through the neurons(j) and compute the dot products with
        the inputs(k) and the weights(k) of the other tensor.

        therefore the inputs and weights have to match the same size

        for each sample and each neuron the full input dot weights is stored in output tensor
    */
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            scalar_t sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += at({i, k}) * other->at({k, j});
            }
            output_tensor->at({i, j}) = sum;
        }
    }

    return output_tensor;
}

// AUTOGRAD

bool Tensor::requires_grad() const { return requires_grad_; }
void Tensor::set_requires_grad(bool val) { requires_grad_ = val; }
Tensor& Tensor::grad() { return *grad_; }

void Tensor::backward(){

    //inital gradient just 1 
    grad_ = std::make_shared<Tensor>(shape_, 1.0f); 


}