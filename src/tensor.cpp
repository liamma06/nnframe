#include "tensor.h" 
#include <cassert> //checks 
#include <numeric> 

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
        assert(index < shape_[i] && "Index out of bounds");
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

 const std::vector<scalar_t>&  Tensor::data() const{
    return *data_; 
 }

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {

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

    return Tensor(data_, new_shape, new_strides, offset_);
}