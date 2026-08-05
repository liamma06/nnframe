#include "core/tensor.h"
#include <cassert>
#include <numeric>
#include <cmath>
#include <unordered_set>

namespace {
    //func to see if we can increment index (odometer style)
    bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape){
        for (int i = idx.size() - 1; i >= 0 ; i--){
            if (++idx[i] < shape[i]) return true;
            idx[i] = 0;
        }
        return false;
    }
}

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

//factory methods 
TensorPtr Tensor::create(std::vector<size_t> shape, scalar_t fill) {
    return std::make_shared<Tensor>(shape, fill);
}

TensorPtr Tensor::zeros(std::vector<size_t> shape) {
    return Tensor::create(shape);
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

    auto result = TensorPtr(new Tensor(data_, new_shape, new_strides, offset_));

    /*
        Why need autograd here?
        because we creating a new tensor adn want gradient to flow through the new tensor too.
    */
    if (requires_grad_){
        result -> set_requires_grad(true);
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());

        result->set_inputs({self});
        result->set_grad_fn([self](const Tensor& upstream){
            if (self->requires_grad()){
                self->init_grad();
                for (size_t i = 0; i < self->numel(); i++){
                    self->grad().mutable_data()[i] += upstream.data()[i];
                }
            }
        });
    }

    return result;
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

    auto result = TensorPtr(new Tensor(data_, new_shape, new_strides, offset_));

    if (requires_grad_){
        result->set_requires_grad(true);
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        result->set_inputs({self});
        result->set_grad_fn([self, axes](const Tensor& upstream){
            if (self->requires_grad()){
                self->init_grad();

                /*
                    We need to map upstream grad into the new tensor
                    but we switched the axes so we need to permute the indices back to the original tensor's order
                    and add them into the correct spots 
                */
                for (size_t i = 0; i < self->numel(); i++){
                    std::vector<size_t> self_idx(self->rank());
                    size_t remaining = i;
                    for (int j = self->rank() - 1; j >= 0; j--){
                        self_idx[j] = remaining % self->shape()[j];
                        remaining /= self->shape()[j];
                    }

                    std::vector<size_t> out_idx(self->rank());
                    for (size_t j = 0; j < axes.size(); j++){
                        out_idx[j] = self_idx[axes[j]];
                    }

                    self->grad().mutable_data()[i] += upstream.at(out_idx);
                }
            }
        });
    }

    return result;
}

TensorPtr Tensor::transpose() const {
    assert(rank() == 2 && "transpose is for 2D tensors, use permute for higher ranks");
    auto result = permute({1, 0});
    if (requires_grad_) {
        result->set_requires_grad(true);
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        result->set_inputs({self});
        // grad of transpose is transpose of upstream gradient
        result->set_grad_fn([self](const Tensor& upstream) {
            if (self->requires_grad()) {
                self->init_grad();
                for (size_t i = 0; i < self->shape()[0]; i++)
                    for (size_t j = 0; j < self->shape()[1]; j++)
                        self->grad().mutable_data()[i * self->shape()[1] + j] += upstream.at({j, i});
            }
        });
    }
    return result;
}

std::vector<scalar_t>& Tensor::mutable_data(){
    /*
        Purpose: return a reference cause in the lambda grad_fn_
        we want to be able to modify the data of the tensor but data_ is private 
    */
    return *data_; 
}

TensorPtr Tensor::add(const TensorPtr& other) const{
    size_t out_rank = std::max(rank(), other->rank()); //take the larger rank/size of the 2 tensors
    std::vector<size_t> new_shape(out_rank);

    for (size_t i = 0; i < out_rank; i ++){
        //basically if the shape don't match or is 1 then bad
        size_t a = ( i < rank()) ? shape_[i] : 1;
        size_t b = ( i >= (out_rank - other->rank())) ? other->shape()[i-(out_rank - other->rank())] : 1;

        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for addition");

        new_shape[i] = std::max(a, b); //take the larger shape for the new tensor
    }

    auto output_tensor = Tensor::create(new_shape);

    //either should match requires grad 
    if (requires_grad_ || other->requires_grad_){
        output_tensor->requires_grad_ = true;
    }

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
        increment_index(cur_idx, new_shape);
    }

    /*
        shared pointer to current tensor so same reference count
        this important for autograd because we want to keep track of the parents and their reference counts so they don't get deleted before we can compute the gradients
        because I learned if the reference count goes to 0 then it removes it ignoring if it still used i nthe computation graph. 
    */
    auto self = std::const_pointer_cast<Tensor>(shared_from_this());

    output_tensor->inputs_ = std::vector<TensorPtr>{self,other}; //parents
    output_tensor->grad_fn_ = [self, other] (const Tensor& upstream){
        /*
            take incoming gradient from upstream 
            and compute the gradient for each parent/input tensor and add it to their gradient
            basically chain rule 
        */
        if (self->requires_grad_){
            self->init_grad();
            for (size_t i = 0; i < upstream.numel(); i++)
                self->grad_->mutable_data()[i % self->numel()] += upstream.data()[i];
        }

        if (other->requires_grad_){
            other->init_grad();
            for (size_t i = 0; i < upstream.numel(); i++)
                other->grad_->mutable_data()[i % other->numel()] += upstream.data()[i];
        }
    };

    return output_tensor;
}

TensorPtr Tensor::sub(const TensorPtr& other) const {
    size_t out_rank = std::max(rank(), other->rank());
    std::vector<size_t> new_shape(out_rank);

    for (size_t i = 0; i < out_rank; i++) {
        size_t a = (i < rank()) ? shape_[i] : 1;
        size_t b = (i >= (out_rank - other->rank())) ? other->shape()[i - (out_rank - other->rank())] : 1;
        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for subtraction");
        new_shape[i] = std::max(a, b);
    }

    auto output_tensor = Tensor::create(new_shape);
    if (requires_grad_ || other->requires_grad_)
        output_tensor->requires_grad_ = true;
    std::vector<size_t> cur_idx(out_rank, 0);

    for (size_t i = 0; i < output_tensor->numel(); i++) {
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());

        for (size_t j = 0; j < rank(); j++)
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];

        for (size_t j = 0; j < other->rank(); j++)
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];

        output_tensor->at(cur_idx) = at(a_idx) - other->at(b_idx);

        increment_index(cur_idx, new_shape);
    }

    auto self = std::const_pointer_cast<Tensor>(shared_from_this());

    output_tensor->inputs_ = std::vector<TensorPtr>{self,other}; //parents
    output_tensor->grad_fn_ = [self, other] (const Tensor& upstream){

        if (self->requires_grad_){
            self->init_grad();
            for (size_t i = 0; i < upstream.numel(); i++)
                self->grad_->mutable_data()[i % self->numel()] += upstream.data()[i];
        }

        if (other->requires_grad_){
            other->init_grad();
            for (size_t i = 0; i < upstream.numel(); i++)
                other->grad_->mutable_data()[i % other->numel()] -= upstream.data()[i];
        }
    };

    return output_tensor;
}

TensorPtr Tensor::mul(const TensorPtr& other) const {
    size_t out_rank = std::max(rank(), other->rank());
    std::vector<size_t> new_shape(out_rank);

    for (size_t i = 0; i < out_rank; i++) {
        size_t a = (i < rank()) ? shape_[i] : 1;
        size_t b = (i >= (out_rank - other->rank())) ? other->shape()[i - (out_rank - other->rank())] : 1;
        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for multiplication");
        new_shape[i] = std::max(a, b);
    }

    auto output_tensor = Tensor::create(new_shape);
    if (requires_grad_ || other->requires_grad_)
        output_tensor->requires_grad_ = true;
    std::vector<size_t> cur_idx(out_rank, 0);

    for (size_t i = 0; i < output_tensor->numel(); i++) {
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());

        for (size_t j = 0; j < rank(); j++)
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];

        for (size_t j = 0; j < other->rank(); j++)
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];

        output_tensor->at(cur_idx) = at(a_idx) * other->at(b_idx);

        increment_index(cur_idx, new_shape);
    }

    auto self = std::const_pointer_cast<Tensor>(shared_from_this());

    output_tensor->inputs_ = std::vector<TensorPtr>{self,other}; //parents
    output_tensor->grad_fn_ = [self, other] (const Tensor& upstream){

        if (self->requires_grad_){
            self->init_grad();
            for (size_t i = 0; i < self->numel(); i++){
                // if other is a broadcasted scalar always use index 0
                size_t j = (other->numel() == 1) ? 0 : i;
                self->grad_->mutable_data()[i] += upstream.data()[i] * other->data()[j];
            }
        }

        if (other->requires_grad_){
            other->init_grad();
            if (other->numel() == 1) {
                scalar_t sum = 0.0f;
                for (size_t i = 0; i < self->numel(); i++)
                    sum += upstream.data()[i] * self->data()[i];
                other->grad_->mutable_data()[0] += sum;
            } else {
                for (size_t i = 0; i < other->numel(); i++){
                    size_t j = (self->numel() == 1) ? 0 : i;
                    other->grad_->mutable_data()[i] += upstream.data()[j] * self->data()[j];
                }
            }
        }
    };

    return output_tensor;
}

TensorPtr Tensor::mean() const{
    auto output_tensor = Tensor::create({1}); //one value

    // sum -> mean
    for (size_t i = 0; i < numel(); i++){
        output_tensor->mutable_data()[0] += data_->at(i);
    }

    output_tensor->mutable_data()[0] /= static_cast<scalar_t>(numel()); 

    //grad calculation (1/N) * upstream
    if (requires_grad_){
        output_tensor->requires_grad_ = true;
    }
    auto self = std::const_pointer_cast<Tensor>(shared_from_this());
    output_tensor->inputs_ = std::vector<TensorPtr>{self};
    output_tensor->grad_fn_ = [self](const Tensor& upstream){
        if (self->requires_grad_){
            self->init_grad();
            scalar_t grad_contribution = upstream.data()[0] / static_cast<scalar_t>(self->numel());
            for (size_t i = 0; i < self->numel(); i++){
                self->grad_->mutable_data()[i] += grad_contribution;
            }
        }
    };

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
    assert(rank() == other->rank() && (rank() == 2 || rank() == 3) && "Both tensors must be rank 2 or rank 3, and match each other");
    assert(shape_[rank() - 1] == other->shape()[rank() - 2] && "Inner dimensions must match for matrix multiplication");

    if (rank() == 2){
        size_t M = shape_[0];
        size_t K = shape_[1];
        size_t N = other->shape()[1];

        auto output_tensor = Tensor::create({M, N});
        if (requires_grad_ || other->requires_grad_)
            output_tensor->requires_grad_ = true;

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

        auto self = std::const_pointer_cast<Tensor>(shared_from_this());

        output_tensor->inputs_ = std::vector<TensorPtr>{self, other}; //parents
        output_tensor->grad_fn_ = [self, other](const Tensor& upstream) {
            
            if (self->requires_grad_) {
                if (!self->grad_) {
                    self->grad_ = Tensor::create(self->shape_);
                }

                // Gradient w.r.t self: upstream * other^T
                for (size_t i = 0; i < self->shape_[0]; i++) {
                    for (size_t k = 0; k < self->shape_[1]; k++) {
                        scalar_t sum = 0.0f;
                        for (size_t j = 0; j < other->shape()[1]; j++) {
                            sum += upstream.at({i, j}) * other->at({k, j});
                        }
                        self->grad_->at({i, k}) += sum;
                    }
                }
            }

            if (other->requires_grad_) {
                if (!other->grad_) {
                    other->grad_ = Tensor::create(other->shape_);
                }

                // Gradient w.r.t other: self^T * upstream
                for (size_t k = 0; k < other->shape()[0]; k++) {
                    for (size_t j = 0; j < other->shape()[1]; j++) {
                        scalar_t sum = 0.0f;
                        for (size_t i = 0; i < self->shape_[0]; i++) {
                            sum += self->at({i, k}) * upstream.at({i, j});
                        }
                        other->grad_->at({k, j}) += sum;
                    }
                }
            }
        };

        return output_tensor;
    }else if (rank() == 3){
        /*
            for attention: [num_heads, seq_len, head_dim]
        */
        assert(shape_[0] == other->shape()[0] && "Leading (head) dimension must match for batched matmul");

        size_t L = shape_[0];
        size_t M = shape_[1];
        size_t K = shape_[2];
        size_t N = other->shape()[2];

        auto output_tensor = Tensor::create({L, M, N});

        if (requires_grad_ || other->requires_grad_)
            output_tensor->requires_grad_ = true;

        // same M/N/K dot-product as the 2D case, just looped once per leading (head) slice
        for (size_t l = 0; l < L; l++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    scalar_t sum = 0.0f;
                    for (size_t k = 0; k < K; k++) {
                        sum += at({l, m, k}) * other->at({l, k, n});
                    }
                    output_tensor->at({l, m, n}) = sum;
                }
            }
        }

        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        output_tensor->inputs_ = std::vector<TensorPtr>{self, other}; //parents
        output_tensor->grad_fn_ = [self, other, L, M, K, N](const Tensor& upstream) {

            if (self->requires_grad_) {
                self->init_grad();
                for (size_t l = 0; l < L; l++) {
                    for (size_t m = 0; m < M; m++) {
                        for (size_t k = 0; k < K; k++) {
                            scalar_t sum = 0.0f;
                            for (size_t n = 0; n < N; n++) {
                                sum += upstream.at({l, m, n}) * other->at({l, k, n});
                            }
                            self->grad_->at({l, m, k}) += sum;
                        }
                    }
                }
            }

            if (other->requires_grad_) {
                other->init_grad();
                for (size_t l = 0; l < L; l++) {
                    for (size_t k = 0; k < K; k++) {
                        for (size_t n = 0; n < N; n++) {
                            scalar_t sum = 0.0f;
                            for (size_t m = 0; m < M; m++) {
                                sum += self->at({l, m, k}) * upstream.at({l, m, n});
                            }
                            other->grad_->at({l, k, n}) += sum;
                        }
                    }
                }
            }
        };

        return output_tensor;
    }
    assert(false && "unreachable");
    return nullptr;
}

// AUTOGRAD

bool Tensor::requires_grad() const { return requires_grad_; }
void Tensor::set_requires_grad(bool val) { requires_grad_ = val; }
void Tensor::set_grad_fn(std::function<void(const Tensor&)> gradfn) { grad_fn_ = gradfn; }
void Tensor::init_grad() { if (!grad_) grad_ = Tensor::create(shape_); }
void Tensor::set_inputs(std::vector<TensorPtr> inputs) { inputs_ = inputs; }
Tensor& Tensor::grad() { return *grad_; }

void Tensor::backward(){

    //inital gradient just 1 
    grad_ = Tensor::create(shape_, 1.0f);

    std::vector<Tensor*> order_list;
    std::unordered_set<Tensor*> visited; 
    std::function<void(Tensor*)> dfs; 

    dfs = [&](Tensor* node){
        if (visited.count(node)) return;

        visited.insert(node);

        //reference loop (no copy) 
        for (auto& input : node->inputs_){
            dfs(input.get());
        }
        order_list.push_back(node); 


    };

    dfs(this); //start recursion

    //we want the this -> end ( 1 grad -> iterate down parents)
    std::reverse(order_list.begin(), order_list.end());

    //loop through each op and comptue the gradient for each passing the upstream gradients into each !
    for(auto& node : order_list){
        if (node->grad_fn_ && node->grad_){
            node->grad_fn_(*node->grad_);
        }
    }
}

//ATTENTION 

TensorPtr Tensor::softmax(size_t dim) const{
    /*
        eventhough we take in dim 
        it assumes it always down the last dimension of the tensor (the neurons)
        so it collects cross the rows(token) and normalizes the values in each row to sum to 1
    */

    assert(dim < rank() && "Dimension out of bounds for softmax");
    assert((rank() == 2 || rank() == 3) && "softmax only supports rank 2 or rank 3");
    assert(dim == rank() - 1 && "softmax currently only supports normalizing the last axis");

    TensorPtr output_tensor = Tensor::create(shape_);

    if (rank() == 2){

        for (size_t i = 0; i < shape_[0]; i++){

            // subtract row max (prevents exp overflow -> NaN)
            scalar_t row_max = at({i, 0});
            for (size_t j = 1; j < shape_[1]; j++)
                row_max = std::max(row_max, at({i, j}));

            //exp
            for (size_t j = 0; j < shape_[1]; j++){
                output_tensor->at({i, j}) = std::exp(at({i, j}) - row_max);
            };

            //sum
            scalar_t row_sum = 0.0f;
            for (size_t j = 0; j < shape_[1]; j++){
                row_sum += output_tensor->at({i, j});
            }

            //normalize
            for (size_t j = 0; j < shape_[1]; j++){
                output_tensor->at({i, j}) /= row_sum;
            }
        };

        if (requires_grad_){
            output_tensor->set_requires_grad(true);
        }

        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        output_tensor->set_inputs(std::vector<TensorPtr>{self});
        output_tensor->set_grad_fn([self, output_tensor](const Tensor& upstream){

            if (self->requires_grad()){
                self->init_grad();

                for (size_t i = 0; i < self->shape_[0]; i++){
                    scalar_t dot_product = 0.0f;
                    for (size_t j = 0; j < self->shape_[1]; j++){
                        dot_product += upstream.at({i, j}) * output_tensor->at({i, j});
                    }

                    for (size_t j = 0; j < self->shape_[1]; j++){
                        self->grad().mutable_data()[i * self->shape_[1] + j] += output_tensor->at({i, j}) * (upstream.at({i, j}) - dot_product);
                    }
                }
            }
        });
    }
    else if (rank() == 3){
        for (size_t i = 0; i < shape_[0]; i++){ //aditional rank 3 loop 
            for (size_t j = 0; j < shape_[1]; j++){

                scalar_t row_max = at({i, j, 0});
                for (size_t k = 1; k < shape_[2]; k++)
                    row_max = std::max(row_max, at({i, j, k}));

                for (size_t k = 0; k < shape_[2]; k++){
                    output_tensor->at({i, j, k}) = std::exp(at({i, j, k}) - row_max);
                };

                scalar_t row_sum = 0.0f;
                for (size_t k = 0; k < shape_[2]; k++){
                    row_sum += output_tensor->at({i, j, k});
                }

                for (size_t k = 0; k < shape_[2]; k++){
                    output_tensor->at({i, j, k}) /= row_sum;
                }
            }
        }

        if (requires_grad_){
            output_tensor->set_requires_grad(true);
        }

        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        output_tensor->set_inputs(std::vector<TensorPtr>{self});
        output_tensor->set_grad_fn([self, output_tensor](const Tensor& upstream){

            if (self->requires_grad()){
                self->init_grad();

                for (size_t i = 0; i < self->shape_[0]; i++){
                    for (size_t j = 0; j < self->shape_[1]; j++){
                        scalar_t dot_product = 0.0f;
                        for (size_t k = 0; k < self->shape_[2]; k++){
                            dot_product += upstream.at({i, j, k}) * output_tensor->at({i, j, k});
                        }

                        for (size_t k = 0; k < self->shape_[2]; k++){
                            self->grad().mutable_data()[i * self->shape_[1] * self->shape_[2] + j * self->shape_[2] + k] += output_tensor->at({i, j, k}) * (upstream.at({i, j, k}) - dot_product);
                        }
                    }
                }
            }
        });
        
    }
    return output_tensor;
}

TensorPtr Tensor::log() const {
    auto output_tensor = Tensor::create(shape_);
    for (size_t i = 0; i < numel(); i++)
        output_tensor->mutable_data()[i] = std::log(data()[i]);

    if (requires_grad_)
        output_tensor->set_requires_grad(true);

    auto self = std::const_pointer_cast<Tensor>(shared_from_this());
    output_tensor->set_inputs({self});
    // d/dx log(x) = 1/x
    output_tensor->set_grad_fn([self](const Tensor& upstream) {
        if (self->requires_grad()) {
            self->init_grad();
            for (size_t i = 0; i < self->numel(); i++)
                self->grad().mutable_data()[i] += upstream.data()[i] / self->data()[i];
        }
    });

    return output_tensor;
}

