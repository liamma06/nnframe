#include "modules/attention.h"
#include <random>



SelfAttention::SelfAttention(size_t embed_dim, size_t num_heads){
    std::mt19937 rng(42);
    // Xavier-style scale 
    float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim));
    std::uniform_real_distribution<float> dist(-scale, scale);

    assert(embed_dim % num_heads == 0 && "Embedding dimension must be divisible by number of heads");
    embed_dim_ = embed_dim;
    num_heads_ = num_heads;
    head_dim_ = embed_dim / num_heads;

    W_q_ = Tensor::create({embed_dim_, embed_dim_}, 0.0f);
    W_k_ = Tensor::create({embed_dim_, embed_dim_}, 0.0f);
    W_v_ = Tensor::create({embed_dim_, embed_dim_}, 0.0f);
    W_o_ = Tensor::create({embed_dim_, embed_dim_}, 0.0f);

    W_q_->set_requires_grad(true);
    W_k_->set_requires_grad(true);
    W_v_->set_requires_grad(true);
    W_o_->set_requires_grad(true);

    for (size_t i = 0; i < W_q_->numel(); i++){
        W_q_->mutable_data()[i] = dist(rng);
        W_k_->mutable_data()[i] = dist(rng);
        W_v_->mutable_data()[i] = dist(rng);
        W_o_->mutable_data()[i] = dist(rng);
    }
}

std::vector<TensorPtr> SelfAttention::parameters() const{
    return {W_q_, W_k_, W_v_, W_o_};
}

TensorPtr SelfAttention::forward(const TensorPtr& input){
    /*
        input: [seq_length, embed_dim]
        output: [seq_length, embed_dim]
    */

    TensorPtr Q = input->matmul(W_q_);
    TensorPtr K = input->matmul(W_k_);
    TensorPtr V = input->matmul(W_v_);

    TensorPtr scores = (Q->matmul(K->transpose()))->mul(Tensor::create({1}, 1.0f / std::sqrt(static_cast<float>(head_dim_))));
    
    //mask and it the only option we need the grad_fn everything else is handled 
    auto masked_scores = Tensor::create(scores->shape());
    for (size_t i = 0; i < scores->shape()[0]; i++){
        for (size_t j = 0; j < scores->shape()[1]; j++){
            masked_scores->at({i,j}) = (j > i) ? -1e9f : scores->at({i,j});
        }
    }
    masked_scores->set_inputs({scores});
    masked_scores->set_requires_grad(scores->requires_grad());
    masked_scores->set_grad_fn([scores](const Tensor& upstream){
        if (scores->requires_grad()){
            scores->init_grad();
            for (size_t i = 0; i < scores->shape()[0]; i++){
                for (size_t j = 0; j < scores->shape()[1]; j++){
                    if (j <= i)
                        scores->grad().mutable_data()[i * scores->shape()[1] + j] += upstream.at({i,j});
                }
            }
        }
    });

    TensorPtr attention_weights = masked_scores->softmax(1);
    TensorPtr attention_output = attention_weights->matmul(V);
    TensorPtr output = attention_output->matmul(W_o_);

    return output;
}