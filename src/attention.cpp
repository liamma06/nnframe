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
        rank 2 (start)
            input: [seq_length, embed_dim]
            output: [seq_length, embed_dim]

        rank 3 (middle -> split into heads)
            Q: [seq_length, embed_dim]
            -> reshape  -> [seq_length, num_heads, head_dim]   (head_dim = embed_dim / num_heads)
            -> permute  -> [num_heads, seq_length, head_dim]


        rank 2 (end)
            input: [seq_length, embed_dim]
            output: [seq_length, embed_dim]
    */


    TensorPtr Q = input->matmul(W_q_);
    TensorPtr K = input->matmul(W_k_);
    TensorPtr V = input->matmul(W_v_);

    //rank 3 split into heads
    /*
        This confused me so ...
        We need to reshape first to break the data up into groups however to do the actual calcualtion 
        We want ALL of the head data to be togther 
        SO -> data split but merged togther 

        With permute we move the head dimension to the front so we can matmul all within the same head togther 
        and not mix between the heads 
    */
    TensorPtr Q_reshaped = Q->reshape({input->shape()[0],num_heads_,  head_dim_});
    TensorPtr K_reshaped = K->reshape({input->shape()[0],num_heads_,  head_dim_});
    TensorPtr V_reshaped = V->reshape({input->shape()[0],num_heads_, head_dim_});

    TensorPtr Q_head = Q_reshaped->permute({1, 0, 2});
    TensorPtr K_head = K_reshaped->permute({1, 0, 2});
    TensorPtr V_head = V_reshaped->permute({1, 0,2}); 

    TensorPtr K_transposed = K_head->permute({0, 2, 1}); //transpose last 2 dims for matmul
    TensorPtr scores = Q_head->matmul(K_transposed)->mul(Tensor::create({1}, 1.0f / std::sqrt(static_cast<float>(head_dim_))));


    //mask looping through each head and each token and setting to -inf if after
    auto masked_scores = Tensor::create(scores->shape());
    for (size_t i = 0; i < scores->shape()[0]; i++){
        for (size_t j = 0; j < scores->shape()[1]; j++){
            for (size_t k = 0; k < scores->shape()[2]; k++){
                if (k <= j)
                    masked_scores->mutable_data()[i * scores->shape()[1] * scores->shape()[2] + j * scores->shape()[2] + k] = scores->at({i,j,k});
                else
                    masked_scores->mutable_data()[i * scores->shape()[1] * scores->shape()[2] + j * scores->shape()[2] + k] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }

    masked_scores->set_inputs({scores});
    masked_scores->set_requires_grad(scores->requires_grad());
    masked_scores->set_grad_fn([scores](const Tensor& upstream){
        if (scores->requires_grad()){
            scores->init_grad();
                for (size_t i = 0; i < scores->shape()[0]; i++){
                    for (size_t j = 0; j < scores->shape()[1]; j++){
                        for (size_t k = 0; k < scores->shape()[2]; k++){
                            if (k <= j)
                                scores->grad().mutable_data()[i * scores->shape()[1] * scores->shape()[2] + j * scores->shape()[2] + k] += upstream.at({i,j,k});
                        }
                    }
                }
        }
    });

    TensorPtr attention_weights = masked_scores->softmax(2);
    TensorPtr attention_output = attention_weights->matmul(V_head);

    //turn the 3 rank back to 2 rank (all togther in W_o_)
    TensorPtr attention_output_reshaped = attention_output->permute({1, 0, 2})->contiguous()->reshape({input->shape()[0], embed_dim_});

    TensorPtr output = attention_output_reshaped->matmul(W_o_);

    return output;
}