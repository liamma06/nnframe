#include "modules/attention.h"
#include <random>
#include <limits>

#ifdef NNFRAME_WITH_CUDA
    #include "cuda/attention_cuda.cuh"
    #include "cuda/elementwise_cuda.cuh"
#endif

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
    TensorPtr raw_scores = Q_head->matmul(K_transposed);
    scalar_t score_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    TensorPtr scores;

    #ifdef NNFRAME_WITH_CUDA
        if (raw_scores->device() == Device::CUDA){
            scalar_t* d_scaled = nullptr;
            CUDA_CHECK(cudaMalloc(&d_scaled, raw_scores->numel() * sizeof(scalar_t)));
            scale_tensor_cuda(raw_scores->device_data(), d_scaled, score_scale, raw_scores->numel());
            scores = Tensor::from_device_ptr(d_scaled, raw_scores->shape(), raw_scores->strides());

            scores->set_inputs({raw_scores});
            scores->set_requires_grad(raw_scores->requires_grad());
            scores->set_grad_fn([raw_scores, score_scale](const Tensor& upstream){
                if (raw_scores->requires_grad()){
                    raw_scores->init_grad();
                    scale_tensor_grad_cuda(upstream.device_data(), raw_scores->grad().mutable_device_data(), score_scale, raw_scores->numel());
                }
            });
        }
    #endif

    if (raw_scores->device() == Device::CPU){
        scores = raw_scores->mul(Tensor::create({1}, score_scale));
    }

    TensorPtr masked_scores;

    #ifdef NNFRAME_WITH_CUDA
        if (scores->device() == Device::CUDA){
            scalar_t* d_out = nullptr;

            size_t heads = scores->shape()[0];
            size_t seq_len_q = scores->shape()[1];
            size_t seq_len_k = scores->shape()[2];

            CUDA_CHECK(cudaMalloc(&d_out, scores->numel() * sizeof(scalar_t)));
            causal_mask_cuda(scores->device_data(), d_out, heads, seq_len_q, seq_len_k);
            masked_scores = Tensor::from_device_ptr(d_out, scores->shape(), scores->strides());
        

            masked_scores->set_inputs({scores});
            masked_scores->set_requires_grad(scores->requires_grad());
            masked_scores->set_grad_fn([scores](const Tensor& upstream){
                if (scores->requires_grad()){
                    size_t heads = scores->shape()[0];
                    size_t seq_len_q = scores->shape()[1];
                    size_t seq_len_k = scores->shape()[2];

                    scores->init_grad();
                    causal_mask_grad_cuda(upstream.device_data(), scores->grad().mutable_device_data(), heads, seq_len_q, seq_len_k);
                }
            });
        }   
    #endif 
    
    if (scores->device() == Device::CPU){
        //mask looping through each head and each token and setting to -inf if after
        masked_scores = Tensor::create(scores->shape());
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
    }

    TensorPtr attention_weights = masked_scores->softmax(2);
    TensorPtr attention_output = attention_weights->matmul(V_head);

    //turn the 3 rank back to 2 rank (all togther in W_o_)
    TensorPtr attention_output_reshaped = attention_output->permute({1, 0, 2})->contiguous()->reshape({input->shape()[0], embed_dim_});

    TensorPtr output = attention_output_reshaped->matmul(W_o_);

    return output;
}

TensorPtr SelfAttention::forward(const TensorPtr& input, KVCache& kv_cache){
    /*
        Same forward but KV Cache used 
        1. compute new K and V from input
        2. append to KVCache
        3. use full K and V from KVcache for calculation
        4. Q is compared to all K and V from cache

        use is also different:
            - other forward for trianing
            - this for prefill and decode (inference)
    */

    TensorPtr Q = input->matmul(W_q_);
    TensorPtr K = input->matmul(W_k_);
    TensorPtr V = input->matmul(W_v_);

    //split into heads then input into the KVCACHE 
    TensorPtr K_split = K->reshape({input->shape()[0],num_heads_,  head_dim_})->permute({1, 0, 2});
    TensorPtr V_split = V->reshape({input->shape()[0],num_heads_,  head_dim_})->permute({1, 0, 2});

    kv_cache.append(K_split, V_split);

    TensorPtr full_K = kv_cache.get_k();
    TensorPtr full_V = kv_cache.get_v();


    TensorPtr Q_split = Q->reshape({input->shape()[0],num_heads_,  head_dim_})->permute({1, 0, 2});

    //scores against all K in cache !
    TensorPtr K_transposed = full_K->permute({0, 2, 1});
    TensorPtr raw_scores = Q_split->matmul(K_transposed);
    scalar_t score_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    TensorPtr scores;

    #ifdef NNFRAME_WITH_CUDA
        if (raw_scores->device() == Device::CUDA){
            scalar_t* d_scaled = nullptr;
            CUDA_CHECK(cudaMalloc(&d_scaled, raw_scores->numel() * sizeof(scalar_t)));
            scale_tensor_cuda(raw_scores->device_data(), d_scaled, score_scale, raw_scores->numel());
            scores = Tensor::from_device_ptr(d_scaled, raw_scores->shape(), raw_scores->strides());
        }
    #endif

    if (raw_scores->device() == Device::CPU){
        scores = raw_scores->mul(Tensor::create({1}, score_scale));
    }

    /*
        maksing is a lil different cause in prefil:
            - we compare multiple new tokens so could have leak
            - but in decode it the newest token so no future ones 
    */

    TensorPtr masked_scores;

    if (input->shape()[0] > 1){ // more than one token [seq_len, embded_dim]

        #ifdef NNFRAME_WITH_CUDA
            if (scores->device() == Device::CUDA){
                scalar_t* d_out = nullptr;

                size_t heads = scores->shape()[0];
                size_t seq_len_q = scores->shape()[1];
                size_t seq_len_k = scores->shape()[2];

                CUDA_CHECK(cudaMalloc(&d_out, scores->numel() * sizeof(scalar_t)));
                causal_mask_cuda(scores->device_data(), d_out, heads, seq_len_q, seq_len_k);
                masked_scores = Tensor::from_device_ptr(d_out, scores->shape(), scores->strides());
            }
        #endif

        if (scores->device() == Device::CPU){
            masked_scores = Tensor::create(scores->shape());
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
        }
    }
    else{
        masked_scores = scores;
    }

    //softmax -> matmul(V) -> Wo -> output (standard/same) 
    TensorPtr attention_weights = masked_scores->softmax(2);
    TensorPtr attention_output = attention_weights->matmul(full_V);
    TensorPtr attention_output_reshaped = attention_output->permute({1, 0, 2})->contiguous()->reshape({input->shape()[0], embed_dim_});
    TensorPtr output = attention_output_reshaped->matmul(W_o_);

    return output;
}