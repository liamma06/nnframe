#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "core/tensor.h"
#include "modules/layer.h"
#include "modules/embed.h"
#include "modules/pos_embed.h"
#include "modules/transformer_block.h"
#include <cassert>

class CharModel : public Layer{
    private:
        size_t vocab_size_;
        size_t embedding_dim_;
        size_t num_heads_;
        size_t num_transformer_blocks_;
        Embed embedding_layer_;
        PositionalEmbed positional_embedding_layer_;
        std::vector<TransformerBlock> transformer_blocks_;

        //note: final linear layer to have proper output of vocab size 
        Linear lm_head_; 

    public:
        CharModel(size_t vocab_size, size_t embedding_dim, size_t num_heads, size_t num_transformer_blocks)
            : vocab_size_(vocab_size),
                embedding_dim_(embedding_dim),
                num_heads_(num_heads),
                num_transformer_blocks_(num_transformer_blocks),
                embedding_layer_(vocab_size, embedding_dim),
                positional_embedding_layer_(),
                lm_head_(embedding_dim, vocab_size) {

                    for (size_t i = 0; i < num_transformer_blocks_; i++){
                        transformer_blocks_.emplace_back(embedding_dim_, num_heads_);
                    }
                    
            };

        std::vector<TensorPtr> parameters() const override{
            std::vector<TensorPtr> params;

            auto embedding_params = embedding_layer_.parameters();
            params.insert(params.end(), embedding_params.begin(), embedding_params.end());

            for (const auto& block : transformer_blocks_){
                auto block_params = block.parameters();
                params.insert(params.end(), block_params.begin(), block_params.end());
            }

            auto lm_head_params = lm_head_.parameters();
            params.insert(params.end(), lm_head_params.begin(), lm_head_params.end());

            return params;
        };
        
        //from my understanding this is more so the training forwardpass
        TensorPtr forward(const TensorPtr& input) override{
            /*
                input: [seq_length]
                output: [seq_length, vocab_size]

                GOAL: take the sequence of tokens and return logits for the next token in sequence for each position in the input sequence.
                
                1. embed 
                2. add positional encoding
                3. pass through transformer blocks
                4. final linear layer to get logits for each token in vocab
            */
            
            TensorPtr x = embedding_layer_.forward(input);
            x = positional_embedding_layer_.forward(x);

            for (auto& block : transformer_blocks_){
                x = block.forward(x);
            }

            return lm_head_.forward(x);
        }

        TensorPtr forward(const TensorPtr& input, std::vector<KVCache>& kv_caches){
            /*
                -same as above but pass in KVCache 
                -used for prefill and decode (inference)
                -track token for the position forward
            */

            assert(kv_caches.size() == transformer_blocks_.size() && "must have one KVCache per transformer block");

            TensorPtr x = embedding_layer_.forward(input);

            

            size_t start_pos = kv_caches[0].get_k() ? kv_caches[0].get_k()->shape()[1] : 0;
            x = positional_embedding_layer_.forward(x, start_pos);

            //represent change in the embeddings (each gets thier own KVCache)
            for (size_t i = 0; i < transformer_blocks_.size(); i++){
                x = transformer_blocks_[i].forward(x, kv_caches[i]);
            }

            //use those embeddings to get logits which go into the sampler to find next token!
            return lm_head_.forward(x); 
        }


};