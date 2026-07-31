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


};