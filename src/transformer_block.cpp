#include <modules/transformer_block.h>

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads)
    : self_attention_(embed_dim, num_heads),
        layer_norm1_(embed_dim),
        layer_norm2_(embed_dim),
        linear1_(embed_dim, embed_dim * 4),
        linear2_(embed_dim * 4, embed_dim) {}

std::vector<TensorPtr> TransformerBlock::parameters() const {
    std::vector<TensorPtr> params;
    auto self_attention_params = self_attention_.parameters();
    auto layer_norm1_params = layer_norm1_.parameters();
    auto layer_norm2_params = layer_norm2_.parameters();
    auto linear1_params = linear1_.parameters();
    auto linear2_params = linear2_.parameters();

    params.insert(params.end(), self_attention_params.begin(), self_attention_params.end());
    params.insert(params.end(), layer_norm1_params.begin(), layer_norm1_params.end());
    params.insert(params.end(), layer_norm2_params.begin(), layer_norm2_params.end());
    params.insert(params.end(), linear1_params.begin(), linear1_params.end());
    params.insert(params.end(), linear2_params.begin(), linear2_params.end());

    return params;
}

void TransformerBlock::set_parameters(const std::vector<TensorPtr>& params){
    size_t i = 0;

    size_t n = self_attention_.parameters().size();
    self_attention_.set_parameters({params.begin() + i, params.begin() + i + n});
    i += n;

    n = layer_norm1_.parameters().size();
    layer_norm1_.set_parameters({params.begin() + i, params.begin() + i + n});
    i += n;

    n = layer_norm2_.parameters().size();
    layer_norm2_.set_parameters({params.begin() + i, params.begin() + i + n});
    i += n;

    n = linear1_.parameters().size();
    linear1_.set_parameters({params.begin() + i, params.begin() + i + n});
    i += n;

    n = linear2_.parameters().size();
    linear2_.set_parameters({params.begin() + i, params.begin() + i + n});
    i += n;
}

TensorPtr TransformerBlock::forward(const TensorPtr& input){
    /*
        input: [seq_length, embed_dim]
        output: [seq_length, embed_dim]

        1. LayerNorm
        2. Self-Attention
        3. Residual Connection
        4. LayerNorm
        5. Feedforward Network (Linear -> GELU -> Linear)

        in the end we have "richer" embeddings from the tokens!
        https://poloclub.github.io/transformer-explainer/
    */

    TensorPtr pre_norm_input = layer_norm1_.forward(input);

    TensorPtr attn_output = self_attention_.forward(pre_norm_input);

    // residual: add delta of embeddings from attention to the input embeddings !
    TensorPtr residual1 = attn_output->add(input);
    TensorPtr pre_norm2 = layer_norm2_.forward(residual1);
    
    TensorPtr linear_output1 = linear1_.forward(pre_norm2);
    TensorPtr gelu_output = GELU().forward(linear_output1);
    TensorPtr linear_output2 = linear2_.forward(gelu_output);

    TensorPtr output = linear_output2->add(residual1);
    return output;
}

TensorPtr TransformerBlock::forward(const TensorPtr& input, KVBlockPool& kv_pool, size_t sequence_id){
    /*
        Same as above but pass in KVBlockPool to attention
    */
    TensorPtr pre_norm_input = layer_norm1_.forward(input);
    TensorPtr attn_output = self_attention_.forward(pre_norm_input, kv_pool, sequence_id);
    TensorPtr residual1 = attn_output->add(input);
    TensorPtr pre_norm2 = layer_norm2_.forward(residual1);
    
    TensorPtr linear_output1 = linear1_.forward(pre_norm2);
    TensorPtr gelu_output = GELU().forward(linear_output1);
    TensorPtr linear_output2 = linear2_.forward(gelu_output);

    TensorPtr output = linear_output2->add(residual1);
    return output;
}
