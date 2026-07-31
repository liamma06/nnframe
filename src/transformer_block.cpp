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
