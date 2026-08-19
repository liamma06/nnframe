#pragma once
#include <vector>
#include <memory>
#include "core/tensor.h"
#include "modules/layer.h"
#include "modules/attention.h"
#include "modules/layernorm.h"
#include "modules/linear.h"
#include "modules/gelu.h"
#include "infer/kv_block_pool.h"

class TransformerBlock : public Layer {
    private:
        SelfAttention self_attention_;
        LayerNorm layer_norm1_;
        LayerNorm layer_norm2_;
        Linear linear1_;
        Linear linear2_;
    
    public:
        TransformerBlock(size_t embed_dim, size_t num_heads);
        std::vector<TensorPtr> parameters() const override;
        void set_parameters(const std::vector<TensorPtr>& params) override;
        TensorPtr forward(const TensorPtr& input) override;

        TensorPtr forward(const TensorPtr& input, KVBlockPool& kv_pool, size_t sequence_id);
};