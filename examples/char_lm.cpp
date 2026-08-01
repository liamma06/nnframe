#include "core/tensor.h"
#include "modules/char_model.h"
#include "loss/cross_entrop.h"
#include "optim/adamw.h"  
#include <iostream>
#include <vector>
#include <string>
#include <map>

int main(){
    std::string text = "hello world hello world hello world";

    //tokenize 
    std::map<char, size_t> char_to_idx;
    size_t idx = 0;
    for (char c : text)
        if (char_to_idx.find(c) == char_to_idx.end())
            char_to_idx[c] = idx++;
            //end up with like h-0, e-> 1 and so on

    size_t vocab_size = char_to_idx.size();

    std::vector<size_t> tokenized;
    for (char c : text)
        tokenized.push_back(char_to_idx[c]);

    //input and map (each spot target is the next token)
    size_t seq_len = tokenized.size() - 1;

    std::vector<scalar_t> input_data, target_data;
    for (size_t i = 0; i < seq_len; i++) {
        input_data.push_back(static_cast<scalar_t>(tokenized[i]));
        target_data.push_back(static_cast<scalar_t>(tokenized[i + 1]));
    }

    auto input_tensor = std::make_shared<Tensor>(std::vector<size_t>{seq_len}, input_data);
    auto target_tensor = std::make_shared<Tensor>(std::vector<size_t>{seq_len}, target_data);

    //model
    auto embedding_dim = 16;
    auto num_heads = 1;
    auto num_transformer_blocks = 2;

    CharModel model(vocab_size, embedding_dim, num_heads, num_transformer_blocks);
    CrossEntropy loss_fn;
    AdamW optimizer(model.parameters(), 1e-3f);


    for (int epoch = 0; epoch < 5000; ++epoch) {
        optimizer.zero_grad();
        auto logits = model.forward(input_tensor);
        auto loss = loss_fn.forward(logits, target_tensor);

        loss->backward();
        optimizer.step();
        

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss->data()[0] << std::endl;
        }
    }




}