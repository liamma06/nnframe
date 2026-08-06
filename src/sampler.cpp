#include "infer/sampler.h"
#include <algorithm>
#include <cassert>

Sampler::Sampler(unsigned seed) : rng_(seed) {};

size_t Sampler::sample(const TensorPtr& logits, float temperature, size_t top_k){
    /* 
        given a tensor of logits:
        1. apply temperature scaling
        2. turn into proba (softmax)
        3. select using top_k 
        4. zero out rest
        5. normalize the top K num (sum =1 )
        6. pick from weighted random pick and return token index chosen

        logits: [vocab_size]
        output: index of sampled token
    */
    auto scaled_logits = logits->mul(Tensor::create({1}, 1.0f / temperature));

    //softmax only take rank 2 and rank 3 (maybe change later)
    auto logits_2d = scaled_logits->reshape({1, scaled_logits->numel()}); 
    auto softmax_probs = logits_2d->softmax(1); 

    std::vector<std::pair<size_t, scalar_t>> prob_pairs; //[index, prob] for each token/logit
    for (size_t i = 0; i < softmax_probs->numel(); i++){
        prob_pairs.push_back({i, softmax_probs->at({0, i})});
    }

    assert(top_k <= prob_pairs.size() && "top_k cannot exceed vocab_size");


    std::sort(prob_pairs.begin(), prob_pairs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    prob_pairs.resize(top_k); 

    //normalize top_k 
    scalar_t sum_top_k = 0.0f;
    for (const auto& p : prob_pairs) {
        sum_top_k += p.second;
    }
    for (auto& p : prob_pairs) {
        p.second /= sum_top_k; 
    }

    std::vector<scalar_t> weights_only;
    for (const auto& p : prob_pairs) {
        weights_only.push_back(p.second);
    }

    //discrete distribution tells index inside, not the token ID
    std::discrete_distribution<size_t> dist(weights_only.begin(), weights_only.end());
    size_t chosen_position = dist(rng_);

    return prob_pairs[chosen_position].first;
}