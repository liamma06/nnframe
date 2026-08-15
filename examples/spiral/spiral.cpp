#include "core/tensor.h"
#include "modules/linear.h"
#include "modules/gelu.h"
#include "modules/sequential.h"
#include "loss/cross_entrop.h"
#include "optim/adamw.h"
#include "spiral_eval.h"
#include <iostream>
#include <cmath>
#include <random>


int main(){
    const float PI = 3.14159265358979323846f;

    // spiral dataset: 3 classes, N points each, interleaved arms
    const size_t num_classes = 3;
    const size_t points_per_class = 100;
    const size_t num_points = num_classes * points_per_class;
    const float turns = 2.0f;
    const float noise = 0.15f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise_dist(-noise, noise);

    std::vector<scalar_t> input_data;   
    std::vector<scalar_t> target_data;  

    for (size_t c = 0; c < num_classes; c++) {
        for (size_t i = 0; i < points_per_class; i++) {
            float r = static_cast<float>(i) / points_per_class;
            float theta = (static_cast<float>(i) / points_per_class) * turns * 2.0f * PI
                        + c * (2.0f * PI / num_classes)
                        + noise_dist(rng);

            float x = r * std::cos(theta);
            float y = r * std::sin(theta);

            input_data.push_back(x);
            input_data.push_back(y);
            target_data.push_back(static_cast<scalar_t>(c));
        }
    }

    //main part 
    auto input_tensor = std::make_shared<Tensor>(std::vector<size_t>{num_points, 2}, input_data);
    auto target_tensor = Tensor::from_vector(target_data);

    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Layer>>{
        std::make_shared<Linear>(2, 16),
        std::make_shared<GELU>(),
        std::make_shared<Linear>(16, 16),
        std::make_shared<GELU>(),
        std::make_shared<Linear>(16, num_classes)
    });

    CrossEntropy loss_fn;
    AdamW optimizer(model->parameters(), 0.01f);

    for (int epoch = 0; epoch < 1000; ++epoch){
        auto logits = model->forward(input_tensor);
        auto loss = loss_fn.forward(logits, target_tensor);

        loss->backward();
        optimizer.step();
        optimizer.zero_grad();

        if (epoch % 100 == 0){
            std::cout << "Epoch " << epoch << ", Loss: " << loss->data()[0] << std::endl;
        }
    }

    evaluate_spiral_model(model, input_tensor, input_data, target_data, num_points, num_classes);
}