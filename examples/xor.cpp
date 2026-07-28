#include "core/tensor.h"
#include "layers/linear.h"
#include "layers/relu.h"
#include "layers/sequential.h"
#include "loss/mse.h"
#include "optim/adamw.h"
#include <iostream>

int main(){
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{4,2},
        std::vector<scalar_t>{
            0.0f, 0.0f, 
            0.0f, 1.0f, 
            1.0f, 0.0f,
            1.0f, 1.0f
        }
    );

    auto y = std::make_shared<Tensor>(
        std::vector<size_t>{4,1},
        std::vector<scalar_t>{
            0.0f, 
            1.0f, 
            1.0f, 
            0.0f
        }
    );

    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Layer>>{
        std::make_shared<Linear>(2, 8),
        std::make_shared<ReLu>(),
        std::make_shared<Linear>(8, 1)
    });

    MSE loss_fn;
    AdamW optimizer(model->parameters(), 0.01f);

    for (int epoch = 0; epoch < 1000; ++epoch) {
        auto predictions = model->forward(x);
        auto loss = loss_fn.forward(predictions, y);

        
        loss->backward();
        optimizer.step();
        optimizer.zero_grad();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss->data()[0] << std::endl;
        }
    }

    auto final_pred = model->forward(x);
    std::cout << "\nPredictions:\n";
    for (size_t i = 0; i < 4; i++) {
        std::cout << final_pred->data()[i] << " (target: " << y->data()[i] << ")\n";
    }
};