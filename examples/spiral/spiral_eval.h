#pragma once
#include "core/tensor.h"
#include "modules/sequential.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// argmax over num_classes scores starting at logits[row * num_classes]
inline size_t argmax_row(const scalar_t* logits, size_t row, size_t num_classes){
    size_t best_class = 0;
    float best_score = logits[row * num_classes];
    for (size_t c = 1; c < num_classes; c++){
        float score = logits[row * num_classes + c];
        if (score > best_score){ best_score = score; best_class = c; }
    }
    return best_class;
}

// prints train accuracy, then writes spiral_grid.csv (model prediction across the plane)
// and spiral_points.csv (the real training points) for visualization
inline void evaluate_spiral_model(
    const std::shared_ptr<Sequential>& model,
    const TensorPtr& input_tensor,
    const std::vector<scalar_t>& input_data,
    const std::vector<scalar_t>& target_data,
    size_t num_points,
    size_t num_classes
){
    // accuracy on the training points: argmax each row of logits, compare to target
    {
        auto logits = model->forward(input_tensor);
        size_t correct = 0;
        for (size_t i = 0; i < num_points; i++){
            size_t best_class = argmax_row(logits->data().data(), i, num_classes);
            if (static_cast<scalar_t>(best_class) == target_data[i]) correct++;
        }
        std::cout << "\nTrain accuracy: " << correct << " / " << num_points
                   << " (" << (100.0f * correct / num_points) << "%)\n";
    }

    // sweep a grid over the plane so we can see the decision boundary the model learned
    {
        const size_t grid_res = 80;
        const float grid_min = -1.3f;
        const float grid_max = 1.3f;

        std::vector<scalar_t> grid_data;
        for (size_t gy = 0; gy < grid_res; gy++){
            for (size_t gx = 0; gx < grid_res; gx++){
                float x = grid_min + (grid_max - grid_min) * gx / (grid_res - 1);
                float y = grid_min + (grid_max - grid_min) * gy / (grid_res - 1);
                grid_data.push_back(x);
                grid_data.push_back(y);
            }
        }
        auto grid_tensor = std::make_shared<Tensor>(std::vector<size_t>{grid_res * grid_res, 2}, grid_data);
        auto grid_logits = model->forward(grid_tensor);

        std::ofstream grid_file("spiral_grid.csv");
        grid_file << "x,y,predicted_class\n";
        for (size_t i = 0; i < grid_res * grid_res; i++){
            size_t best_class = argmax_row(grid_logits->data().data(), i, num_classes);
            grid_file << grid_data[i * 2] << "," << grid_data[i * 2 + 1] << "," << best_class << "\n";
        }
        grid_file.close();

        std::ofstream points_file("spiral_points.csv");
        points_file << "x,y,class\n";
        for (size_t i = 0; i < num_points; i++){
            points_file << input_data[i * 2] << "," << input_data[i * 2 + 1] << "," << target_data[i] << "\n";
        }
        points_file.close();

        std::cout << "Wrote spiral_grid.csv and spiral_points.csv\n";
    }
}
