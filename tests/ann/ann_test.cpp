#include "../src/matrix/matrix.h"
#include "../src/functions/functions.h"
#include "../src/ann/ann.h"
#include <iostream>
#include <cmath>
#include <random>

int test_forward() {
    ANN ann({2, 500, 500, 1}, {"ReLu","ReLu", "ReLu"});
    float v[2][1] = {{1.0}, {2.0}}; 
    Matrix input(2, 1, *v);
    ann.forward(input);
    
    std::cout << "test_forward passed.\n";
    return 0;
}

int test_backprop() {
    ANN ann({2, 500, 250, 1}, {"ReLu","ReLu", "ReLu"});
    float v[2][1] = {{1.0}, {2.0}}; 
    Matrix input(2, 1, *v);
    ann.forward(input);
    ann.backprop();

    std::cout << "test_backprop passed.\n";
    return 0;
}

int test_calcualte_loss() {
    ANN ann({2, 500, 250, 2}, {"ReLu","ReLu", "ReLu"});
    float v[2][1] = {{1.0}, {2.0}}; 
    Matrix input(2, 1, *v);
    Matrix target(2, 1, *v);
    ann.forward(input);
    ann.calcualte_loss(target);
    ann.backprop();
    std::cout << "test_backprop passed.\n";
    return 0;
}

int test_one_sample_training() {
    ANN ann({2, 10, 2}, {"ReLu", "linear"});
    float v[2][1] = {{1.0}, {2.0}}; 
    Matrix input(2, 1, *v);
    Matrix target(2, 1, *v);
    ann.set_optimizer("SGD", "MSE", 0.1f);
    for (int ct = 0; ct < 100; ct++) {
        //std::cout << "Training iteration: " << ct << "\n";
        ann.forward(input);
        ann.reset_gradients(); // Reset gradients before each training step
        ann.calcualte_loss(target);
        ann.backprop();
        ann.update_weights(); // Update weights after backpropagation
    }
    
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 1; c++) {
            if (std::abs(target.get_val(r, c) - ann.get_output_val(r, c) > 1e-4)) {               
            std::cout << "test_one_sample_training FAILED at row " << r << " col "<<  c<< "\n";
            std::cout << "Expected: " << target.get_val(r, c) << ", Got: " << ann.get_output_val(r, c) << "\n";
            return -1;
            }
        }
    }
    std::cout << "test_one_sample_training passed.\n";
    return 0;
}

int test_set_optimizer_valid() {
    ANN ann({2, 3, 1}, {"ReLu", "linear"});
    try {
        ann.set_optimizer("SGD", "MSE", 0.05f);
        ann.set_optimizer("SGD", "Cross_Entropy", 0.01f);
        std::cout << "test_set_optimizer_valid passed.\n";
        return 0;
    } catch (...) {
        std::cout << "test_set_optimizer_valid failed.\n";
        return -1;
    }
}

int test_set_optimizer_invalid() {
    ANN ann({2, 3, 1}, {"ReLu", "linear"});
    try {
        ann.set_optimizer("Adam", "MSE", 0.01f);
        std::cout << "test_set_optimizer_invalid failed (no exception).\n";
        return -1;
    } catch (...) {
        std::cout << "test_set_optimizer_invalid passed.\n";
        return 0;
    }
}

void generate_smaples(int num_of_samples, std::vector<std::array<Matrix, 2>>& samples) {
    std::random_device rd;  // non-deterministic seed source
    std::mt19937 sample_gen(rd()); // Mersenne Twister engine seeded with rd()
    // Define a distribution in the range [0.0, 1.0)
    std::uniform_real_distribution<float> sample_dist(-100.0f, 100.0f);

    for (int i = 0; i < num_of_samples; i++) {
        float x1 = sample_dist(sample_gen);
        float x2 = sample_dist(sample_gen);
        float x3 = sample_dist(sample_gen);
        
        float input_vals[3][1] = {{x1}, {x2}, {x3}};
        float y1 = x1 + x2 - x3;
        float y2 = x1 * x2 * x3;
        float y3 = std::sin(x1) + x2 + std::cos(x3);
        float y4 = x1 + std::pow(x2, 3) + std::pow(x3, 2);
        float target_vals[4][1] = {{y1}, {y2}, {y3}, {y4}};
        //float target_vals[1][1] = {{y1}};
        
        Matrix input_matrix(3, 1, *input_vals);
        Matrix target_matrix(4, 1, *target_vals);
        //Matrix target_matrix(1, 1, *target_vals);
        std::array<Matrix, 2> innerList = {input_matrix, target_matrix};
        samples.push_back(innerList);
    }

}

void normalize_set(Matrix& input_min, Matrix& input_max, 
                   Matrix& target_min, Matrix& target_max, 
                   std::vector<std::array<Matrix, 2>>& samples) {
                    
    for (auto& [input, target] : samples) {
        for (int i = 0; i < input.get_rows_num(); i++) {
            for (int j = 0; j < input.get_columns_num(); j++) {
                input.set_val(i, j, (input.get_val(i, j) - input_min.get_val(i, j)) / 
                                   (input_max.get_val(i, j) - input_min.get_val(i, j)));
            }
        }
        for (int i = 0; i < target.get_rows_num(); i++) {
            for (int j = 0; j < target.get_columns_num(); j++) { 
                target.set_val(i, j, (target.get_val(i, j) - target_min.get_val(i, j)) / 
                                   (target_max.get_val(i, j) - target_min.get_val(i, j)));
            }
        }
    }

}

int run_ann_tests() {
    int status = 0;

    std::cout << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << "##############  RUNNING ANN TESTS... ##############" << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << std::endl;

    if (test_forward() != 0) status = -1;
    if (test_backprop() != 0) status = -1;
    if (test_calcualte_loss() != 0) status = -1;
    if (test_one_sample_training() != 0) status = -1;
    if (test_set_optimizer_valid() != 0) status = -1;
    if (test_set_optimizer_invalid() != 0) status = -1;

    if (status == 0) {
        std::cout << "All ANN tests passed successfully!\n";
    } else {
        std::cerr << "Some ANN tests failed.\n";
    }

    std::cout << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << "###############  ANN TESTS DONE... ################" << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << std::endl;

    return status;
}