#include "../src/matrix/matrix.h"
#include "../src/functions/functions.h"
#include "../src/ann/ann.h"
#include <iostream>

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
    
    for (int ct = 0; ct < 40; ct++) {
        std::cout << "Training iteration: " << ct << "\n";
        ann.forward(input);
        ann.reset_gradients(); // Reset gradients before each training step
        ann.calcualte_loss(target);
        ann.backprop();
        ann.update_weights(); // Update weights after backpropagation
    }

    std::cout << "test_one_sample_training passed.\n";
    return 0;
}



int run_ann_tests() {
    int status = 0;

    if (test_forward() != 0) status = -1;
    if (test_backprop() != 0) status = -1;
    if (test_calcualte_loss() != 0) status = -1;
    if (test_one_sample_training() != 0) status = -1;

    if (status == 0) {
        std::cout << "All ANN tests passed successfully!\n";
    } else {
        std::cerr << "Some ANN tests failed.\n";
    }
    return 0;
}