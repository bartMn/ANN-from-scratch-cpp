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

    return 0;
}