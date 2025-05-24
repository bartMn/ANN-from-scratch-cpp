#include "../src/matrix/matrix.h"
#include "../src/functions/functions.h"
#include "../src/ann/ann.h"
#include <iostream>
#include <thread>   // for sleep_for
#include <chrono>   // for seconds

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



int run_ann_tests() {
    int status = 0;

    if (test_forward() != 0) status = -1;
    if (test_backprop() != 0) status = -1;
    if (test_calcualte_loss() != 0) status = -1;
    //std::this_thread::sleep_for(std::chrono::seconds(1));
    if (status == 0) {
        std::cout << "All ANN tests passed successfully!\n";
    } else {
        std::cerr << "Some ANN tests failed.\n";
    }
    return 0;
}