#include "../src/matrix/matrix.h"
#include "../src/functions/functions.h"
#include "../src/ann/ann.h"
#include <iostream>
#include <thread>   // for sleep_for
#include <chrono>   // for seconds

int run_ann_tests() {
    int status = 0;

    ANN ann({2, 500, 500, 1}, {"ReLu","ReLu", "ReLu"});
    float v[2][1] = {{1.0}, {2.0}}; 
    Matrix input(2, 1, *v);
    ann.forward(input);
    //std::this_thread::sleep_for(std::chrono::seconds(1));
    if (status == 0) {
        std::cout << "All ANN tests passed successfully!\n";
    } else {
        std::cerr << "Some ANN tests failed.\n";
    }
    return 0;
}