#include <iostream>
#include "matrix/matrix.h"
#include "../tests/matrix/matrix_test.h"
#include "../tests/functions/functions_test.h"
#include "../tests/ann/ann_test.h"

int main()
{
    printf("Hello from Main\n");
    //matrix_test1();
    int status = 0;
    if (run_matrix_tests() != 0) status = -1;
    if (run_functions_tests() != 0) status = -1;
    if (run_ann_tests() != 0) status = -1;

    if (status == 0) {
        std::cout << "All tests passed successfully!\n";
    } else {
        std::cerr << "Some tests failed.\n";
    }

    return 0;
}

