#include <iostream>
#include "matrix/matrix.h"
#include "../tests/matrix/matrix_test.h"
#include "../tests/functions/functions_test.h"
#include "../tests/ann/ann_test.h"

int main()
{
    printf("Hello from Main\n");
    //matrix_test1();
    run_matrix_tests();
    run_functions_tests();
    run_ann_tests();
    return 0;
}

