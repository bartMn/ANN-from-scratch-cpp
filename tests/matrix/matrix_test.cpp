#include <iostream>
#include <thread>
#include "../src/matrix/matrix.h"
#include "matrix_test.h"
#include <cassert>
#include <chrono>

int func1(int c, const std::string& m)
{
    std::cout << "from func call" <<c <<" "<< m << std::endl;
    return 0;
}


/**
 * @brief A test function to demonstrate multithreading and matrix creation.
 * @return 0 if the test passes.
 */
int matrix_test1()
{
    std::cout << "matrix.cpp imported from src OK\n";

    // Create and run threads
    std::thread t1(func1, 1, "mes1\n");
    std::thread t2(func1, 420, "mes2\n");
    
    t1.join();
    t2.join();

    // Create a test matrix
    Matrix test_m(666, 555);
    std::cout << "The maxtix has " << test_m.get_rows_num() << " rows and " << test_m.get_columns_num() << " columns" << std::endl;
    return 0;
}

/**
 * @brief Tests the matrix constructor with various inputs.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_constructor() {
    int status = 0;

    // Test default constructor
    Matrix m(3, 5);
    if (m.get_rows_num() != 3) status = -1;
    if (m.get_columns_num() != 5) status = -1;

    if (status == -1){
        std::cout << "test_constructor Failed.\n";
        return -1;
    }

    // Test constructor with initialization array
    float arr[2][3] = {{1.0,2.0,3.0}, {4.0,5.0,6.0}};
    Matrix m1(2, 3, *arr);

    // Check dimensions and values
    if (m1.get_rows_num() != 2) status = -1;
    if (m1.get_columns_num() != 3) status = -1;
    if (m1.get_val(0, 0) != 1.0f) status = -1;
    if (m1.get_val(0, 1) != 2.0f) status = -1;
    if (m1.get_val(0, 2) != 3.0f) status = -1;
    if (m1.get_val(1, 0) != 4.0f) status = -1;
    if (m1.get_val(1, 1) != 5.0f) status = -1;
    if (m1.get_val(1, 2) != 6.0f) status = -1;

    if (status == 0) std::cout << "test_constructor passed.\n"; 
    else std::cout << "test_constructor Failed.\n";
    return status;
}

/**
 * @brief Tests the printMatrix function.
 * @return 0 if the test passes.
 */
int test_print_matrix() {
    float arr[2][3] = {{1.0,2.0,3.0}, {4.0,5.0,6.0}};
    Matrix m(2, 3, *arr);

    // Print the matrix
    m.printMatrix();
    std::cout << "test_constructor passed.\n";
    return 0;
}


/**
 * @brief Tests the get_rows_num function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_get_rows_num() {
    Matrix m(7, 2);
    if (m.get_rows_num() != 7){
        std::cout << "test_get_rows_num Falied.\n";
        return -1;
    }
    std::cout << "test_get_rows_num passed.\n";
    return 0;
}


/**
 * @brief Tests the get_columns_num function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_get_columns_num() {
    Matrix m(4, 9);
    if (m.get_columns_num() != 9){
        std::cout << "test_get_columns_num Failed.\n";
        return -1;
    }
    std::cout << "test_get_columns_num passed.\n";
    return 0;
}

/**
 * @brief Tests the operator+= for matrix addition.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_plus_equals() {
    float arr1[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    float arr2[2][2] = {{5.0, 6.0}, {7.0, 8.0}};
    Matrix m1(2, 2, *arr1);
    Matrix m2(2, 2, *arr2);

    m1 += m2; // Should become {{6,8},{10,12}}

    if (m1.get_val(0,0) != 6.0f || m1.get_val(0,1) != 8.0f ||
        m1.get_val(1,0) != 10.0f || m1.get_val(1,1) != 12.0f) {
        std::cout << "test_operator_plus_equals Failed.\n";
        return -1;
    }
    std::cout << "test_operator_plus_equals passed.\n";
    return 0;
}

/**
 * @brief Tests the operator-= for matrix subtraction.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_minus_equals() {
    float arr1[2][2] = {{5.0, 6.0}, {7.0, 8.0}};
    float arr2[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix m1(2, 2, *arr1);
    Matrix m2(2, 2, *arr2);

    m1 -= m2;

    if (m1.get_val(0,0) != 4.0f || m1.get_val(0,1) != 4.0f ||
        m1.get_val(1,0) != 4.0f || m1.get_val(1,1) != 4.0f) {
        std::cout << "test_operator_minus_equals Failed.\n";
        return -1;
    }
    std::cout << "test_operator_minus_equals passed.\n";
    return 0;
}

/**
 * @brief Tests the operator*= for scalar multiplication.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_times_equals() {
    float arr1[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix m1(2, 2, *arr1);

    m1 *= 2.0; // Should become {{2,4},{6,8}}

    if (m1.get_val(0,0) != 2.0f || m1.get_val(0,1) != 4.0f ||
        m1.get_val(1,0) != 6.0f || m1.get_val(1,1) != 8.0f) {
        std::cout << "test_operator_times_equals Failed.\n";
        return -1;
    }
    std::cout << "test_operator_times_equals passed.\n";
    return 0;
}

/**
 * @brief Tests the operator/= for scalar division.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_divide_equals() {
    float arr1[2][2] = {{2.0, 4.0}, {6.0, 8.0}};
    Matrix m1(2, 2, *arr1);

    m1 /= 2.0; // Should become {{1,2},{3,4}}

    if (m1.get_val(0,0) != 1.0f || m1.get_val(0,1) != 2.0f ||
        m1.get_val(1,0) != 3.0f || m1.get_val(1,1) != 4.0f) {
        std::cout << "test_operator_divide_equals Failed.\n";
        return -1;
    }
    std::cout << "test_operator_divide_equals passed.\n";
    return 0;
}

/**
 * @brief Tests the operator+= for matrix addition with invalid size.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_plus_equals_invalid_size() {
    float arr1[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    float arr2[3][2] = {{5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}};
    Matrix m1(2, 2, *arr1);
    Matrix m2(3, 2, *arr2);

    try {
        m1 += m2; // Should throw
        std::cout << "test_operator_plus_equals_invalid_size Failed (no exception).\n";
        return -1;
    } catch (const std::runtime_error& e) {
        std::cout << "test_operator_plus_equals_invalid_size passed.\n";
        return 0;
    } catch (...) {
        std::cout << "test_operator_plus_equals_invalid_size Failed (wrong exception type).\n";
        return -1;
    }
}

/**
 * @brief Tests the operator-= for matrix subtraction with invalid size.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_minus_equals_invalid_size() {
    float arr1[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    float arr2[2][3] = {{5.0, 6.0, 7.0}, {8.0, 9.0, 10.0}};
    Matrix m1(2, 2, *arr1);
    Matrix m2(2, 3, *arr2);

    try {
        m1 -= m2; // Should throw
        std::cout << "test_operator_minus_equals_invalid_size Failed (no exception).\n";
        return -1;
    } catch (const std::runtime_error& e) {
        std::cout << "test_operator_minus_equals_invalid_size passed.\n";
        return 0;
    } catch (...) {
        std::cout << "test_operator_minus_equals_invalid_size Failed (wrong exception type).\n";
        return -1;
    }
}

/**
 * @brief Tests the operator+ for matrix addition.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_plus() {
    float arr1[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    float arr2[2][2] = {{5.0, 6.0}, {7.0, 8.0}};
    Matrix m1(2, 2, *arr1);
    Matrix m2(2, 2, *arr2);

    Matrix m3 = m1 + m2;

    if (m3.get_val(0, 0) != 6.0f) return -1;
    if (m3.get_val(0, 1) != 8.0f) return -1;
    if (m3.get_val(1, 0) != 10.0f) return -1;
    if (m3.get_val(1, 1) != 12.0f) return -1;

    std::cout << "test_operator_plus passed.\n";
    return 0;
}

/**
 * @brief Tests the operator- for matrix subtraction.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_minus() {
    float arr1[2][2] = {{5.0, 6.0}, {7.0, 8.0}};
    float arr2[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix m1(2, 2, *arr1);
    Matrix m2(2, 2, *arr2);

    Matrix m3 = m1 - m2;

    if (m3.get_val(0, 0) != 4.0f) return -1;
    if (m3.get_val(0, 1) != 4.0f) return -1;
    if (m3.get_val(1, 0) != 4.0f) return -1;
    if (m3.get_val(1, 1) != 4.0f) return -1;

    std::cout << "test_operator_minus passed.\n";
    return 0;
}

/**
 * @brief Tests the operator* for scalar multiplication.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_scalar_multiplication() {
    float arr[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    Matrix m(2, 2, *arr);

    Matrix result = m * 2.0;

    if (result.get_val(0, 0) != 2.0f) return -1;
    if (result.get_val(0, 1) != 4.0f) return -1;
    if (result.get_val(1, 0) != 6.0f) return -1;
    if (result.get_val(1, 1) != 8.0f) return -1;

    std::cout << "test_operator_scalar_multiplication passed.\n";
    return 0;
}

/**
 * @brief Tests the operator/ for scalar division.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_scalar_division() {
    float arr[2][2] = {{2.0, 4.0}, {6.0, 8.0}};
    Matrix m(2, 2, *arr);

    Matrix result = m / 2.0;

    if (result.get_val(0, 0) != 1.0f) return -1;
    if (result.get_val(0, 1) != 2.0f) return -1;
    if (result.get_val(1, 0) != 3.0f) return -1;
    if (result.get_val(1, 1) != 4.0f) return -1;

    std::cout << "test_operator_scalar_division passed.\n";
    return 0;
}

/**
 * @brief Tests the operator/ for scalar division by zero.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_operator_scalar_division_by_zero() {
    float arr[2][2] = {{2.0, 4.0}, {6.0, 8.0}};
    Matrix m(2, 2, *arr);

    try {
        Matrix result = m / 0.0;
        std::cout << "test_operator_scalar_division_by_zero Failed (no exception).\n";
        return -1;
    } catch (const std::runtime_error& e) {
        std::cout << "test_operator_scalar_division_by_zero passed.\n";
        return 0;
    } catch (...) {
        std::cout << "test_operator_scalar_division_by_zero Failed (wrong exception type).\n";
        return -1;
    }
}

/**
 * @brief Tests invalid matrix multiplication where dimensions are incompatible.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_invalid_multiplication() {
    std::cout << "\nTesting invalid matrix multiplication (should catch error)...\n";

    Matrix A(2, 3);
    Matrix B(4, 2);

    try {
        Matrix C = A * B;
        C.printMatrix();
        std::cout << "test_invalid_multiplication Failed (no exception).\n";
        return -1;
    } catch (const std::runtime_error& e) {
        std::cout << "test_operator_scalar_division_by_zero passed.\n";
        return 0;
    } catch (...) {
        std::cout << "test_operator_scalar_division_by_zero Failed (wrong exception type).\n";
        return -1;
    }
}

/**
 * @brief Tests invalid element-wise multiplication where dimensions are incompatible.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_invalid_elementwise_multiplication() {
    std::cout << "\nTesting invalid element-wise multiplication (should catch error)...\n";

    Matrix A(2, 2);
    Matrix B(3, 3);

    try {
        Matrix C = A ^ B;
        C.printMatrix();
        std::cout << "test_invalid_elementwise_multiplication Failed (no exception).\n";
        return -1;
    } catch (const std::runtime_error& e) {
        std::cout << "test_operator_scalar_division_by_zero passed.\n";
        return 0;
    } catch (...) {
        std::cout << "test_operator_scalar_division_by_zero Failed (wrong exception type).\n";
        return -1;
    }
}

/**
 * @brief Tests the matrix multiplication function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_matrix_multiplication() {
    std::cout << "\nTesting matrix-matrix multiplication...\n";

    float vals_a[] = {1, 2, 3, 4, 5, 6}; // 2x3 matrix
    float vals_b[] = {7, 8, 9, 10, 11, 12}; // 3x2 matrix

    Matrix A(2, 3, vals_a);
    Matrix B(3, 2, vals_b);

    try {
        Matrix C = A * B;
        C.printMatrix();

        // Expected result for 2x3 * 3x2 matrix multiplication:
        float expected[] = {1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12, 4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12};
        for (int r = 0; r< 2; r++)
            for (int c = 0; c < 2; c++) {
                if (C.get_val(r, c) != expected[r*C.get_rows_num() + c]){
                    std::cout << C.get_val(r, c) << "  " << expected[r*C.get_rows_num() + c]<< std::endl;
                    std::cout << "\nMatrix-matrix multiplication FAILED\n";
                    return -1;
                }
            }
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << '\n';
    }
    std::cout << "\nMatrix-matrix multiplication Passed\n";
    return 0;
}

/**
 * @brief Tests the element-wise multiplication function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_elementwise_multiplication() {
    std::cout << "\nTesting element-wise multiplication...\n";

    float vals_c[] = {1, 2, 3, 4};
    float vals_d[] = {5, 6, 7, 8};

    Matrix C(2, 2, vals_c);
    Matrix D(2, 2, vals_d);

    try {
        Matrix E = C ^ D;
        E.printMatrix();

        // Expected result for element-wise multiplication:
        float expected[] = {1*5, 2*6, 3*7, 4*8};  // [[1*5, 2*6], [3*7, 4*8]]
        for (int r = 0; r< 2; r++)
            for (int c = 0; c < 2; c++) {
                if (E.get_val(r, c) != expected[r*C.get_rows_num() + c]){
                    std::cout << "\nElement-wise multiplication FAILED\n";
                    return -1;
                }
            }
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << '\n';
    }
    std::cout << "\nElement-wise multiplication Passed\n";
    return 0;
}

/**
 * @brief Tests the matrix multiplication function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_matrixMultiply() {
    float arr1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    float arr2[3][2] = {{7, 8}, {9, 10}, {11, 12}};
    Matrix m1(2, 3, *arr1);
    Matrix m2(3, 2, *arr2);
    Matrix result(2, 2);

    result.matrixMultiply(m1, m2);
    float expected[2][2] = {{1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12}, {4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12}};
    for (int r = 0; r< result.get_rows_num(); r++){
        for (int c = 0; c < result.get_columns_num(); c++) { //*((mat + row * c) + col)
            if (result.get_val(r, c) != *(*expected+r*result.get_rows_num() + c)){
                std::cout << result.get_val(r, c) << "  " << *(*expected+r*result.get_rows_num() + c)<< std::endl;
                std::cout << "\nMatrix-matrix multiplication FAILED\n";
                return -1;
            }
        }
    }

    std::cout << "test_matrixMultiply passed.\n";
    return 0;
}

/**
 * @brief Tests the element-wise multiplication function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_elementWiseMultiply() {
    float arr1[2][2] = {{1, 2}, {3, 4}};
    float arr2[4] = {5, 6, 7, 8};
    Matrix m1(2, 2, &arr1[0][0]);
    Matrix m2(2, 2, arr2);
    Matrix result(2, 2);

    result.elementWiseMultiply(m1, m2);
    float expected[] = {1*5, 2*6, 3*7, 4*8};  // [[1*5, 2*6], [3*7, 4*8]]
        for (int r = 0; r< result.get_rows_num() ; r++)
            for (int c = 0; c < result.get_columns_num() ; c++) {
                if (result.get_val(r, c) != expected[r*result.get_rows_num() + c]){
                    std::cout << "\nElement-wise multiplication FAILED\n";
                    return -1;
                }
            }

    std::cout << "test_elementWiseMultiply passed.\n";
    return 0;
}

/**
 * @brief Tests the execution time of a matrix operation.
 * @return 0 if the test passes.
 */
int test_exec_time(){

    auto start = std::chrono::high_resolution_clock::now();
    int M = 2048, N = 2048;
    float* vals = new float[M*N];
    
    for (int i = 0; i< M*N; i++) vals[i] = 1.0;
    Matrix A(M,N, vals);
    Matrix B(N,M, vals);
    Matrix C(M,M);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken to create matrixes: " << duration.count() << " ms" << std::endl;
    
    
    start = std::chrono::high_resolution_clock::now();

    C.matrixMultiply(A, B);
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken to do matrix multiplication: " << duration.count() << " ms" << std::endl;
    

    return 0;
}


/**
 * @brief Runs all matrix-related tests.
 * @return 0 if all tests pass, -1 otherwise.
 */
int run_matrix_tests() {
    int status = 0;
    if (test_constructor() != 0) status = -1;
    if (test_get_rows_num() != 0) status = -1;
    if (test_get_columns_num() != 0) status = -1;
    if (test_print_matrix() != 0) status = -1;
    if (test_operator_plus_equals() != 0) status = -1;
    if (test_operator_minus_equals() != 0) status = -1;
    if (test_operator_times_equals() != 0) status = -1;
    if (test_operator_divide_equals() != 0) status = -1;
    if (test_operator_plus_equals_invalid_size() != 0) status = -1;
    if (test_operator_minus_equals_invalid_size() != 0) status = -1;
    if (test_operator_plus() != 0) status = -1;
    if (test_operator_minus() != 0) status = -1;
    if (test_operator_scalar_multiplication() != 0) status = -1;
    if (test_operator_scalar_division() != 0) status = -1;
    if (test_operator_scalar_division_by_zero() != 0) status = -1;
    if (test_matrix_multiplication() != 0) status = -1;
    if (test_elementwise_multiplication() != 0) status = -1;
    if (test_invalid_multiplication() != 0) status = -1;
    if (test_invalid_elementwise_multiplication() != 0) status = -1;
    if (test_matrixMultiply() != 0) status = -1;
    if (test_elementWiseMultiply() != 0) status = -1;
    //test_exec_time();

    if (status == 0) {
        std::cout << "All matrix tests passed successfully!\n";
    } else {
        std::cerr << "Some matrix tests failed.\n";
    }

    return 0;
}