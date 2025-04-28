#include <iostream>
#include <thread>
#include "../src/matrix/matrix.h"
#include "matrix_test.h"
#include <cassert>


int func1(int c, const std::string& m)
{
    std::cout << "from func call" <<c <<" "<< m << std::endl;
    return 0;
}

int matrix_test1()
{
    std::cout << "matrix.cpp imported from src OK\n";
    std::thread t1(func1, 1, "mes1\n");
    std::thread t2(func1, 420, "mes2\n");
    
    t1.join();
    t2.join();
    Matrix test_m(666, 555);
    std::cout << "The maxtix has " << test_m.get_rows_num() << " rows and " << test_m.get_columns_num() << " columns" << std::endl;
    return 0;
}

int test_constructor() {
    int status = 0;
    Matrix m(3, 5);
    if (m.get_rows_num() != 3) status = -1;
    if (m.get_columns_num() != 5) status = -1;

    if (status == -1){
        std::cout << "test_constructor Failed.\n";
        return -1;
    }

    float arr[2][3] = {{1.0,2.0,3.0}, {4.0,5.0,6.0}};
    Matrix m1(2, 3, *arr);

    // Check dimensions
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

int test_print_matrix() {
    float arr[2][3] = {{1.0,2.0,3.0}, {4.0,5.0,6.0}};
    Matrix m(2, 3, *arr);
    m.printMatrix();
    std::cout << "test_constructor passed.\n";
    return 0;
}

int test_get_rows_num() {
    Matrix m(7, 2);
    if (m.get_rows_num() != 7){
        std::cout << "test_get_rows_num Falied.\n";
        return -1;
    }
    std::cout << "test_get_rows_num passed.\n";
    return 0;
}

int test_get_columns_num() {
    Matrix m(4, 9);
    if (m.get_columns_num() != 9){
        std::cout << "test_get_columns_num Failed.\n";
        return -1;
    }
    std::cout << "test_get_columns_num passed.\n";
    return 0;
}


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

    if (status == 0) {
        std::cout << "All matrix tests passed successfully!\n";
    } else {
        std::cerr << "Some matrix tests failed.\n";
    }

    return 0;
}