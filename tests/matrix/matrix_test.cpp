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

void matrix_test1()
{
    std::cout << "matrix.cpp imported from src OK\n";
    std::thread t1(func1, 1, "mes1\n");
    std::thread t2(func1, 420, "mes2\n");
    
    t1.join();
    t2.join();
    Matrix test_m(666, 555);
    test_m.test();
    std::cout << "The maxtix has " << test_m.get_rows_num() << " rows and " << test_m.get_columns_num() << " columns" << std::endl;
    //return 0;
}

void test_constructor() {
    Matrix m(3, 5);
    assert(m.get_rows_num() == 3);
    assert(m.get_columns_num() == 5);
    std::cout << "test_constructor passed.\n";
}

void test_get_rows_num() {
    Matrix m(7, 2);
    assert(m.get_rows_num() == 7);
    std::cout << "test_get_rows_num passed.\n";
}

void test_get_columns_num() {
    Matrix m(4, 9);
    assert(m.get_columns_num() == 9);
    std::cout << "test_get_columns_num passed.\n";
}

void test_test_function() {
    Matrix m(2, 3);
    m.test();  // Should just print without crashing
    std::cout << "test_test_function passed (no crash).\n";
}


int run_matrix_tests() {
    test_constructor();
    test_get_rows_num();
    test_get_columns_num();
    test_test_function();

    std::cout << "All tests passed!\n";
    return 0;
}