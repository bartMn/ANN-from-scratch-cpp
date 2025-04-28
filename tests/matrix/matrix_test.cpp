#include <iostream>
#include <thread>
#include "../src/matrix/matrix.h"
#include "matrix_test.h"


int func1(int c, const std::string& m)
{
    std::cout << "from func call" <<c <<" "<< m << std::endl;
    return 0;
}

void matrix_test1()
{
    std::cout << "matrix.cpp imported from src OK\n";
    std::thread t1(func1, 1, "mes1\n");
    std::thread t2(func1, 420, "hehehe\n");
    
    t1.join();
    t2.join();
    Matrix test_m(666, 555);
    test_m.test();
    //return 0;
}