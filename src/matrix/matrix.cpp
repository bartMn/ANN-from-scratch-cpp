#include "matrix.h"
#include <iostream>

void Matrix::test(){
    std::cout << this->rows << std::endl;
}

int Matrix::get_rows_num(){
    return this -> rows;
}
int Matrix::get_columns_num(){
    return this -> columns;
}