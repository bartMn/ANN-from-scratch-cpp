#include "matrix.h"
#include <iostream>

Matrix::Matrix(int r, int c, float* mat){
    rows = r;
    columns = c;
    matrix_vals.resize(r*c, 0.0f);

    for (int row=0; row < r; row++){
        for (int col=0; col< c; col++){
            matrix_vals[row * c + col] = *((mat + row * c) + col);
        }
    }
    //matrix_vals[0][0] = 0.0;
}

Matrix::Matrix(int r, int c){
    rows = r;
    columns = c;
    matrix_vals.resize(r*c, 0.0f);
}


void Matrix::printMatrix(){
    std::cout << "rows: " << rows << "\tcolumns: " << columns << std::endl;
    for (int r=0; r < this->rows; r++){
        for (int c=0;c < this->columns; c++){
            printf("%f    ", Matrix::get_val(r, c));
        }
        std::cout << std::endl;
    }
}

int Matrix::get_rows_num(){ return this -> rows; } 
int Matrix::get_columns_num(){ return this -> columns; }
float Matrix::get_val(int row, int col) {return this-> matrix_vals[row * this->columns + col]; }