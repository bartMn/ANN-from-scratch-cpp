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


Matrix& Matrix::operator+=(const Matrix& other) {
    
    if (rows != other.rows || columns != other.columns) {
        throw std::runtime_error("Matrix dimensions must match for addition.");
    }
    
    for (int i=0; i<rows*columns; i++) matrix_vals[i] += other.matrix_vals[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    
    if (rows != other.rows || columns != other.columns) {
        throw std::runtime_error("Matrix dimensions must match for addition.");
    }
    
    for (int i=0; i<rows*columns; i++) matrix_vals[i] -= other.matrix_vals[i];
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {

    for (int i=0; i<rows*columns; i++) matrix_vals[i] *= scalar;
    return *this;
}


Matrix& Matrix::operator/=(double scalar) {
    if (scalar == 0) {
        throw std::runtime_error("division by 0!");
    }

    for (int i=0; i<rows*columns; i++) matrix_vals[i] /= scalar;
    return *this;
}


Matrix operator+(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::runtime_error("Matrix dimensions must match for addition.");
    }
    Matrix result(a.rows, a.columns);
    for (int i=0; i<a.rows*a.columns; i++) result.matrix_vals[i] = a.matrix_vals[i] + b.matrix_vals[i];
    
    return result;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::runtime_error("Matrix dimensions must match for addition.");
    }
    Matrix result(a.rows, a.columns);
    for (int i=0; i<a.rows*a.columns; i++) result.matrix_vals[i] = a.matrix_vals[i] - b.matrix_vals[i];
    
    return result;
}

Matrix operator*(const Matrix& m, double scalar){
    Matrix result(m.rows, m.columns);
    for (int i=0; i<m.rows*m.columns; i++) result.matrix_vals[i] = m.matrix_vals[i] * scalar;
    return result;
}

Matrix operator*(double scalar, const Matrix& m){
    Matrix result(m.rows, m.columns);
    for (int i=0; i<m.rows*m.columns; i++) result.matrix_vals[i] = m.matrix_vals[i] * scalar;
    return result;
}


Matrix operator/(const Matrix& m, double scalar){
    if (scalar == 0) {
        throw std::runtime_error("division by 0!");
    }
    Matrix result(m.rows, m.columns);
    for (int i=0; i<m.rows*m.columns; i++) result.matrix_vals[i] = m.matrix_vals[i] / scalar;
    return result;
}

Matrix operator*(const Matrix& a, const Matrix& b){
    
    if (a.columns != b.rows) {
        throw std::runtime_error("Invalid matrix dimensions for matrix multiplication.");
    }
    Matrix result(a.rows, b.columns);
    
    for (int a_row=0; a_row < a.rows; a_row++){
        for (int k=0; k< a.columns; k++){
            for (int b_col= 0; b_col < b.columns; b_col++){
                result.matrix_vals[a_row * b.columns + b_col] += a.matrix_vals[a_row * a.columns + k] * b.matrix_vals[k * b.columns + b_col];
            }
        }
    }
    return result;
}

Matrix operator^(const Matrix& a, const Matrix& b){
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::runtime_error("Matrix dimensions must match for element-wise multiplication.");
    }
    
    Matrix result(a.rows, a.columns);
    for (int i=0; i<a.rows*a.columns; i++) result.matrix_vals[i] = a.matrix_vals[i] * b.matrix_vals[i];
    return result;
}