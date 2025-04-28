#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

class Matrix{
    public:
        Matrix(int r, int c, float* mat);
        Matrix(int r, int c);
        int get_rows_num();
        int get_columns_num();
        void printMatrix();
        float get_val(int row, int col);

    /*
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);
    */

    private:
        int rows;
        int columns;
        std::vector<std::vector<float>> matrix_vals;
};
/*
    
// Vector operators
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& v, double scalar);
Matrix operator*(const Matrix& a, const Matrix& b);
Matrix operator*(double scalar, const Matrix& v);
Matrix operator/(const Matrix& v, double scalar);
Matrix operator-(const Matrix& v);

*/

#endif