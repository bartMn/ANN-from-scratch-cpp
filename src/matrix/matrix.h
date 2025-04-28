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

    
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    friend Matrix operator+(const Matrix& a, const Matrix& b);
    friend Matrix operator-(const Matrix& a, const Matrix& b);
    friend Matrix operator*(const Matrix& m, double scalar);
    friend Matrix operator*(double scalar, const Matrix& m);
    //Matrix operator*(const Matrix& a, const Matrix& b);
    friend Matrix operator/(const Matrix& m, double scalar);
    

    private:
        int rows;
        int columns;
        std::vector<float> matrix_vals;
};
    



#endif