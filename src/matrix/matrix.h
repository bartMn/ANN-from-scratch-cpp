#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

class Functions; ///< Forward declaration of Functions class

/**
 * @class Matrix
 * @brief Represents a 2D matrix and provides basic matrix operations.
 */
class Matrix {
    public:
        Matrix(int r, int c, float* mat); ///< Constructs a matrix with specified dimensions and initializes values from an array.
        Matrix(int r, int c); ///< Constructs a matrix with specified dimensions and initializes all values to zero.
        int get_rows_num(); ///< Gets the number of rows in the matrix.
        int get_columns_num(); ///< Gets the number of columns in the matrix.
        void printMatrix(); ///< Prints the matrix to the console.
        float get_val(int row, int col); ///< Gets the value at a specific position in the matrix.
        void set_val(int row, int col, float val); ///< Sets the value at a specific position in the matrix.
        void randomInit(); ///< Initializes the matrix with random values.
        void randomHeNormalInit(); ///< Initializes the matrix with random values.
        void randomHeUniformInit(); ///< Initializes the matrix with random values.
        
        void resetWithVal(float val);
        // Operator overloads for matrix operations
        Matrix& operator+=(const Matrix& other); ///< Adds another matrix to this matrix.
        Matrix& operator-=(const Matrix& other); ///< Subtracts another matrix from this matrix.
        Matrix& operator*=(double scalar); ///< Multiplies this matrix by a scalar.
        Matrix& operator/=(double scalar); ///< Divides this matrix by a scalar.

        void matrixMultiply(const Matrix& a, const Matrix& b); ///< Performs matrix multiplication and stores the result in the current matrix.
        void elementWiseMultiply(const Matrix& a, const Matrix& b); ///< Performs element-wise multiplication and stores the result in the current matrix.

        void setValsFormMatrix(const Matrix& m); ///< Sets the values of this matrix from another matrix.

        // Friend functions for operator overloads
        friend Matrix operator+(const Matrix& a, const Matrix& b); ///< Adds two matrices.
        friend Matrix operator-(const Matrix& a, const Matrix& b); ///< Subtracts two matrices.
        friend Matrix operator*(const Matrix& m, double scalar); ///< Multiplies a matrix by a scalar.
        friend Matrix operator*(double scalar, const Matrix& m); ///< Multiplies a scalar by a matrix.
        friend Matrix operator*(const Matrix& a, const Matrix& b); ///< Multiplies two matrices.
        friend Matrix operator^(const Matrix& a, const Matrix& b); ///< Performs element-wise multiplication.
        friend Matrix operator/(const Matrix& m, double scalar); ///< Divides a matrix by a scalar.
        friend Matrix operator+(const Matrix& m, double scalar); ///< Adds a scalar to a matrix.
        friend Matrix operator+(double scalar, const Matrix& m); ///< Adds a scalar to a matrix.
        friend Matrix operator-(const Matrix& m, double scalar); ///< Subtracts a scalar from a matrix.
        friend Matrix operator-(double scalar, const Matrix& m); ///< Subtracts a matrix from a scalar.

        friend Matrix transpose(const Matrix& m);
        friend class Functions; ///< Allows the Functions class to access private members of Matrix.

    private:
        int rows; ///< Number of rows in the matrix.
        int columns; ///< Number of columns in the matrix.
        std::vector<float> matrix_vals; ///< Flattened 1D vector storing matrix values.
};

#endif