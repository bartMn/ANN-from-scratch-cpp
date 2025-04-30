#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

class Functions; ///< Forward declaration of Functions class

/**
 * @class Matrix
 * @brief Represents a 2D matrix and provides basic matrix operations.
 */
class Matrix{
    public:
        /**
         * @brief Constructs a matrix with specified dimensions and initializes values from an array.
         * @param r Number of rows.
         * @param c Number of columns.
         * @param mat Pointer to an array of values to initialize the matrix.
         */
        Matrix(int r, int c, float* mat);

        /**
         * @brief Constructs a matrix with specified dimensions and initializes all values to zero.
         * @param r Number of rows.
         * @param c Number of columns.
         */
        Matrix(int r, int c);

        /**
         * @brief Gets the number of rows in the matrix.
         * @return Number of rows.
         */
        int get_rows_num();

        /**
         * @brief Gets the number of columns in the matrix.
         * @return Number of columns.
         */
        int get_columns_num();
        
        /**
         * @brief Prints the matrix to the console.
         */
        void printMatrix();

        /**
         * @brief Gets the value at a specific position in the matrix.
         * @param row Row index.
         * @param col Column index.
         * @return Value at the specified position.
         */
        float get_val(int row, int col);

        // Operator overloads for matrix operations
        Matrix& operator+=(const Matrix& other); ///< Adds another matrix to this matrix.
        Matrix& operator-=(const Matrix& other); ///< Subtracts another matrix from this matrix.
        Matrix& operator*=(double scalar); ///< Multiplies this matrix by a scalar.
        Matrix& operator/=(double scalar); ///< Divides this matrix by a scalar.
        
        // Matrix multiplication and element-wise multiplication that put results in the current matrix
        void matrixMultiply(const Matrix& a, const Matrix& b); ///< Performs matrix multiplication.
        void elementWiseMultiply(const Matrix& a, const Matrix& b); ///< Performs element-wise multiplication.


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
    
        friend class Functions; ///< Allows the Functions class to access private members of Matrix.

    private:
        int rows; ///< Number of rows in the matrix.
        int columns; ///< Number of columns in the matrix.
        std::vector<float> matrix_vals; ///< Flattened 1D vector storing matrix values.
};
    
#endif