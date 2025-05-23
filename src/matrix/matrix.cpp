#include "matrix.h"
#include <iostream>

/**
 * @brief Constructs a matrix with specified dimensions and initializes values from an array.
 * @param r Number of rows.
 * @param c Number of columns.
 * @param mat Pointer to an array of values to initialize the matrix.
 */
 Matrix::Matrix(int r, int c, float* mat) {
    rows = r;
    columns = c;
    matrix_vals.resize(r * c, 0.0f);

    for (int row = 0; row < r; row++) {
        for (int col = 0; col < c; col++) {
            matrix_vals[row * c + col] = *((mat + row * c) + col);
        }
    }
}

/**
 * @brief Constructs a matrix with specified dimensions and initializes all values to zero.
 * @param r Number of rows.
 * @param c Number of columns.
 */
Matrix::Matrix(int r, int c) {
    rows = r;
    columns = c;
    matrix_vals.resize(r * c, 0.0f);
}

/**
 * @brief Prints the matrix to the console, including its dimensions and values.
 */
void Matrix::printMatrix() {
    std::cout << "rows: " << rows << "\tcolumns: " << columns << std::endl;
    for (int r = 0; r < this->rows; r++) {
        for (int c = 0; c < this->columns; c++) {
            printf("%f    ", Matrix::get_val(r, c));
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Returns the number of rows in the matrix.
 * @return The number of rows.
 */
int Matrix::get_rows_num() { 
    return this->rows; 
}

/**
 * @brief Returns the number of columns in the matrix.
 * @return The number of columns.
 */
int Matrix::get_columns_num() { 
    return this->columns; 
}

/**
 * @brief Retrieves the value at a specific position in the matrix.
 * @param row The row index.
 * @param col The column index.
 * @return The value at the specified position.
 */
float Matrix::get_val(int row, int col) { 
    return this->matrix_vals[row * this->columns + col]; 
}

/**
 * @brief Adds another matrix to this matrix element-wise.
 * @param other The matrix to add.
 * @return A reference to the updated matrix.
 * @throws std::runtime_error if the dimensions of the matrices do not match.
 */
Matrix& Matrix::operator+=(const Matrix& other) {
    
    if (rows != other.rows || columns != other.columns) {
        throw std::runtime_error("Matrix dimensions must match for addition.");
    }
    
    #pragma omp parallel for
    for (int i=0; i<rows*columns; i++) matrix_vals[i] += other.matrix_vals[i];
    return *this;
}
/**
 * @brief Subtracts another matrix from this matrix element-wise.
 * @param other The matrix to subtract.
 * @return A reference to the updated matrix.
 * @throws std::runtime_error if the dimensions of the matrices do not match.
 */
 Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || columns != other.columns) {
        throw std::runtime_error("Matrix dimensions must match for subtraction.");
    }

    #pragma omp parallel for
    for (int i = 0; i < rows * columns; i++) {
        matrix_vals[i] -= other.matrix_vals[i];
    }
    return *this;
}

/**
 * @brief Multiplies this matrix by a scalar value.
 * @param scalar The scalar value to multiply by.
 * @return A reference to the updated matrix.
 */
Matrix& Matrix::operator*=(double scalar) {
    #pragma omp parallel for
    for (int i = 0; i < rows * columns; i++) {
        matrix_vals[i] *= scalar;
    }
    return *this;
}

/**
 * @brief Divides this matrix by a scalar value.
 * @param scalar The scalar value to divide by.
 * @return A reference to the updated matrix.
 * @throws std::runtime_error if the scalar value is zero.
 */
Matrix& Matrix::operator/=(double scalar) {
    if (scalar == 0) {
        throw std::runtime_error("division by 0!");
    }

    #pragma omp parallel for
    for (int i = 0; i < rows * columns; i++) {
        matrix_vals[i] /= scalar;
    }
    return *this;
}
/**
 * @brief Adds two matrices element-wise.
 * @param a The first matrix.
 * @param b The second matrix.
 * @return A new matrix containing the element-wise sum of the two matrices.
 * @throws std::runtime_error if the dimensions of the matrices do not match.
 */
 Matrix operator+(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::runtime_error("Matrix dimensions must match for addition.");
    }

    Matrix result(a.rows, a.columns);
    #pragma omp parallel for
    for (int i = 0; i < a.rows * a.columns; i++) {
        result.matrix_vals[i] = a.matrix_vals[i] + b.matrix_vals[i];
    }
    return result;
}

/**
 * @brief Subtracts one matrix from another element-wise.
 * @param a The first matrix.
 * @param b The second matrix.
 * @return A new matrix containing the element-wise difference of the two matrices.
 * @throws std::runtime_error if the dimensions of the matrices do not match.
 */
Matrix operator-(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::runtime_error("Matrix dimensions must match for subtraction.");
    }

    Matrix result(a.rows, a.columns);
    #pragma omp parallel for
    for (int i = 0; i < a.rows * a.columns; i++) {
        result.matrix_vals[i] = a.matrix_vals[i] - b.matrix_vals[i];
    }
    return result;
}

/**
 * @brief Multiplies a matrix by a scalar value.
 * @param m The matrix to multiply.
 * @param scalar The scalar value.
 * @return A new matrix containing the result of the scalar multiplication.
 */
Matrix operator*(const Matrix& m, double scalar) {
    Matrix result(m.rows, m.columns);
    #pragma omp parallel for
    for (int i = 0; i < m.rows * m.columns; i++) {
        result.matrix_vals[i] = m.matrix_vals[i] * scalar;
    }
    return result;
}

/**
 * @brief Multiplies a scalar value by a matrix.
 * @param scalar The scalar value.
 * @param m The matrix to multiply.
 * @return A new matrix containing the result of the scalar multiplication.
 */
Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar; // Reuse the operator*(Matrix, double) implementation.
}

/**
 * @brief Multiplies two matrices using standard matrix multiplication.
 * @param a The first matrix.
 * @param b The second matrix.
 * @return A new matrix containing the result of the matrix multiplication.
 * @throws std::runtime_error if the dimensions of the matrices are incompatible for multiplication.
 */
Matrix operator*(const Matrix& a, const Matrix& b) {
    if (a.columns != b.rows) {
        throw std::runtime_error("Matrix dimensions must match for multiplication.");
    }

    Matrix result(a.rows, b.columns);
    result.matrixMultiply(a, b);
    return result;
}

/**
 * @brief Performs element-wise multiplication of two matrices.
 * @param a The first matrix.
 * @param b The second matrix.
 * @return A new matrix containing the element-wise product of the two matrices.
 * @throws std::runtime_error if the dimensions of the matrices do not match.
 */
Matrix operator^(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::runtime_error("Matrix dimensions must match for element-wise multiplication.");
    }

    Matrix result(a.rows, a.columns);
    #pragma omp parallel for
    for (int i = 0; i < a.rows * a.columns; i++) {
        result.matrix_vals[i] = a.matrix_vals[i] * b.matrix_vals[i];
    }
    return result;
}

/**
 * @brief Divides a matrix by a scalar value.
 * @param m The matrix to divide.
 * @param scalar The scalar value.
 * @return A new matrix containing the result of the scalar division.
 * @throws std::runtime_error if the scalar value is zero.
 */
Matrix operator/(const Matrix& m, double scalar) {
    if (scalar == 0) {
        throw std::runtime_error("Division by zero is not allowed.");
    }

    Matrix result(m.rows, m.columns);
    #pragma omp parallel for
    for (int i = 0; i < m.rows * m.columns; i++) {
        result.matrix_vals[i] = m.matrix_vals[i] / scalar;
    }
    return result;
}

/**
 * @brief Adds a scalar value to each element of a matrix.
 * @param m The matrix.
 * @param scalar The scalar value to add.
 * @return A new matrix containing the result of the addition.
 */
Matrix operator+(const Matrix& m, double scalar) {
    Matrix result(m.rows, m.columns);
    #pragma omp parallel for
    for (int i = 0; i < m.rows * m.columns; i++) {
        result.matrix_vals[i] = m.matrix_vals[i] + scalar;
    }
    return result;
}

/**
 * @brief Adds a scalar value to each element of a matrix.
 * @param scalar The scalar value to add.
 * @param m The matrix.
 * @return A new matrix containing the result of the addition.
 */
Matrix operator+(double scalar, const Matrix& m) {
    return m + scalar; // Reuse the operator+(Matrix, double) implementation.
}

/**
 * @brief Subtracts a scalar value from each element of a matrix.
 * @param m The matrix.
 * @param scalar The scalar value to subtract.
 * @return A new matrix containing the result of the subtraction.
 */
Matrix operator-(const Matrix& m, double scalar) {
    Matrix result(m.rows, m.columns);
    #pragma omp parallel for
    for (int i = 0; i < m.rows * m.columns; i++) {
        result.matrix_vals[i] = m.matrix_vals[i] - scalar;
    }
    return result;
}

/**
 * @brief Subtracts each element of a matrix from a scalar value.
 * @param scalar The scalar value.
 * @param m The matrix.
 * @return A new matrix containing the result of the subtraction.
 */
Matrix operator-(double scalar, const Matrix& m) {
    Matrix result(m.rows, m.columns);
    #pragma omp parallel for
    for (int i = 0; i < m.rows * m.columns; i++) {
        result.matrix_vals[i] = scalar - m.matrix_vals[i];
    }
    return result;
}

/**
 * @brief Performs matrix multiplication and stores the result in the current matrix.
 * @param a The first matrix.
 * @param b The second matrix.
 * @throws std::runtime_error if the dimensions of the matrices are incompatible for multiplication.
 */
 void Matrix::matrixMultiply(const Matrix& a, const Matrix& b) {
    if (a.columns != b.rows) {
        throw std::runtime_error("Matrix dimensions must match for multiplication.");
    }

    if (this->rows != a.rows || this->columns != b.columns) {
        throw std::runtime_error("Result matrix dimensions do not match.");
    }

    std::fill(this->matrix_vals.begin(), this->matrix_vals.end(), 0.0f);

    #pragma omp parallel for
    for (int a_row = 0; a_row < a.rows; a_row++) {
        for (int b_col = 0; b_col < b.columns; b_col++) {
            for (int k = 0; k < a.columns; k++) {
                this->matrix_vals[a_row * b.columns + b_col] += 
                    a.matrix_vals[a_row * a.columns + k] * b.matrix_vals[k * b.columns + b_col];
            }
        }
    }
}

/**
 * @brief Performs element-wise multiplication and stores the result in the current matrix.
 * @param a The first matrix.
 * @param b The second matrix.
 * @throws std::runtime_error if the dimensions of the matrices do not match.
 */
void Matrix::elementWiseMultiply(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::runtime_error("Matrix dimensions must match for element-wise multiplication.");
    }

    if (this->rows != a.rows || this->columns != a.columns) {
        throw std::runtime_error("Result matrix dimensions do not match.");
    }

    #pragma omp parallel for
    for (int i = 0; i < a.rows * a.columns; i++) {
        this->matrix_vals[i] = a.matrix_vals[i] * b.matrix_vals[i];
    }
}

/**
 * @brief Sets the values of this matrix from another matrix.
 * @param m The matrix to copy values from.
 */
void Matrix::setValsFormMatrix(const Matrix& m) {
    if (this->rows != m.rows || this->columns != m.columns) {
        throw std::runtime_error("Matrix dimensions must match for assignment.");
    }

    #pragma omp parallel for
    for (int i = 0; i < m.rows * m.columns; i++) {
        this->matrix_vals[i] = m.matrix_vals[i];
    }
}

/**
 * @brief Initializes the matrix with random values.
 */
void Matrix::randomInit() {
    for (int i = 0; i < this->rows * this->columns; i++) {
        this->matrix_vals[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}
