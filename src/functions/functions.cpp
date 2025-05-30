#include "functions.h"
#include <cmath>
#include <iostream>

/**
 * @brief Applies the ReLU activation function element-wise.
 * @param m The matrix to apply ReLU on.
 */
void Functions::ReLu(Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) m.matrix_vals[i] = std::max(0.0f, m.matrix_vals[i]);
}

/**
 * @brief Applies the sigmoid activation function element-wise.
 * @param m The matrix to apply sigmoid on.
 */
void Functions::sigmoid(Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) m.matrix_vals[i] = 1.0 / (1.0 + std::exp(-m.matrix_vals[i]));
}

/**
 * @brief Applies the softmax function to the matrix.
 * @param m The matrix to apply softmax on.
 */
void Functions::softmax(Matrix& m){
    float sum_of_exp = 0.0f;
    for (int i = 0; i < m.columns*m.rows; i++){
        m.matrix_vals[i] = std::exp(m.matrix_vals[i]);
        sum_of_exp += m.matrix_vals[i];
    } 
    m /= sum_of_exp;
}


/**
 * @brief Applies the hyperbolic tangent function element-wise.
 * @param m The matrix to apply tanh on.
 */
void Functions::Tanh(Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) m.matrix_vals[i] = std::tanh(m.matrix_vals[i]);
}

/**
 * @brief Applies the linear activation function element-wise (identity function).
 * @param m The matrix to apply the linear function on.
 */
 void Functions::linear(Matrix& m) {
    // Linear activation is essentially the identity function, so no changes are needed.
    // This function is included for consistency and clarity.
}




/**
 * @brief Computes the difference between predictions and ground truth.
 * @param m_diff The matrix to store the difference.
 * @param predictions The matrix of predicted values.
 * @param y The matrix of ground truth values.
 * @throws std::runtime_error if the dimensions of predictions and ground truth do not match.
 */
void Functions::diff(Matrix& m_diff, Matrix& predictions, Matrix& y){
    if (predictions.rows != y.rows || predictions.columns != y.columns) {
        throw std::runtime_error("Matrix dimensions must match for diff calculation.");
    }

    for (int i = 0; i < predictions.rows * predictions.columns; i++) {
        m_diff.matrix_vals[i] = predictions.matrix_vals[i] - y.matrix_vals[i];
    }
}


/**
 * @brief Computes the Mean Squared Error (MSE) between predictions and ground truth.
 * @param predictions The matrix of predicted values.
 * @param y The matrix of ground truth values.
 * @return The computed MSE value.
 * @throws std::runtime_error if the dimensions of predictions and ground truth do not match.
 */
float Functions::MSE(Matrix& predictions, Matrix& y) {
    if (predictions.rows != y.rows || predictions.columns != y.columns) {
        throw std::runtime_error("Matrix dimensions must match for MSE calculation.");
    }

    float mse = 0.0f;
    for (int i = 0; i < predictions.rows * predictions.columns; i++) {
        float diff = predictions.matrix_vals[i] - y.matrix_vals[i];
        mse += diff * diff;
    }
    mse /= (predictions.rows * predictions.columns);
    return mse;
}

/**
 * @brief Computes the Mean Squared Error (MSE) between predictions and ground truth.
 * @param predictions The matrix of errors.
 * @return The computed MSE value.
 */
 float Functions::MSE(Matrix& m_diff){
    float mse = 0.0f;
    for (int i = 0; i < m_diff.columns*m_diff.rows; i++) {
        mse += m_diff.matrix_vals[i] * m_diff.matrix_vals[i];
    }
    mse /= (m_diff.rows * m_diff.columns);
    return mse;
}

/**
 * @brief Computes the cross-entropy loss between predictions and ground truth.
 * @param predictions The matrix of predicted probabilities.
 * @param y The matrix of ground truth values.
 * @return The computed cross-entropy loss value.
 * @throws std::runtime_error if the dimensions of predictions and ground truth do not match.
 */
float Functions::Cross_Entropy(Matrix& predictions, Matrix& y) {
    if (predictions.rows != y.rows || predictions.columns != y.columns) {
        throw std::runtime_error("Matrix dimensions must match for Cross Entropy calculation.");
    }

    float cross_entropy = 0.0f;
    for (int i = 0; i < predictions.rows * predictions.columns; i++) {
        if (y.matrix_vals[i] > 0) { 
            cross_entropy -= y.matrix_vals[i] * std::log(predictions.matrix_vals[i] + 1e-9f); // Add small epsilon to avoid log(0)
        }
    }
    return cross_entropy;
}

/**
 * @brief Computes the derivative of the ReLU function.
 * @param m_derivatives The matrix to store the derivatives.
 * @param m The matrix of input values.
 */
void Functions::ReLu_derivative(Matrix& m_derivatives, Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) {
        if (m.matrix_vals[i] > 0) {
            m_derivatives.matrix_vals[i] = 1.0f;
        } else {
            m_derivatives.matrix_vals[i] = 0.0f;
        }
    }
}

/**
 * @brief Computes the derivative of the sigmoid function.
 * @param m_derivatives The matrix to store the derivatives.
 * @param m The matrix of input values.
 */
void Functions::sigmoid_derivative(Matrix& m_derivatives, Matrix& m){
    m_derivatives.elementWiseMultiply(m, 1.0 - m);
}


/**
 * @brief Computes the derivative of the linear function (always 1).
 * @param m_derivatives The matrix to store the derivatives.
 * @param m The matrix of input values.
 */
 void Functions::linear_derivative(Matrix& m_derivatives, Matrix& m) {
    for (int i = 0; i < m.columns * m.rows; i++) {
        m_derivatives.matrix_vals[i] = 1.0f; // Derivative of linear function is 1.
    }
}


/**
 * @brief Computes the derivative of the MSE loss.
 * @param m_derivatives The matrix to store the derivatives.
 * @param m_diff The matrix of differences between predictions and ground truth.
 */
void Functions::MSE_derivative(Matrix& m_derivatives, Matrix& m_diff){
    int N = m_diff.rows * m_diff.columns;
    for (int i = 0; i < m_diff.columns*m_diff.rows; i++) {
        m_derivatives.matrix_vals[i] = (2.0f * m_diff.matrix_vals[i]) / N;
    }
}

/**
 * @brief Computes the derivative of the cross-entropy loss.
 * @param m_derivatives The matrix to store the derivatives.
 * @param m_diff The matrix of differences between predictions and ground truth.
 */
void Functions::Cross_Entropy_derivative(Matrix& m_derivatives, Matrix& m_diff){
    for (int i = 0; i < m_diff.columns*m_diff.rows; i++) {
        m_derivatives.matrix_vals[i] = -m_diff.matrix_vals[i];
    }
}

/**
 * @brief Computes the derivative of the hyperbolic tangent (tanh) function.
 * @param m_derivatives The matrix to store the derivatives.
 * @param m The matrix of input values.
 */
 void Functions::Tanh_derivative(Matrix& m_derivatives, Matrix& m) {
    for (int i = 0; i < m.columns * m.rows; i++) {
        float tanh_val = std::tanh(m.matrix_vals[i]);
        m_derivatives.matrix_vals[i] = 1.0f - tanh_val * tanh_val; // Derivative of tanh is 1 - tanh^2(x).
    }
}

/**
 * @brief Computes the derivative of the softmax function.
 * @param m_derivatives The matrix to store the derivatives.
 * @param m The matrix of input values (assumed to be the output of the softmax function).
 */
 void Functions::softmax_derivative(Matrix& m_derivatives, Matrix& m) {
    // Softmax derivative is computed as the Jacobian matrix.
    // For each element, the derivative is:
    // d(softmax_i)/d(x_j) = softmax_i * (1 - softmax_i) if i == j
    //                      -softmax_i * softmax_j if i != j

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.columns; j++) {
            float softmax_i = m.matrix_vals[i * m.columns + j];
            for (int k = 0; k < m.columns; k++) {
                if (j == k) {
                    m_derivatives.matrix_vals[i * m.columns * m.columns + j * m.columns + k] =
                        softmax_i * (1.0f - softmax_i);
                } else {
                    float softmax_k = m.matrix_vals[i * m.columns + k];
                    m_derivatives.matrix_vals[i * m.columns * m.columns + j * m.columns + k] =
                        -softmax_i * softmax_k;
                }
            }
        }
    }
}