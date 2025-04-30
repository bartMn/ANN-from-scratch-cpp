#include "functions.h"
#include <cmath>
#include <iostream>

void Functions::ReLu(Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) m.matrix_vals[i] = std::max(0.0f, m.matrix_vals[i]);
}

void Functions::sigmoid(Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) m.matrix_vals[i] = 1.0 / (1.0 + std::exp(-m.matrix_vals[i]));
}

void Functions::softmax(Matrix& m){
    float sum_of_exp = 0.0f;
    for (int i = 0; i < m.columns*m.rows; i++){
        m.matrix_vals[i] = std::exp(m.matrix_vals[i]);
        sum_of_exp += m.matrix_vals[i];
    } 
    m /= sum_of_exp;
}

void Functions::Tanh(Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) m.matrix_vals[i] = std::tanh(m.matrix_vals[i]);
}

void Functions::diff(Matrix& m_diff, Matrix& predictions, Matrix& y){
    if (predictions.rows != y.rows || predictions.columns != y.columns) {
        throw std::runtime_error("Matrix dimensions must match for diff calculation.");
    }

    for (int i = 0; i < predictions.rows * predictions.columns; i++) {
        m_diff.matrix_vals[i] = predictions.matrix_vals[i] - y.matrix_vals[i];
    }
}

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

float Functions::Cross_Entropy(Matrix& predictions, Matrix& y) {
    if (predictions.rows != y.rows || predictions.columns != y.columns) {
        throw std::runtime_error("Matrix dimensions must match for Cross Entropy calculation.");
    }

    float cross_entropy = 0.0f;
    for (int i = 0; i < predictions.rows * predictions.columns; i++) {
        if (y.matrix_vals[i] > 0) { // Avoid log(0) for non-target classes
            cross_entropy -= y.matrix_vals[i] * std::log(predictions.matrix_vals[i] + 1e-9f); // Add small epsilon to avoid log(0)
        }
    }
    cross_entropy /= predictions.rows*predictions.columns; // Average over the number of samples
    return cross_entropy;
}

void Functions::ReLu_derivative(Matrix& m_derivatives, Matrix& m){
    for (int i = 0; i < m.columns*m.rows; i++) {
        if (m.matrix_vals[i] > 0) {
            m_derivatives.matrix_vals[i] = 1.0f;
        } else {
            m_derivatives.matrix_vals[i] = 0.0f;
        }
    }
}
void Functions::sigmoid_derivative(Matrix& m_derivatives, Matrix& m){
    m_derivatives.elementWiseMultiply(m, 1.0 - m);
}

float Functions::MSE(Matrix& m_diff){
    float mse = 0.0f;
    for (int i = 0; i < m_diff.columns*m_diff.rows; i++) {
        mse += m_diff.matrix_vals[i] * m_diff.matrix_vals[i];
    }
    mse /= (m_diff.rows * m_diff.columns);
    return mse;
}

void Functions::MSE_derivative(Matrix& m_derivatives, Matrix& m_diff){
    int N = m_diff.rows * m_diff.columns;
    for (int i = 0; i < m_diff.columns*m_diff.rows; i++) {
        m_derivatives.matrix_vals[i] = (2.0f * m_diff.matrix_vals[i]) / N;
    }
}
void Functions::Cross_Entropy_derivative(Matrix& m_derivatives, Matrix& m_diff){
    for (int i = 0; i < m_diff.columns*m_diff.rows; i++) {
        m_derivatives.matrix_vals[i] = -m_diff.matrix_vals[i];
    }
}
//void Functions::softmax_derivative(Matrix& m);
//void Functions::Tanh_derivative(Matrix& m);