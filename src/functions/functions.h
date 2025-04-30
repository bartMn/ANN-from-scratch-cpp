#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "../matrix/matrix.h"

/**
 * @class Functions
 * @brief Implements various activation functions, loss functions, and their derivatives.
 */
class Functions {
    public:
        Functions() {} 
        // Activation functions
        void ReLu(Matrix& m); ///< Applies the ReLU activation function element-wise.
        void sigmoid(Matrix& m); ///< Applies the sigmoid activation function element-wise.
        void softmax(Matrix& m); ///< Applies the softmax function to the matrix.
        void Tanh(Matrix& m); ///< Applies the hyperbolic tangent function element-wise.

        // Loss functions
        void diff(Matrix& m_diff, Matrix& predictions, Matrix& y); ///< Computes the difference between predictions and ground truth.
        float MSE(Matrix& m_diff); ///< Computes the Mean Squared Error (MSE) from the difference matrix.
        float MSE(Matrix& predictions, Matrix& y); ///< Computes the MSE between predictions and ground truth.
        float Cross_Entropy(Matrix& predictions, Matrix& y); ///< Computes the cross-entropy loss.

        // Derivatives of activation and loss functions
        void ReLu_derivative(Matrix& m_derivatives, Matrix& m); ///< Computes the derivative of the ReLU function.
        void sigmoid_derivative(Matrix& m_derivatives, Matrix& m); ///< Computes the derivative of the sigmoid function.
        void MSE_derivative(Matrix& m_derivatives, Matrix& m_diff); ///< Computes the derivative of the MSE loss.
        void Cross_Entropy_derivative(Matrix& m_derivatives, Matrix& m_diff); ///< Computes the derivative of the cross-entropy loss.

        //void softmax_derivative(Matrix& m_derivatives, Matrix& m);
        //void Tanh_derivative(Matrix& m_derivatives, Matrix& m);
};


#endif