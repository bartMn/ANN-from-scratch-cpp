#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "../matrix/matrix.h"

class Functions {
    public:
        Functions() {} 
        void ReLu(Matrix& m);
        void sigmoid(Matrix& m);
        void softmax(Matrix& m);
        void Tanh(Matrix& m);
        void diff(Matrix& m_diff, Matrix& predictions, Matrix& y);
        float MSE(Matrix& m_diff);
        float MSE(Matrix& predictions, Matrix& y);
        float Cross_Entropy(Matrix& predictions, Matrix& y);

        void ReLu_derivative(Matrix& m_derivatives, Matrix& m);
        void sigmoid_derivative(Matrix& m_derivatives, Matrix& m);
        void softmax_derivative(Matrix& m_derivatives, Matrix& m);
        void Tanh_derivative(Matrix& m_derivatives, Matrix& m);
        void MSE_derivative(Matrix& m_derivatives, Matrix& m_diff);
        void Cross_Entropy_derivative(Matrix& m_derivatives, Matrix& m_diff);

};


#endif