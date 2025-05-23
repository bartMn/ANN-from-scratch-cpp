#include "ann.h"
#include <iostream>

ANN::ANN(std::vector<int> layer_sizes, std::vector<std::string> activations){
    //std::function<void(Matrix&)> myFunc = [&](int a, int b) { obj.func1(a, b); };
    
    //activation_map["Relu"] = [&]() { F.ReLu(); };
    activation_map["ReLu"] = [&](Matrix& m) { F.ReLu(m); };
    derivative_map["ReLu"] = [&](Matrix& m_derivatives, Matrix& m) { F.ReLu_derivative(m_derivatives, m); };
    activation_map["sigmoid"] = [&](Matrix& m) { F.sigmoid(m); };
    derivative_map["sigmoid"] = [&](Matrix& m_derivatives, Matrix& m) { F.sigmoid_derivative(m_derivatives, m); };
    //activation_map["Cross_Entropy"] = [&](Matrix& m) { F.Cross_Entropy(m); };
    //derivative_map["Cross_Entropy"] = [&](Matrix& m_derivatives, Matrix& m) { F.Cross_Entropy_derivative(m_derivatives, m); };

    a_values.push_back(Matrix(layer_sizes[0], 1)); // Placeholder for outputs
    for (size_t i = 1; i < layer_sizes.size(); i++) {
        if (layer_sizes[i] == 0 || layer_sizes[i - 1] == 0) {
            throw std::runtime_error("Layer sizes must be greater than zero.");
        }
        weights.push_back(Matrix(layer_sizes[i], layer_sizes[i - 1])); // Random initialization
        biases.push_back(Matrix(layer_sizes[i], 1)); // Random initialization
        
        z_values.push_back(Matrix(layer_sizes[i], 1)); // Placeholder for outputs
        a_values.push_back(Matrix(layer_sizes[i], 1)); // Placeholder for outputs
        error_signals.push_back(Matrix(layer_sizes[i], 1)); // Placeholder for error signals

        db_accumulated.push_back(Matrix(layer_sizes[i], 1)); // Placeholder for gradients
        db_temp.push_back(Matrix(layer_sizes[i], 1));// Placeholder for gradients
        dw_accumulated.push_back(Matrix(layer_sizes[i], layer_sizes[i - 1])); // Placeholder for gradients
        dw_temp.push_back(Matrix(layer_sizes[i], layer_sizes[i - 1])); // Placeholder for gradients
        
        
        activation_functions.push_back(activation_map[activations[i - 1]]);
        derivatives_functions.push_back(derivative_map[activations[i - 1]]);
        
    }

    for (auto & w : weights) {
        w.randomInit();
        //w.printMatrix();
    }
    for (auto & b : biases) {
        b.randomInit();
    }
    std::cout << "ANN initialized with " << layer_sizes.size() << " layers.\n";
    std::cout << weights[0].get_rows_num() << "\n";
    std::cout << "weights.size() = " << weights.size() << "\n";
    std::cout << "layer_sizes.size() = " << layer_sizes.size() << "\n";
    std::cout << "z_values.size() = " << z_values.size() << "\n";
    std::cout << "a_values.size() = " << a_values.size() << "\n";
    std::cout << "activation_functions.size() = " << activation_functions.size() << "\n";
}

void ANN::forward(Matrix& input) {
    a_values[0].setValsFormMatrix(input); // Input layer
    
    for (size_t i = 0; i < weights.size(); i++) {
        z_values[i].matrixMultiply(weights[i], a_values[i]);// + biases[i];
        z_values[i] += biases[i];
        a_values[i+1].setValsFormMatrix(z_values[i]); // Copy z_values to a_values
        activation_functions[i]( a_values[i+1] ); // Apply the activation function
    }
    
    a_values.back().printMatrix();
}


void ANN::backprop() {
    for (int i = weights.size() - 1; i >= 0; i--) {
        dw_temp[i] = error_signals[i] * a_values[i]; // Gradient for weights
        db_temp[i] = error_signals[i]; // Gradient for biase    s
        error_signals[i-1].matrixMultiply(weights[i], error_signals[i]); // Backpropagate the error signal
        error_signals[i-1].elementWiseMultiply(error_signals[i-1], a_values[i]); // Element-wise multiplication
        //error_signals[i-1] = (weights[i] * error_signals[i]) ^ derivatives_functions[i](z_values[i-1]); // Backpropagate the error signal
    }
}

void ANN::update_weights(float learning_rate) {
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] -= dw_accumulated[i] * learning_rate;
        biases[i] -= db_accumulated[i] * learning_rate;
    }
}

