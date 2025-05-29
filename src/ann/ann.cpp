#include "ann.h"
#include <iostream>
#include <cstring>

ANN::ANN(std::vector<int> layer_sizes, std::vector<std::string> activations){
    //std::function<void(Matrix&)> myFunc = [&](int a, int b) { obj.func1(a, b); };
    
    //activation_map["Relu"] = [&]() { F.ReLu(); };
    activation_map["ReLu"] = [&](Matrix& m) { F.ReLu(m); };
    derivative_map["ReLu"] = [&](Matrix& m_derivatives, Matrix& m) { F.ReLu_derivative(m_derivatives, m); };
    
    activation_map["sigmoid"] = [&](Matrix& m) { F.sigmoid(m); };
    derivative_map["sigmoid"] = [&](Matrix& m_derivatives, Matrix& m) { F.sigmoid_derivative(m_derivatives, m); };
    
    activation_map["softmax"] = [&](Matrix& m) { F.softmax(m); };
    derivative_map["softmax"] = [&](Matrix& m_derivatives, Matrix& m) { F.softmax_derivative(m_derivatives, m); };
    
    activation_map["Tanh"] = [&](Matrix& m) { F.Tanh(m); };
    derivative_map["Tanh"] = [&](Matrix& m_derivatives, Matrix& m) { F.Tanh_derivative(m_derivatives, m); };
    
    activation_map["linear"] = [&](Matrix& m) { F.linear(m); };
    derivative_map["linear"] = [&](Matrix& m_derivatives, Matrix& m) { F.linear_derivative(m_derivatives, m); };
    
    a_values.push_back(Matrix(layer_sizes[0], 1)); // Placeholder for outputs
    for (size_t i = 1; i < layer_sizes.size(); i++) {
        if (layer_sizes[i] == 0 || layer_sizes[i - 1] == 0) {
            throw std::runtime_error("Layer sizes must be greater than zero.");
        }
        weights.push_back(Matrix(layer_sizes[i], layer_sizes[i - 1])); // Random initialization
        biases.push_back(Matrix(layer_sizes[i], 1)); // Random initialization
        
        z_values.push_back(Matrix(layer_sizes[i], 1)); // Placeholder for outputs
        dz_values.push_back(Matrix(layer_sizes[i], 1)); // Placeholder for derivatives of outputs
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

    this->learning_rate = 0.01f; // Default learning rate
    this->loss_function = new char[4]; // Allocate memory for "MSE"
    strcpy(this->loss_function, "MSE");
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
    for (int i = weights.size() - 1; i > 0; i--) {
        dw_temp[i] = error_signals[i] * transpose(a_values[i]); // Gradient for weights
        db_temp[i] = error_signals[i]; // Gradient for biases
        dw_accumulated[i] += dw_temp[i]; // Accumulate gradients
        db_accumulated[i] += db_temp[i]; // Accumulate gradients
        error_signals[i-1].matrixMultiply(transpose(weights[i]), error_signals[i]); // Backpropagate the error signal
        derivatives_functions[i](dz_values[i-1], z_values[i-1]);
        error_signals[i-1].elementWiseMultiply(error_signals[i-1], dz_values[i-1]); // Element-wise multiplication
    }
    // Calculate gradients for the first layer
    dw_temp[0] = error_signals[0] * transpose(a_values[0]); 
    db_temp[0] = error_signals[0];
        
}

void ANN::set_optimizer(std::string optimizer, std::string loss_function, float learning_rate) {
    if (optimizer == "SGD") {
        std::cout << "Using Stochastic Gradient Descent (SGD) optimizer.\n";
    }
    else  {
        throw std::runtime_error("Unsupported optimizer: " + optimizer);
    }
    
    if (loss_function == "MSE" || loss_function == "Cross_Entropy"){
        std::cout << "Using " << loss_function << " loss function.\n";
    }
    else {
        throw std::runtime_error("Unsupported loss function: " + loss_function);
    }
    delete[] this->loss_function; // Free previous memory
    this->loss_function = new char[loss_function.length() + 1]; // Allocate memory for the new loss function
    strcpy(this->loss_function, loss_function.c_str()); // Copy the new loss function
    this->learning_rate = learning_rate;    
    std::cout << "Optimizer set to " << optimizer << " with learning rate " << learning_rate << ".\n";
}

void ANN::update_weights() {
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] -= dw_accumulated[i] * learning_rate;
        biases[i] -= db_accumulated[i] * learning_rate;
    }
}

void ANN::reset_gradients() {
    for (size_t i = 0; i < dw_accumulated.size(); i++) {
        dw_accumulated[i].resetWithVal(0.0f);
        db_accumulated[i].resetWithVal(0.0f);
    }
}


float ANN::calcualte_loss(Matrix& target) {
    if (a_values.back().get_rows_num() != target.get_rows_num() || a_values.back().get_columns_num() != target.get_columns_num()) {
        throw std::runtime_error("Output dimensions must match target dimensions for loss calculation.");
    }
    float loss = 0.0f;
    
    if (strcmp(loss_function, "MSE") == 0) {
        F.diff(error_signals.back(), a_values.back(), target);
        loss = F.MSE(error_signals.back());
        F.MSE_derivative(error_signals.back(), error_signals.back());
        derivatives_functions[derivatives_functions.size()-1](dz_values[dz_values.size() - 1], z_values[dz_values.size() - 1]);
        error_signals[error_signals.size()-1].elementWiseMultiply(error_signals[error_signals.size()-1], dz_values[dz_values.size()-1]);
    
    }
    else {
        F.diff(error_signals.back(), a_values.back(), target);
        //F.Cross_Entropy_derivative(error_signals.back(), error_signals.back());
        loss = F.Cross_Entropy(a_values.back(), target);
    }
    std::cout << "Loss: " << loss << "\n";
    return loss;
}

ANN::~ANN() {
    delete[] this->loss_function; // Free allocated memory
}