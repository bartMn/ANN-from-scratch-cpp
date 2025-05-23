#ifndef ANN_H
#define ANN_H


#include <functional>
#include <unordered_map>
#include <vector>
#include "../matrix/matrix.h"
#include "../functions/functions.h"

class ANN {
public:
    ANN(std::vector<int> layer_sizes, std::vector<std::string> activations);

    void forward(Matrix& input); // Forward pass
    void backprop(); // Backpropagation
    void update_weights(float learning_rate); // Update weights using gradients

private:
    Functions F;
    std::unordered_map<std::string, std::function<void(Matrix&)>> activation_map;
    std::unordered_map<std::string, std::function<void(Matrix&, Matrix&)>> derivative_map;
    std::vector<Matrix> weights; // Weight matrices for each layer
    std::vector<Matrix> biases; // Bias vectors for each layer
    std::vector<Matrix> z_values; // Outputs for each layer
    std::vector<Matrix> dz_values; // Outputs for each layer
    std::vector<Matrix> a_values; // Outputs after activation
    std::vector<Matrix> dw_accumulated; // Gradients for backpropagation
    std::vector<Matrix> dw_temp; // Gradients for backpropagation
    std::vector<Matrix> db_accumulated; // Gradients for backpropagation
    std::vector<Matrix> db_temp; // Gradients for backpropagation
    std::vector<Matrix> error_signals; // Error signals for backpropagation
    //std::unordered_map<std::string, std::function<void(Matrix&)>> activation_map;
    std::vector<std::function<void(Matrix&)>> activation_functions; // Activation derivatives
    std::vector<std::function<void(Matrix&, Matrix&)>> derivatives_functions; // Activation derivatives

};

#endif // ANN_H