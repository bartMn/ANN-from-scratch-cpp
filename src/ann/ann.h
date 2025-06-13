#ifndef ANN_H
#define ANN_H

#include <functional>
#include <unordered_map>
#include <vector>
#include "../matrix/matrix.h"
#include "../functions/functions.h"


/**
 * @class ANN
 * @brief Implements an Artificial Neural Network with customizable layers and activations.
 */
class ANN {
public:
    ANN(std::vector<int> layer_sizes, std::vector<std::string> activations); // Constructor
    ~ANN(); // Destructor

    void forward(Matrix& input); // Forward pass
    void backprop(); // Backpropagation
    void set_optimizer(std::string optimizer = "SGD", std::string loss_function = "MSE", float learning_rate = 0.01f); // Set optimizer and loss function
    void update_weights(); // Update weights using gradients
    float calcualte_loss(Matrix& target); // Calculate loss
    void reset_gradients(); // Reset gradients for backpropagation
    void average_gradients(int batch_size);
    void clip_gradients(float max_norm); // Clip gradients to prevent exploding gradients
    float get_output_val(int row, int col);
    
private:
    Functions F; // Functions object for activations/losses
    float learning_rate; // Learning rate for weight updates
    char *loss_function; // Loss function to be used (e.g., "MSE", "Cross_Entropy")
    std::unordered_map<std::string, std::function<void(Matrix&)>> activation_map;
    std::unordered_map<std::string, std::function<void(Matrix&, Matrix&)>> derivative_map;
    std::vector<Matrix> weights; // Weight matrices for each layer
    std::vector<Matrix> biases; // Bias vectors for each layer
    std::vector<Matrix> z_values; // Pre-activation outputs for each layer
    std::vector<Matrix> dz_values; // Derivatives of pre-activation outputs
    std::vector<Matrix> a_values; // Outputs after activation
    std::vector<Matrix> dw_accumulated; // Accumulated gradients for weights
    std::vector<Matrix> dw_temp; // Temporary gradients for weights
    std::vector<Matrix> db_accumulated; // Accumulated gradients for biases
    std::vector<Matrix> db_temp; // Temporary gradients for biases
    std::vector<Matrix> error_signals; // Error signals for backpropagation
    std::vector<std::function<void(Matrix&)>> activation_functions; // Activation functions
    std::vector<std::function<void(Matrix&, Matrix&)>> derivatives_functions; // Activation derivatives

};

#endif // ANN_H