#include "ann.h"
#include <iostream>
#include <cstring>
#include <cmath>
/**
 * @brief Constructs an ANN with the given layer sizes and activation functions.
 * Initializes weights, biases, and function maps.
 * @param layer_sizes Vector of integers specifying the size of each layer.
 * @param activations Vector of strings specifying the activation function for each layer.
 */
ANN::ANN(std::vector<int> layer_sizes, std::vector<std::string> activations){

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
    
    a_values.push_back(Matrix(layer_sizes[0], 1)); // Input layer output
    for (size_t i = 1; i < layer_sizes.size(); i++) {
        if (layer_sizes[i] == 0 || layer_sizes[i - 1] == 0) {
            throw std::runtime_error("Layer sizes must be greater than zero.");
        }

        weights.push_back(Matrix(layer_sizes[i], layer_sizes[i - 1])); 
        biases.push_back(Matrix(layer_sizes[i], 1));
        
        z_values.push_back(Matrix(layer_sizes[i], 1)); 
        dz_values.push_back(Matrix(layer_sizes[i], 1));
        a_values.push_back(Matrix(layer_sizes[i], 1));
        error_signals.push_back(Matrix(layer_sizes[i], 1));

        db_accumulated.push_back(Matrix(layer_sizes[i], 1)); 
        db_temp.push_back(Matrix(layer_sizes[i], 1));
        dw_accumulated.push_back(Matrix(layer_sizes[i], layer_sizes[i - 1])); 
        dw_temp.push_back(Matrix(layer_sizes[i], layer_sizes[i - 1]));
        
        activation_functions.push_back(activation_map[activations[i - 1]]);
        derivatives_functions.push_back(derivative_map[activations[i - 1]]);
        
    }

    // Random initialization of weights
    for (auto & w : weights) {
        //w.randomInit();
        w.randomHeUniformInit();
    }

    // Random initialization of biases
    for (auto & b : biases) {
        //b.randomInit();
        //b.randomHeInit();
        float bias_scale = std::sqrt(2.0f / b.get_rows_num());  // Match He-normal std dev
        float bias = 0.01f * bias_scale;
        b.resetWithVal(bias); // Initialize biases to 0.0001f
    }

    this->learning_rate = 0.01f; // Default learning rate
    this->loss_function = new char[4]; // Allocate memory for "MSE"
    strcpy(this->loss_function, "MSE");
    
    /*
    std::cout << "ANN initialized with " << layer_sizes.size() << " layers.\n";
    std::cout << weights[0].get_rows_num() << "\n";
    std::cout << "weights.size() = " << weights.size() << "\n";
    std::cout << "layer_sizes.size() = " << layer_sizes.size() << "\n";
    std::cout << "z_values.size() = " << z_values.size() << "\n";
    std::cout << "a_values.size() = " << a_values.size() << "\n";
    std::cout << "activation_functions.size() = " << activation_functions.size() << "\n";
    */
}


/**
 * @brief Performs a forward pass through the network.
 * @param input Input matrix to the network.
 */
void ANN::forward(Matrix& input) {
    a_values[0].setValsFormMatrix(input); // Input layer
    for (size_t i = 0; i < weights.size(); i++) {
        z_values[i].matrixMultiply(weights[i], a_values[i]);
        z_values[i] += biases[i];
        a_values[i+1].setValsFormMatrix(z_values[i]); // Copy z_values to a_values
        activation_functions[i]( a_values[i+1] ); // Apply the activation function
    }
    //a_values.back().printMatrix();
}


/**
 * @brief Performs backpropagation to compute gradients.
 */
void ANN::backprop() {
    for (int i = weights.size() - 1; i > 0; i--) {
        dw_temp[i] = error_signals[i] * transpose(a_values[i]); // Gradient for weights
        db_temp[i] = error_signals[i]; // Gradient for biases
        dw_accumulated[i] += dw_temp[i]; // Accumulate gradients for weights
        db_accumulated[i] += db_temp[i]; // Accumulate gradients for biases
        error_signals[i-1].matrixMultiply(transpose(weights[i]), error_signals[i]); // Backpropagate the error signal
        derivatives_functions[i](dz_values[i-1], z_values[i-1]); // Calculate the derivative of the activation function
        // Element-wise multiplication of the error signal with the derivative
        error_signals[i-1].elementWiseMultiply(error_signals[i-1], dz_values[i-1]); // Element-wise multiplication
    }
    // Calculate gradients for the first layer
    dw_temp[0] = error_signals[0] * transpose(a_values[0]); 
    db_temp[0] = error_signals[0];
        
}


/**
 * @brief Sets the optimizer, loss function, and learning rate.
 * @param optimizer Optimizer name (default: "SGD").
 * @param loss_function Loss function name (default: "MSE").
 * @param learning_rate Learning rate (default: 0.01).
 */
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


/**
 * @brief Updates the weights and biases using accumulated gradients.
 */
void ANN::update_weights() {
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] -= dw_accumulated[i] * learning_rate;
        biases[i] -= db_accumulated[i] * learning_rate;
    }
}

/**
 * @brief Resets all accumulated gradients to zero.
 */
void ANN::reset_gradients() {
    for (size_t i = 0; i < dw_accumulated.size(); i++) {
        dw_accumulated[i].resetWithVal(0.0f);
        db_accumulated[i].resetWithVal(0.0f);
    }
}

void ANN::average_gradients(int batch_size) {
    for (size_t i = 0; i < dw_accumulated.size(); i++) {
        dw_accumulated[i] /= batch_size;
        db_accumulated[i] /= batch_size;
    }
}

void ANN::clip_gradients(float max_norm){
    
    for (long unsigned grad_idx =0; grad_idx < dw_accumulated.size(); grad_idx++){
        float sum = 0.0f;
        for (int row = 0; row < dw_accumulated[grad_idx].get_rows_num(); row++) {
            for (int col = 0; col < dw_accumulated[grad_idx].get_columns_num(); col++) {
                sum += dw_accumulated[grad_idx].get_val(row, col) * dw_accumulated[grad_idx].get_val(row, col);
            }
        }
        for (int row = 0; row < db_accumulated[grad_idx].get_rows_num(); row++) {
            for (int col = 0; col < db_accumulated[grad_idx].get_columns_num(); col++) {
                sum += db_accumulated[grad_idx].get_val(row, col) * db_accumulated[grad_idx].get_val(row, col);
            }
        }

        float norm = std::sqrt(sum);
        if (norm > max_norm) {
            float scale = max_norm / norm;
            dw_accumulated[grad_idx] *= scale;
            db_accumulated[grad_idx] *= scale;
        }

    }
    
}


/**
 * @brief Calculates the loss and prepares error signals for backpropagation.
 * @param target Target output matrix.
 * @return Computed loss value.
 */
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
    else { // Cross-Entropy Loss
        // assuming softmax or sigmoid activation in the last layer
        F.diff(error_signals.back(), a_values.back(), target);
        loss = F.Cross_Entropy(a_values.back(), target);
    }
    //std::cout << "Loss: " << loss << "\n";
    return loss;
}

float ANN::get_output_val(int row, int col){
    return a_values.back().get_val(row, col);
}

float ANN::train_epoch(std::vector<std::array<Matrix, 2>>& train_set, int batch_size){
    
    std::cout << "Training with batch size: " << batch_size << "\n";
    std::cout << "Number of training batches: " << int(train_set.size()/ batch_size) << "\n";
    
    float running_loss = 0.0f;
    int ct = 0;
    for (int batch_num=0; batch_num < int(train_set.size()/ batch_size); batch_num++){
        reset_gradients();
        for (int sample_num=0; sample_num< batch_size; sample_num++){
            auto& [x, y] = train_set[batch_num * batch_size + sample_num];
            forward(x);
            running_loss += calcualte_loss(y);
            ct++;
            backprop();
        }
        average_gradients(batch_size);
        clip_gradients(1.0f); // Clip gradients to prevent exploding gradients
        update_weights();
    }
    std::cout << "ct: " << ct << " int(train_set.size()/ batch_size): "<<int(train_set.size()/ batch_size)<<"\n";
    return running_loss / ct;
}


float ANN::run_evaluation(std::vector<std::array<Matrix, 2>>& eval_set){
    float running_loss = 0.0f;
    for (long unsigned sample_idx =0; sample_idx < eval_set.size(); sample_idx++){
        auto& [x, y] = eval_set[sample_idx];
        forward(x);
        running_loss += calcualte_loss(y);
    }

    return running_loss / eval_set.size();
}

void ANN::train_model(std::vector<std::array<Matrix, 2>>& train_set, std::vector<std::array<Matrix, 2>>& eval_set, int epochs, long unsigned batch_size) {
    if (epochs <= 0) {
        throw std::runtime_error("Number of epochs must be greater than zero.");
    }
    if (batch_size <= 0 || batch_size > train_set.size()) {
        throw std::runtime_error("Invalid batch size.");
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float train_loss = train_epoch(train_set, batch_size);
        float eval_loss = run_evaluation(eval_set);
        std::cout << "Epoch " << epoch + 1 << ": Train Loss = " << train_loss << ", Eval Loss = " << eval_loss << "\n";
    }
}

/**
 * @brief Destructor for ANN. Frees allocated memory.
 */
ANN::~ANN() {
    delete[] this->loss_function; // Free allocated memory
}