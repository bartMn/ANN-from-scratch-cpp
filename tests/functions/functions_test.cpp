#include <iostream>
#include <cassert>
#include <cmath>
#include "../../src/functions/functions.h"
#include "../../src/matrix/matrix.h"
#include "functions_test.h"

/**
 * @brief Tests the ReLU activation function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_relu() {
    float vals[] = {-3, -2, -1, 0, 1, 2, 3};
    Matrix m(1, 7, vals);
    Functions F;
    F.ReLu(m);

    float expected_vals[] = {0, 0, 0, 0, 1, 2, 3};
    for (int i = 0; i < 7; i++) {
        if (m.get_val(0, i) != expected_vals[i]) {
            std::cout << "test_relu Failed.\n";
            return -1;
        }
    }
    std::cout << "test_relu passed.\n";
    return 0;
}

/**
 * @brief Tests the sigmoid activation function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_sigmoid() {
    float vals[] = {0, 1, -1, 2, -2};
    Matrix m(1, 5, vals);
    Functions F;
    F.sigmoid(m);

    float expected_vals[] = {0.5, 0.7310586, 0.2689414, 0.8807971, 0.1192029};
    for (int i = 0; i < 5; i++) {
        if (std::abs(m.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_sigmoid Failed.\n";
            return -1;
        }
    }
    std::cout << "test_sigmoid passed.\n";
    return 0;
}

/**
 * @brief Tests the softmax function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_softmax() {
    float vals[] = {1, 2, 3};
    Matrix m(1, 3, vals);
    Functions F;
    F.softmax(m);

    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        sum += m.get_val(0, i);
    }

    if (std::abs(sum - 1.0f) > 1e-6) {
        std::cout << "test_softmax Failed (sum != 1).\n";
        return -1;
    }

    std::cout << "test_softmax passed.\n";
    return 0;
}

/**
 * @brief Tests the hyperbolic tangent (tanh) activation function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_tanh() {
    float vals[] = {0, 1, -1, 2, -2};
    Matrix m(1, 5, vals);
    Functions F;
    F.Tanh(m);

    float expected_vals[] = {0, 0.7615942, -0.7615942, 0.9640276, -0.9640276};
    for (int i = 0; i < 5; i++) {
        if (std::abs(m.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_tanh Failed.\n";
            return -1;
        }
    }
    std::cout << "test_tanh passed.\n";
    return 0;
}

/**
 * @brief Tests the difference calculation between predictions and ground truth.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_diff() {
    float pred_vals[] = {1.0, 2.0, 3.0};
    float true_vals[] = {1.0, 2.5, 3.5};
    Matrix predictions(1, 3, pred_vals);
    Matrix y(1, 3, true_vals);
    Matrix diff(1, 3);

    Functions F;
    F.diff(diff, predictions, y);

    float expected_vals[] = {0.0, -0.5, -0.5};
    for (int i = 0; i < 3; i++) {
        if (std::abs(diff.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_diff Failed.\n";
            return -1;
        }
    }

    std::cout << "test_diff passed.\n";
    return 0;
}

/**
 * @brief Tests the derivative of the ReLU function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_relu_derivative() {
    float vals[] = {-1.0, 0.0, 1.0};
    Matrix m(1, 3, vals);
    Matrix derivatives(1, 3);

    Functions F;
    F.ReLu_derivative(derivatives, m);

    float expected_vals[] = {0.0, 0.0, 1.0};
    for (int i = 0; i < 3; i++) {
        if (std::abs(derivatives.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_relu_derivative Failed.\n";
            return -1;
        }
    }

    std::cout << "test_relu_derivative passed.\n";
    return 0;
}

/**
 * @brief Tests the derivative of the sigmoid function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_sigmoid_derivative() {
    float vals[] = {0.5, 0.8, 0.2};
    Matrix m(1, 3, vals);
    Matrix derivatives(1, 3);

    Functions F;
    F.sigmoid_derivative(derivatives, m);

    float expected_vals[] = {0.25, 0.16, 0.16};
    for (int i = 0; i < 3; i++) {
        if (std::abs(derivatives.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_sigmoid_derivative Failed.\n";
            return -1;
        }
    }

    std::cout << "test_sigmoid_derivative passed.\n";
    return 0;
}

/**
 * @brief Tests the derivative of the softmax function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_tanh_derivative() {
    // Input values to test
    float input_vals[] = {0.0f, 0.5f, -1.0f, 2.0f};
    Matrix input(1, 4, input_vals);
    Matrix derivatives(1, 4);

    Functions F;
    F.Tanh_derivative(derivatives, input);

    // Expected derivatives: 1 - tanh(x)^2
    float expected_vals[4];
    for (int i = 0; i < 4; i++) {
        float tanh_val = std::tanh(input_vals[i]);
        expected_vals[i] = 1.0f - tanh_val * tanh_val;
    }

    // Check results
    for (int i = 0; i < 4; i++) {
        if (std::abs(derivatives.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_tanh_derivative FAILED at index " << i << "\n";
            std::cout << "Expected: " << expected_vals[i]
                      << ", Got: " << derivatives.get_val(0, i) << "\n";
            return -1;
        }
    }

    std::cout << "test_tanh_derivative passed.\n";
    return 0;
}

/**
 * @brief Tests the linear activation function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_linear_activation() {
    float vals[] = {1.0, -2.0, 3.0};
    Matrix m(1, 3, vals);
    Functions F;
    F.linear(m);
    for (int i = 0; i < 3; i++) {
        if (m.get_val(0, i) != vals[i]) {
            std::cout << "test_linear_activation Failed.\n";
            return -1;
        }
    }
    std::cout << "test_linear_activation passed.\n";
    return 0;
}


/**
 * @brief Tests the derivative of the linear activation function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_linear_derivative() {
    float vals[] = {1.0, -2.0, 3.0};
    Matrix m(1, 3, vals);
    Matrix deriv(1, 3);
    Functions F;
    F.linear_derivative(deriv, m);
    for (int i = 0; i < 3; i++) {
        if (deriv.get_val(0, i) != 1.0f) {
            std::cout << "test_linear_derivative Failed.\n";
            return -1;
        }
    }
    std::cout << "test_linear_derivative passed.\n";
    return 0;
}


/**
 * @brief Tests the Mean Squared Error (MSE) loss function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_mse() {
    float pred_vals[] = {1.0f, 2.0f, 3.0f};
    float y_vals[] = {2.0f, 2.0f, 4.0f};
    Matrix predictions(1, 3, pred_vals);
    Matrix y(1, 3, y_vals);

    Functions F;
    float mse = F.MSE(predictions, y);

    // Expected: ((1-2)^2 + (2-2)^2 + (3-4)^2) / 3 = (1 + 0 + 1) / 3 = 0.666...
    float expected = (1.0f + 0.0f + 1.0f) / 3.0f;
    if (std::abs(mse - expected) > 1e-6) {
        std::cout << "test_mse Failed.\n";
        return -1;
    }
    std::cout << "test_mse passed.\n";
    return 0;
}

/**
 * @brief Tests the derivative of the Mean Squared Error (MSE) loss.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_mse_derivative() {
    float diff_vals[] = {0.5, -0.5, 1.0};
    Matrix diff(1, 3, diff_vals);
    Matrix derivatives(1, 3);

    Functions F;
    F.MSE_derivative(derivatives, diff);

    int N = diff.get_rows_num() * diff.get_columns_num();
    float expected_vals[] = {(2.0f * 0.5f) / N, (2.0f * -0.5f) / N, (2.0f * 1.0f) / N};
    for (int i = 0; i < 3; i++) {
        if (std::abs(derivatives.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_mse_derivative Failed.\n";
            return -1;
        }
    }

    std::cout << "test_mse_derivative passed.\n";
    return 0;
}


/**
 * @brief Tests the cross-entropy loss function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_cross_entropy() {
    // predictions: softmax output, y: one-hot
    float pred_vals[] = {0.7f, 0.2f, 0.1f};
    float y_vals[] = {1.0f, 0.0f, 0.0f};
    Matrix predictions(1, 3, pred_vals);
    Matrix y(1, 3, y_vals);

    Functions F;
    float ce = F.Cross_Entropy(predictions, y);

    // Expected: -sum(y_i * log(pred_i)) = -log(0.7)
    float expected = -std::log(0.7f);
    if (std::abs(ce - expected) > 1e-5) {
        std::cout << "test_cross_entropy Failed.\n";
        return -1;
    }
    std::cout << "test_cross_entropy passed.\n";
    return 0;
}

/**
 * @brief Tests the derivative of the cross-entropy loss.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_cross_entropy_derivative() {
    // Example: y = [1, 0, 1], y_pred = [0.8, 0.5, 0.25]
    float y_vals[]      = {1.0, 0.0, 1.0};
    float y_pred_vals[] = {0.8, 0.5, 0.25};
    Matrix y(1, 3, y_vals);
    Matrix y_pred(1, 3, y_pred_vals);
    Matrix derivatives(1, 3);

    Functions F;
    F.Cross_Entropy_derivative(derivatives, y, y_pred);

    // Expected derivative: -y_i / y_pred_i = [-1.25, 0, -4.0]
    float expected_vals[] = {-1.25, 0.0, -4.0};

    for (int i = 0; i < 3; i++) {
        if (std::abs(derivatives.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_cross_entropy_derivative FAILED at index " << i << "\n";
            std::cout << "Expected: " << expected_vals[i]
                      << ", Got: " << derivatives.get_val(0, i) << "\n";
            return -1;
        }
    }

    std::cout << "test_cross_entropy_derivative passed.\n";
    return 0;
}

/**
 * @brief Tests the derivative of the softmax function.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_softmax_derivative() {
    // Single example, 3 classes softmax output vector
    float softmax_vals[] = {0.2f, 0.5f, 0.3f};
    Matrix softmax_output(3, 1, softmax_vals);

    // Jacobian matrix size: 3 classes * 3 classes = 9 elements
    Matrix jacobian(3, 3); // pre-allocated to hold the 3x3 Jacobian for the single example

    Functions F;
    F.softmax_derivative(jacobian, softmax_output);

    // Compute expected Jacobian manually
    // Formula:
    // J[i,j] = softmax_i * (1 - softmax_i) if i == j
    //         = -softmax_i * softmax_j if i != j
    float expected_jacobian[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                expected_jacobian[i * 3 + j] = softmax_vals[i] * (1.0f - softmax_vals[i]);
            } else {
                expected_jacobian[i * 3 + j] = -softmax_vals[i] * softmax_vals[j];
            }
        }
    }

    // Check all entries

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            if (std::abs(jacobian.get_val(r, c) - expected_jacobian[r*3+c]) > 1e-6) {
            std::cout << "test_softmax_derivative FAILED at row " << r << " col "<<  c<< "\n";
            std::cout << "Expected: " << expected_jacobian[r*3+c] << ", Got: " << jacobian.get_val(r, c) << "\n";
            return -1;
            }
        }
    }

    std::cout << "test_softmax_derivative passed.\n";
    return 0;
}

/**
 * @brief Runs all function-related tests.
 * @return 0 if all tests pass, -1 otherwise.
 */
int run_functions_tests() {
    int status = 0;

    std::cout << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << "#########   RUNNING FUNCTIONS TESTS... ############" << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << std::endl;

    if (test_relu() != 0) status = -1;
    if (test_relu_derivative() != 0) status = -1;
    if (test_sigmoid() != 0) status = -1;
    if (test_sigmoid_derivative() != 0) status = -1;
    if (test_softmax() != 0) status = -1;
    if (test_softmax_derivative() != 0) status = -1;
    if (test_tanh() != 0) status = -1;
    if (test_tanh_derivative() != 0) status = -1;
    if (test_diff() != 0) status = -1;
    if (test_linear_activation() != 0) status = -1;
    if (test_linear_derivative() != 0) status = -1;
    if (test_mse() != 0) status = -1;
    if (test_mse_derivative() != 0) status = -1;
    if (test_cross_entropy() != 0) status = -1;
    if (test_cross_entropy_derivative() != 0) status = -1;

    if (status == 0) {
        std::cout << "All functions tests passed successfully!\n";
    } else {
        std::cerr << "Some functions tests failed.\n";
    }

    std::cout << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << "############  FUNCTIONS TESTS DONE... #############" << std::endl;
    std::cout << "###################################################" << std::endl;
    std::cout << std::endl;

    return status;
}