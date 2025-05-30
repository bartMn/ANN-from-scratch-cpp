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
 * @brief Tests the derivative of the cross-entropy loss.
 * @return 0 if the test passes, -1 otherwise.
 */
int test_cross_entropy_derivative() {
    float diff_vals[] = {0.5, -0.5, 1.0};
    Matrix diff(1, 3, diff_vals);
    Matrix derivatives(1, 3);

    Functions F;
    F.Cross_Entropy_derivative(derivatives, diff);

    float expected_vals[] = {-0.5, 0.5, -1.0};
    for (int i = 0; i < 3; i++) {
        if (std::abs(derivatives.get_val(0, i) - expected_vals[i]) > 1e-6) {
            std::cout << "test_cross_entropy_derivative Failed.\n";
            return -1;
        }
    }

    std::cout << "test_cross_entropy_derivative passed.\n";
    return 0;
}

/**
 * @brief Runs all function-related tests.
 * @return 0 if all tests pass, -1 otherwise.
 */
int run_functions_tests() {
    int status = 0;

    if (test_relu() != 0) status = -1;
    if (test_sigmoid() != 0) status = -1;
    if (test_softmax() != 0) status = -1;
    if (test_tanh() != 0) status = -1;
    if (test_diff() != 0) status = -1;
    if (test_relu_derivative() != 0) status = -1;
    if (test_sigmoid_derivative() != 0) status = -1;
    if (test_mse_derivative() != 0) status = -1;
    if (test_cross_entropy_derivative() != 0) status = -1;

    if (status == 0) {
        std::cout << "All functions tests passed successfully!\n";
    } else {
        std::cerr << "Some functions tests failed.\n";
    }

    return status;
}