# ANN from Scratch

## Disclaimer
This project is a work in progress. The current implementation includes foundational components for building neural networks, such as matrix operations, activation functions, and loss functions. The implementation of a Multi-Layer Perceptron (MLP) with forward and backpropagation is provided, but some features are still under development.

## Overview
This project implements a neural network library from scratch in C++ with support for matrix operations, activation functions, and loss functions. It includes a basic framework for building and testing artificial neural networks (ANNs).

## Features
- Matrix operations (addition, subtraction, multiplication, etc.)
- Activation functions: ReLU, sigmoid, softmax, tanh, linear
- Loss functions: Mean Squared Error (MSE), Cross-Entropy
- Derivatives for all activation and loss functions (for backpropagation)
- Modular design: separate folders for matrix, functions, and ANN logic
- Unit tests for all core components (matrix, functions, ANN)
- Support for setting optimizer (currently only SGD) and loss function via [`ANN::set_optimizer`](src/ann/ann.cpp)
- training support via [`ANN::train_epoch`](src/ann/ann.cpp) and [`ANN::train_model`](src/ann/ann.cpp)
- Evaluation on validation/test sets via [`ANN::run_evaluation`](src/ann/ann.cpp)
- Example training loop and loss calculation in [`tests/ann/ann_test.cpp`](tests/ann/ann_test.cpp)
- Multithreading support for performance optimization

## Project Structure
```
ANN_from_scratch/
├── src/
│   ├── ann/             # Artificial Neural Network (ANN)
│   ├── functions/       # Activation, loss and derivative functions
│   ├── matrix/          # Matrix operations
│   └── main.cpp         # Entry point of the program
├── tests/
│   ├── ann/             # Unit tests for ANN
│   ├── functions/       # Unit tests for functions
│   └── matrix/          # Unit tests for matrix operations
├── Makefile.mak         # Build configuration
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites
- A C++ compiler (e.g., `g++`) with support for C++17 or later
- `make` for building the project
- Linux or any Unix-based operating system (tested on Linux)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bartMn/ANN_from_scratch.git
   cd ANN_from_scratch
   ```

2. Build the project:
   ```bash
   make -f Makefile.mak
   ```

3. Run the tests:
   ```bash
   ./my_project
   ```

## Usage

- The main entry point is [`src/main.cpp`](src/main.cpp), which runs all unit tests.
- To use the ANN, see the example code in [`tests/ann/ann_test.cpp`](tests/ann/ann_test.cpp).
- You can set the optimizer and loss function using `ANN::set_optimizer`. Only `"SGD"` optimizer and `"MSE"` or `"Cross_Entropy"` loss functions are currently supported.

## Current Limitations

- **Cross-Entropy Loss:** When using cross-entropy loss, it is assumed that the output layer uses either softmax or sigmoid activation. No checks are enforced for this.
- **Regression and Classification:** Only basic regression with small dataset has been tested. Classification has not been tested yet. More extensive testing is needed.
- **Optimizers:** Only Stochastic Gradient Descent (SGD) is implemented. Other optimizers (e.g., Adam) are not supported.
- **Error Handling:** Some error messages may be basic or missing for certain edge cases.

## Planned Improvements

- Add more complex regression tests.
- Add classification tests and examples.
- Implement and test additional optimizers (e.g., Adam).
- Improve error handling and input validation.
- Add more documentation and usage examples.

---

For more details, see the code in the [`src/`](src/) and [`tests/`](tests/) directories.