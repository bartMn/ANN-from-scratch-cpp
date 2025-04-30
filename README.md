# ANN from Scratch

## Disclaimer
This project is a work in progress. The current implementation includes foundational components for building neural networks, such as matrix operations, activation functions, and loss functions. The implementation of a Multi-Layer Perceptron (MLP) with forward and backpropagation is planned for future updates.

## Overview
This project implements a neural network library from scratch in C++ with support for matrix operations, activation functions, and loss functions. It includes a basic framework for building and testing artificial neural networks (ANNs).

## Features
- Matrix operations (addition, subtraction, multiplication, etc.)
- Activation functions (ReLU, sigmoid, softmax, tanh)
- Loss functions (Mean Squared Error, Cross-Entropy)
- Derivatives for backpropagation
- Unit tests for all core components
- Multithreading support for performance optimization

## Getting Started

### Prerequisites
- A C++ compiler (e.g., `g++`) with support for C++11 or later
- `make` for building the project
- Linux or any Unix-based operating system (tested on Linux)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ANN_from_scratch.git
   cd ANN_from_scratch

2. Build the project:
    make -f Makefile.mak

3. Run the tests:
    ./my_project


### Project Structure
ANN_from_scratch/  
├── src/  
│   ├── matrix/          # Matrix operations  
│   ├── functions/       # Activation, loss and derivative functions  
│   └── main.cpp         # Entry point of the program  
├── tests/  
│   ├── matrix/          # Unit tests for matrix operations  
│   ├── functions/       # Unit tests for functions  
├── Makefile.mak         # Build configuration  
└── README.md            # Project documentation  