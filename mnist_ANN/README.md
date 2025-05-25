# MNIST Handwritten Digit Predictor

A modular implementation of an Artificial Neural Network (ANN) for predicting handwritten digits using the MNIST dataset.

## Overview

This project implements a simple yet effective neural network to classify handwritten digits (0-9) from the famous MNIST dataset. The implementation is designed with modularity in mind, breaking down the machine learning pipeline into reusable functions.

## Features

- **Modular Design**: Each step of the ML pipeline is implemented as a separate function
- **Data Preprocessing**: Automatic normalization of pixel values
- **Simple Architecture**: 3-layer neural network (input → hidden → output)
- **Visualization**: Built-in prediction visualization with probability distributions
- **Random Testing**: Automated testing on random samples with confidence scores
- **High Accuracy**: Achieves good performance with minimal training

## Model Architecture

```
Input Layer:    784 neurons (28×28 flattened image)
Hidden Layer:   128 neurons (ReLU activation)
Output Layer:   10 neurons (Softmax activation for 10 digits)
```

## Key Functions

### Data Handling
- `load_and_preprocess_data()`: Loads MNIST data and normalizes pixel values
- `normalize_data()`: Utility function for data normalization

### Model Creation
- `create_model()`: Builds the neural network architecture
- `compile_model()`: Configures optimizer, loss function, and metrics

### Training & Evaluation
- `train_model()`: Trains the model on training data
- `evaluate_model()`: Tests model performance on test data

### Prediction & Visualization
- `make_prediction()`: Makes predictions on single images
- `test_random_predictions()`: Tests model on random samples
- `visualize_prediction()`: Shows image and prediction probabilities
- `format_prediction_result()`: Formats prediction output

### Utilities
- `select_random_samples()`: Selects random indices for testing
- `is_high_confidence()`: Checks if prediction confidence meets threshold

## Usage

Simply run all cells in the notebook. The `main()` function orchestrates the entire pipeline:

1. **Data Loading**: Loads and preprocesses MNIST dataset
2. **Model Creation**: Creates and compiles the neural network
3. **Training**: Trains the model for 3 epochs
4. **Evaluation**: Tests model accuracy on test set
5. **Random Testing**: Makes predictions on 5 random samples
6. **Visualization**: Shows detailed prediction for one sample

## Requirements

- TensorFlow/Keras
- NumPy
- Matplotlib

## Expected Performance

- **Training Time**: ~1-2 minutes on CPU
- **Test Accuracy**: ~97-98%
- **Model Size**: Lightweight (< 1MB)

