import numpy as np

# Mean Square Error

def mse(actual, predicted):
    return np.mean(np.power(actual - predicted, 2))

def mse_prime(actual, predicted):
    return 2 * (predicted - actual) / np.size(actual)

# Binary Cross Entropy

def bce(actual, predicted):
    return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

def bce_prime(actual, predicted):
    return ((1 - actual) / (1 - predicted) - actual / predicted) / np.size(actual)
