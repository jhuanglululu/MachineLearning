import numpy as np

from neuralnetwork.layer import SoftMax

# Mean Square Error

def mse(actual, predicted):
    return np.mean(np.power(actual - predicted, 2))

def mse_prime(actual, predicted):
    if actual.ndim > 1:
        batch_size = actual.shape[0]
        return 2 * (predicted - actual) / batch_size
    else:
        return 2 * (predicted - actual) / np.size(actual)

# Binary Cross Entropy

def bce(actual, predicted):
    return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

def bce_prime(actual, predicted):
    if actual.ndim > 1:
        batch_size = actual.shape[0]
        return ((1 - actual) / (1 - predicted) - actual / predicted) / batch_size
    else:
        return ((1 - actual) / (1 - predicted) - actual / predicted) / np.size(actual)

# Cross Entropy

def cross_entropy(predict: np.ndarray, actual: np.ndarray, padding_idx: int = 0) -> float:
    # Apply softmax to the logits (predict)
    softmax = SoftMax()
    predict_softmax = softmax.forward(predict)

    epsilon = 1e-9
    predict_clipped = np.clip(predict_softmax, epsilon, 1. - epsilon)
    non_padding_mask = (actual[:, :, padding_idx] != 1)

    loss_per_element = -np.sum(actual * np.log(predict_clipped), axis=-1)

    masked_loss_per_element = loss_per_element * non_padding_mask

    total_loss = np.sum(masked_loss_per_element)

    num_non_padding_elements = np.sum(non_padding_mask)

    if num_non_padding_elements == 0:
        return 0.0

    return total_loss / num_non_padding_elements

def cross_entropy_prime(predict: np.ndarray, actual: np.ndarray, padding_idx: int = 0) -> np.ndarray:
    softmax = SoftMax()
    predict_softmax = softmax.forward(predict)

    gradient = predict_softmax - actual

    non_padding_mask = (actual[:, :, padding_idx] != 1)
    non_padding_mask_expanded = np.expand_dims(non_padding_mask, axis=-1)

    masked_gradient = gradient * non_padding_mask_expanded

    num_non_padding_elements = np.sum(non_padding_mask)

    if num_non_padding_elements == 0:
        return np.zeros_like(predict)

    return masked_gradient / num_non_padding_elements
