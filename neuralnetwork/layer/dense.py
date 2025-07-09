from .layer import Layer
import numpy as np

class Dense(Layer):

    def __init__(self, input_size, output_size):
        # Y = W * X + B
        # Where:
        #   Y is output (output_size * 1)
        #   W is weights (output_size * input_size)
        #   X is input (input_size * 1)
        #   B is bias (output_size * 1)
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, inputs):
        self.input = inputs
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate):
        self.weights += -learning_rate * np.dot(output_gradient, self.input.T)
        self.biases += -learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
