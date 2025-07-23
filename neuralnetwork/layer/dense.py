from .layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.random.randn(1, output_size)

    def forward(self, inputs):
        self.input = inputs
        self.output = np.matmul(inputs, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]

        weight_gradient = np.matmul(self.input.T, output_gradient)

        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        input_gradient = np.matmul(output_gradient, self.weights.T)

        weight_gradient /= batch_size
        bias_gradient /= batch_size

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient

        return input_gradient