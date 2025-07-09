import numpy as np

from .layer import Layer

class Activation(Layer):

    def __init__(self, activation, activation_prime):
        # Y = f(X) for all i
        # Where:
        #   Y is output (size)
        #   f(x) is weights (size -> size)
        #   X is input (size)
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, inputs):
        self.input = inputs
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):

    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):

    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class SoftMax(Layer):

    def forward(self, inputs):
        tmp = np.exp(inputs)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        tmp = np.tile(self.output, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

class ReLU(Activation):

    def __init__(self):
        def relu(x):
            return x * (x > 0)

        def relu_prime(x):
            return int(x > 0)

        super().__init__(relu, relu_prime())
