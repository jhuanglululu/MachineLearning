import numpy as np

from .layer import Layer

class Activation(Layer):
    """
    Base activation layer for neural networks.
    
    Expected input shape: Any shape (batch_size, ...)
    Output shape: Same as input shape
    
    Applies element-wise activation function and its derivative.
    """

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
    """
    Hyperbolic tangent activation function.
    
    Expected input shape: Any shape (batch_size, ...)
    Output shape: Same as input shape
    
    Applies tanh activation element-wise.
    """

    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    
    Expected input shape: Any shape (batch_size, ...)
    Output shape: Same as input shape
    
    Applies sigmoid activation element-wise with numerical stability.
    """

    def __init__(self):
        def sigmoid(x):
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            x = np.clip(x, -500, 500)
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class SoftMax(Layer):
    """
    SoftMax activation function for classification.

    Expected input shape: (batch_size, seq_len, vocab_size)
    Output shape: Same as input shape

    Applies softmax along the last dimension (vocab_size) using vectorized NumPy operations.
    """

    def forward(self, inputs):
        self.input = inputs
        stable_inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_inputs = np.exp(stable_inputs)
        sum_exp = np.sum(exp_inputs, axis=-1, keepdims=True)
        self.output = exp_inputs / sum_exp

        return self.output

    def backward(self, output_gradient, learning_rate):
        s_dot_grad = np.sum(self.output * output_gradient, axis=-1, keepdims=True)

        input_gradient = self.output * (output_gradient - s_dot_grad)

        return input_gradient

class ReLU(Activation):

    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(x.dtype)

        super().__init__(relu, relu_prime)
