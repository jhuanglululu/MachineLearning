import numpy as np

from neuralnetwork.layer import Layer

class Normalization(Layer):
    """
    Layer Normalization for Transformer architecture.
    
    Expected input shape: (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    
    Normalizes across the feature dimension (d_model) for each position independently.
    """

    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon

        # Learnable parameters for layer normalization
        self.gamma = np.ones(d_model)  # Scale parameter
        self.beta = np.zeros(d_model)  # Shift parameter

        # Cache for backward pass
        self.mean = None
        self.variance = None
        self.std = None
        self.normalized = None

    def forward(self, inputs):
        self.input = inputs
        batch_size, seq_len, d_model = inputs.shape

        # Compute mean and variance across the feature dimension (vectorized)
        self.mean = np.mean(inputs, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        self.variance = np.var(inputs, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        self.std = np.sqrt(self.variance + self.epsilon)  # (batch_size, seq_len, 1)

        # Normalize (vectorized)
        self.normalized = (inputs - self.mean) / self.std  # (batch_size, seq_len, d_model)

        # Apply learnable scale and shift (vectorized)
        self.output = self.gamma * self.normalized + self.beta  # (batch_size, seq_len, d_model)

        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, seq_len, d_model = self.input.shape
        N = d_model  # Number of features

        # Gradient for gamma and beta (sum over batch and seq_len)
        d_gamma = np.sum(output_gradient * self.normalized, axis=(0, 1))
        d_beta = np.sum(output_gradient, axis=(0, 1))

        # Gradient w.r.t normalized values
        d_normalized = output_gradient * self.gamma

        # Numerically stable layer norm backward pass
        std_clamped = np.maximum(self.std, self.epsilon)
        inv_std = 1.0 / std_clamped

        sum_d_normalized = np.sum(d_normalized, axis=-1, keepdims=True)
        sum_d_norm_times_norm = np.sum(d_normalized * self.normalized, axis=-1, keepdims=True)

        input_gradient = (N * d_normalized - sum_d_normalized - self.normalized * sum_d_norm_times_norm) * inv_std / N

        # Clip gradients to prevent overflow (kept for the input_gradient passed to previous layer)
        input_gradient = np.clip(input_gradient, -10.0, 10.0)

        # Update learnable parameters without direct clipping on d_gamma and d_beta here
        # Assuming global clipping is handled elsewhere, or not needed for these specific updates.
        self.gamma -= learning_rate * d_gamma / (batch_size * seq_len)  # Average gradients for update
        self.beta -= learning_rate * d_beta / (batch_size * seq_len)  # Average gradients for update

        return input_gradient
