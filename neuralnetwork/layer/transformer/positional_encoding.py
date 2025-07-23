import numpy as np

from neuralnetwork.layer import Layer

class PositionalEncoding(Layer):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, inputs):
        self.input = inputs
        batch_size, seq_len, d_model = inputs.shape

        position_encoding = self._compute_position_encoding(seq_len)
        # Broadcast to match input shape
        position_encoding = position_encoding[np.newaxis, :, :]  # (1, seq_len, d_model)

        self.output = inputs + position_encoding

        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient

    def _compute_position_encoding(self, seq_len):
        position = np.arange(seq_len)[:, np.newaxis]

        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))[np.newaxis, :]

        angles = position * div_term

        position_encoding = np.zeros((seq_len, self.d_model))

        position_encoding[:, 0::2] = np.sin(angles)
        if self.d_model % 2 == 1:
            position_encoding[:, 1::2] = np.cos(angles[:, :self.d_model // 2])
        else:
            position_encoding[:, 1::2] = np.cos(angles)

        return position_encoding
