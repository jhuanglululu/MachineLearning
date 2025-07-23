import numpy as np
from neuralnetwork.layer import Layer
from neuralnetwork.layer import Dense, ReLU

class TransformerFFN(Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.dense1 = Dense(d_model, d_ff)
        self.dense2 = Dense(d_ff, d_model)
        self.activation = ReLU()

    def forward(self, inputs):
        self.input = inputs
        batch_size, seq_len, d_model = inputs.shape

        reshaped_inputs = inputs.reshape(-1, d_model)

        dense1_output_flat = self.dense1.forward(reshaped_inputs)
        self.dense1_outputs = dense1_output_flat.reshape(batch_size, seq_len, self.d_ff)

        activation_output_flat = self.activation.forward(dense1_output_flat)
        self.activation_outputs = activation_output_flat.reshape(batch_size, seq_len, self.d_ff)

        output_flat = self.dense2.forward(activation_output_flat)
        self.output = output_flat.reshape(batch_size, seq_len, d_model)

        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, seq_len, d_model = self.input.shape

        grad_flat = output_gradient.reshape(-1, d_model)

        grad_after_dense2 = self.dense2.backward(grad_flat, learning_rate)

        grad_after_activation = self.activation.backward(grad_after_dense2, learning_rate)

        grad_after_dense1 = self.dense1.backward(grad_after_activation, learning_rate)

        input_gradient = grad_after_dense1.reshape(batch_size, seq_len, d_model)

        return input_gradient