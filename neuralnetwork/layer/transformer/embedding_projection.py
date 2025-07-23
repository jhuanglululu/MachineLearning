import numpy as np

# Assuming Layer is defined elsewhere. For completeness, including a placeholder.
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError


class Embedding(Layer):
    """
    Token embedding layer for Transformer architecture.

    Expected input shape: (batch_size, seq_len) - token IDs
    Output shape: (batch_size, seq_len, d_model) - embedded vectors
    """

    def __init__(self, vocab_size, d_model, shared_weights=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        if shared_weights is not None:
            self.weights = shared_weights
            self._owns_weights = False
            self._shared_weights_ref = shared_weights
        else:
            self.weights = np.random.normal(0, np.sqrt(1.0 / d_model), (d_model, vocab_size))
            self._owns_weights = True
            self._shared_weights_ref = None

    def forward(self, token_ids):
        self.input = token_ids
        self.output = self.weights.T[token_ids] * np.sqrt(self.d_model)
        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, seq_len = self.input.shape

        # Scale gradients by sqrt(d_model) to account for forward scaling
        scaled_gradient = output_gradient * np.sqrt(self.d_model)
        token_one_hot = np.eye(self.vocab_size)[self.input.flatten()].reshape(batch_size, seq_len, self.vocab_size)
        grad_weights = np.einsum('bsd,bsv->dv', scaled_gradient, token_one_hot)

        # Update shared weights if they are shared
        if self._shared_weights_ref is not None:
            self._shared_weights_ref -= learning_rate * grad_weights
        elif self._owns_weights:
            self.weights -= learning_rate * grad_weights

        return None

class Projection(Layer):
    """
    Output projection layer for Transformer architecture.

    Expected input shape: (batch_size, seq_len, d_model) - hidden states
    Output shape: (batch_size, seq_len, vocab_size) - logits over vocabulary
    """

    def __init__(self, d_model, vocab_size, shared_weights=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        if shared_weights is not None:
            self.weights = shared_weights
            self._shared_weights_ref = shared_weights
        else:
            self.weights = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, vocab_size))
            self._shared_weights_ref = None

        self.bias = np.zeros((vocab_size, 1))

    def forward(self, hidden_states):
        self.input = hidden_states
        self.output = np.matmul(hidden_states, self.weights) + self.bias.flatten()
        return self.output


    def backward(self, output_gradient, learning_rate):
        d_weights = np.einsum('bsd,bsv->dv', self.input, output_gradient)

        d_bias = np.sum(output_gradient, axis=(0, 1), keepdims=True).reshape(self.vocab_size, 1)

        input_gradient = np.matmul(output_gradient, self.weights.T)
        if self._shared_weights_ref is None: # Only update if weights are not explicitly shared
            self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return input_gradient

def create_shared_embedding_projection(vocab_size, d_model):
    shared_weights = np.random.normal(0, np.sqrt(1.0 / d_model), (d_model, vocab_size))
    embedding = Embedding(vocab_size, d_model, shared_weights)
    projection = Projection(d_model, vocab_size, shared_weights)

    return embedding, projection