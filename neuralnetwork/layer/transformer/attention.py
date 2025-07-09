import numpy as np

from neuralnetwork.layer import Layer

def _softmax(inputs):
    tmp = np.exp(inputs)
    return tmp / np.sum(tmp, axis=-1, keepdims=True)

def _softmax_backward(softmax_output, grad_output):
    d_inputs = np.zeros_like(softmax_output)

    for i in range(softmax_output.shape[0]):
        s = softmax_output[i].reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        d_inputs[i] = np.dot(jacobian, grad_output[i])

    return d_inputs

class SingleHeadAttention(Layer):

    def __init__(self, d_model, mask=True):
        super().__init__()
        self.d_model = d_model
        self.mask = mask
        self.weight_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.weight_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.weight_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

        self.Q = None
        self.K = None
        self.V = None
        self.attention_weights = None

    def forward(self, inputs):
        self.input = inputs
        self.Q = np.dot(inputs, self.weight_q)
        self.K = np.dot(inputs, self.weight_k)
        self.V = np.dot(inputs, self.weight_v)

        scores = np.dot(self.Q, self.K.T) / np.sqrt(self.d_model)

        if self.mask:
            seq_len = len(scores)
            mask = np.tril(np.ones((seq_len, seq_len))).T
            scores[mask == 0] = -1e9

        self.attention_weights = _softmax(scores)
        self.output = np.dot(self.attention_weights, self.V)

        return self.output

    def backward(self, output_gradient, learning_rate):
        d_v = np.dot(self.attention_weights.T, output_gradient)
        d_attention_weights = np.dot(output_gradient, self.V.T)

        d_scores = _softmax_backward(self.attention_weights, d_attention_weights)

        if self.mask:
            seq_len = len(d_scores)
            mask = np.tril(np.ones((seq_len, seq_len))).T
            d_scores[mask == 0] = -1e9

        d_q = np.dot(d_scores, self.K) / np.sqrt(self.d_model)
        d_k = np.dot(d_scores.T, self.Q) / np.sqrt(self.d_model)

        d_weight_q = np.dot(self.input.T, d_q)
        d_weight_k = np.dot(self.input.T, d_k)
        d_weight_v = np.dot(self.input.T, d_v)

        input_gradient = np.dot(d_q, self.weight_q.T) + np.dot(d_k, self.weight_k.T) + np.dot(d_v, self.weight_v.T)

        self.weight_q += -learning_rate * d_weight_q
        self.weight_k += -learning_rate * d_weight_k
        self.weight_v += -learning_rate * d_weight_v

        return input_gradient

class MultiHeadAttention(Layer):

    def __init__(self, d_model, num_heads, mask=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_heads = [SingleHeadAttention(self.d_model, mask=mask) for _ in range(num_heads)]
        self.weight_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.concat_output = None

    def forward(self, inputs):
        self.input = inputs
        head_outputs = [head.forward(self.input) for head in self.attention_heads]
        self.concat_output = np.concatenate(head_outputs, axis=-1)
        self.output = np.dot(self.concat_output, self.weight_o)
        return self.output

    def backward(self, output_gradient, learning_rate):
        d_concat_output = np.dot(output_gradient, self.weight_o.T)
        d_weight_output = np.dot(d_concat_output.T, output_gradient)

        split_grads = np.split(d_concat_output, self.num_heads, axis=-1)
        input_gradients = [head.backward(g, learning_rate) for head, g in zip(self.attention_heads, split_grads)]
        input_gradient = sum(input_gradients) / self.num_heads

        self.weight_o += -learning_rate * d_weight_output

        return input_gradient