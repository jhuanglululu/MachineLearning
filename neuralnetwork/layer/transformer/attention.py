import numpy as np

from neuralnetwork.layer import Layer
from neuralnetwork.layer.activation import SoftMax

class SingleHeadAttention(Layer):
    """
    Attention for Transformer architecture.

    Expected input shape: (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, mask=True):
        super().__init__()
        self.d_model = d_model
        self.mask = mask
        self.weight_q = np.random.normal(0, np.sqrt(1.0 / d_model), (d_model, d_model))
        self.weight_k = np.random.normal(0, np.sqrt(1.0 / d_model), (d_model, d_model))
        self.weight_v = np.random.normal(0, np.sqrt(1.0 / d_model), (d_model, d_model))

        self.Q = None
        self.K = None
        self.V = None
        self.attention_weights = None
        self.softmax = SoftMax()

    def forward(self, inputs):
        self.input = inputs
        batch_size, seq_len, d_model = inputs.shape
        self.Q = np.matmul(inputs, self.weight_q)
        self.K = np.matmul(inputs, self.weight_k)
        self.V = np.matmul(inputs, self.weight_v)

        scores = np.einsum('bsd,btd->bst', self.Q, self.K) / np.sqrt(self.d_model)

        if self.mask:
            # Causal mask (look-ahead mask)
            seq_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[:, seq_mask] = -1e9

        self.attention_weights = self.softmax.forward(scores)

        self.output = np.matmul(self.attention_weights, self.V)

        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, seq_len, d_model = self.input.shape
        d_v = np.einsum('bts,bsd->btd', self.attention_weights.transpose(0, 2, 1), output_gradient)
        d_attention_weights = np.einsum('bsd,bdt->bst', output_gradient, self.V.transpose(0, 2, 1))

        d_scores = self.softmax.backward(d_attention_weights, 0.0)  # Softmax doesn't use learning_rate

        if self.mask:
            seq_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            d_scores[:, seq_mask] = 0
            
        d_q = np.matmul(d_scores, self.K) / np.sqrt(self.d_model)
        d_k = np.matmul(d_scores.transpose(0, 2, 1), self.Q) / np.sqrt(self.d_model)

        d_weight_q = np.einsum('bij,bjk->ik', self.input.transpose(0, 2, 1), d_q)
        d_weight_k = np.einsum('bij,bjk->ik', self.input.transpose(0, 2, 1), d_k)
        d_weight_v = np.einsum('bij,bjk->ik', self.input.transpose(0, 2, 1), d_v)

        input_gradient = (np.matmul(d_q, self.weight_q.T) +
                          np.matmul(d_k, self.weight_k.T) +
                          np.matmul(d_v, self.weight_v.T))

        self.weight_q -= learning_rate * d_weight_q
        self.weight_k -= learning_rate * d_weight_k
        self.weight_v -= learning_rate * d_weight_v

        return input_gradient

class MultiHeadAttention(Layer):
    """
    Multi-Head Attention for Transformer architecture using multiple SingleHeadAttention instances.

    Expected input shape: (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, num_heads, mask=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            
        self.d_k = d_model // num_heads

        self.attention_heads = [SingleHeadAttention(self.d_k, mask=mask) for _ in range(num_heads)]

        self.weight_o = np.random.normal(0, np.sqrt(1.0 / d_model), (d_model, d_model))

        self.head_outputs = None
        self.concat_output = None

    def forward(self, inputs):
        self.input = inputs
        batch_size, seq_len, d_model = inputs.shape

        # Split input into heads along feature dimension
        input_heads = inputs.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        self.head_outputs = []
        for h in range(self.num_heads):
            head_input = input_heads[:, :, h, :]  # (batch_size, seq_len, d_k)
            head_output = self.attention_heads[h].forward(head_input)
            self.head_outputs.append(head_output)

        self.concat_output = np.concatenate(self.head_outputs, axis=-1)
        self.output = np.matmul(self.concat_output, self.weight_o)

        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, seq_len, d_model = self.input.shape

        d_concat_output = np.matmul(output_gradient, self.weight_o.T)

        d_weight_o = np.einsum('bsd,bsh->dh', self.concat_output, output_gradient)
        head_grad_size = self.d_k
        head_gradients = []
        for h in range(self.num_heads):
            start_idx = h * head_grad_size
            end_idx = (h + 1) * head_grad_size
            head_gradients.append(d_concat_output[:, :, start_idx:end_idx])

        input_gradient_heads = np.zeros((batch_size, seq_len, self.num_heads, self.d_k))

        for h in range(self.num_heads):
            head_grad = head_gradients[h]
            head_input_grad = self.attention_heads[h].backward(head_grad, learning_rate)
            input_gradient_heads[:, :, h, :] = head_input_grad
            
        input_gradient = input_gradient_heads.reshape(batch_size, seq_len, d_model)

        self.weight_o -= learning_rate * d_weight_o

        return input_gradient
