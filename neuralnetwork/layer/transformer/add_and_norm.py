from neuralnetwork.layer import Layer
from neuralnetwork.layer.transformer.normalization import Normalization

class AddAndNorm(Layer):
    """
    Add & Norm layer for Transformer architecture (Pre-Layer Norm variant).
    
    Expected input shape: (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    
    Applies: LayerNorm(x) -> Sublayer -> Add residual connection
    """

    def __init__(self, d_model, sublayer):
        super().__init__()
        self.normalization = Normalization(d_model)
        self.sublayer = sublayer
        self.sub_output = None
        self.norm_output = None

    def forward(self, inputs):
        """
        Process inputs through Add & Norm layer
        """
        self.input = inputs

        # Apply layer normalization
        self.norm_output = self.normalization.forward(self.input)
        
        # Apply sublayer (attention or FFN)
        self.sub_output = self.sublayer.forward(self.norm_output)

        # Add residual connection (vectorized)
        self.output = self.input + self.sub_output

        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Process backward pass through Add & Norm layer
        """
        # Gradient for residual connection (identity function)
        residual_gradient = output_gradient
        
        # Backward through sublayer
        sub_gradient = self.sublayer.backward(output_gradient, learning_rate)
        
        # Backward through normalization
        norm_gradient = self.normalization.backward(sub_gradient, learning_rate)
        
        # Combine gradients (vectorized)
        input_gradient = residual_gradient + norm_gradient

        return input_gradient