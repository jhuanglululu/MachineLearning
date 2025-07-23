import numpy as np
from .layer import Layer

class DebugLayer(Layer):
    """
    Debug layer that prints input and gradient information during forward and backward passes.
    This layer passes through all data unchanged but logs statistics for debugging.
    """
    
    def __init__(self, name="Debug", print_shapes=True, print_stats=True, print_samples=False, max_elements=5):
        """
        Initialize debug layer
        
        Args:
            name: Name for this debug layer (for identification)
            print_shapes: Whether to print tensor shapes
            print_stats: Whether to print min/max/mean statistics
            print_samples: Whether to print sample values
            max_elements: Maximum number of sample elements to print
        """
        super().__init__()
        self.name = name
        self.print_shapes = print_shapes
        self.print_stats = print_stats
        self.print_samples = print_samples
        self.max_elements = max_elements
        self.forward_count = 0
        self.backward_count = 0
    
    def forward(self, inputs):
        """Forward pass - log input information and pass through unchanged"""
        self.input = inputs
        self.forward_count += 1
        
        print(f"\n=== {self.name} - FORWARD PASS #{self.forward_count} ===")
        self._print_tensor_info("INPUT", inputs)
        
        # Pass through unchanged
        self.output = inputs
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass - log gradient information and pass through unchanged"""
        self.backward_count += 1
        
        print(f"\n=== {self.name} - BACKWARD PASS #{self.backward_count} ===")
        print(f"Learning Rate: {learning_rate}")
        self._print_tensor_info("OUTPUT_GRADIENT", output_gradient)
        
        # Pass through unchanged
        input_gradient = output_gradient
        return input_gradient
    
    def _print_tensor_info(self, label, tensor):
        """Print comprehensive tensor information"""
        if tensor is None:
            print(f"{label}: None")
            return
            
        print(f"{label}:")
        
        # Shape information
        if self.print_shapes:
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Size: {tensor.size}")
        
        # Statistical information  
        if self.print_stats:
            print(f"  Min: {np.min(tensor):.6f}")
            print(f"  Max: {np.max(tensor):.6f}")
            print(f"  Mean: {np.mean(tensor):.6f}")
            print(f"  Std: {np.std(tensor):.6f}")
            
            # Check for problematic values
            num_nan = np.count_nonzero(np.isnan(tensor))
            num_inf = np.count_nonzero(np.isinf(tensor))
            num_zero = np.count_nonzero(tensor == 0)
            
            if num_nan > 0:
                print(f"  ⚠️  NaN values: {num_nan}")
            if num_inf > 0:
                print(f"  ⚠️  Inf values: {num_inf}")
            if num_zero > tensor.size * 0.9:  # More than 90% zeros
                print(f"  ⚠️  Mostly zeros: {num_zero}/{tensor.size}")
        
        # Sample values
        if self.print_samples and tensor.size > 0:
            flat_tensor = tensor.flatten()
            sample_size = min(self.max_elements, len(flat_tensor))
            sample_indices = np.linspace(0, len(flat_tensor)-1, sample_size, dtype=int)
            sample_values = flat_tensor[sample_indices]
            print(f"  Sample values: {sample_values}")
            
            # For 3D tensors (batch, seq, features), show one example
            if len(tensor.shape) == 3 and self.print_samples:
                print(f"  First element [0,0,:5]: {tensor[0,0,:min(5, tensor.shape[2])]}")


class ShapeDebugLayer(DebugLayer):
    """Lightweight debug layer that only prints shapes"""
    
    def __init__(self, name="ShapeDebug"):
        super().__init__(name=name, print_shapes=True, print_stats=False, print_samples=False)


class StatDebugLayer(DebugLayer):
    """Debug layer that prints shapes and statistics but no sample values"""
    
    def __init__(self, name="StatDebug"):
        super().__init__(name=name, print_shapes=True, print_stats=True, print_samples=False)


class FullDebugLayer(DebugLayer):
    """Full debug layer that prints everything"""
    
    def __init__(self, name="FullDebug", max_elements=10):
        super().__init__(name=name, print_shapes=True, print_stats=True, 
                        print_samples=True, max_elements=max_elements)