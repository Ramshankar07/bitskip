
import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F



def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard transform recursively.
    
    Args:
        x: Input tensor
        
    Returns:
        Transformed tensor with same shape as input
    """
    # Store original shape
    original_shape = x.shape
    n = x.shape[-1]
    
    if n == 1:
        return x
    
    # Reshape for recursive application
    x = x.view(-1, n)
    half = n // 2
    
    # Split and transform recursively
    x1 = hadamard_transform(x[:, :half])
    x2 = hadamard_transform(x[:, half:])
    
    # Combine results
    result = torch.cat([x1 + x2, x1 - x2], dim=-1)
    
    # Reshape back to original shape
    return result.view(original_shape)


class HBitLinear(nn.Module):
    """
    H-BitLinear layer with Hadamard transformation, Layer Normalization, and quantization.
    
    Args:
        in_features: Input feature dimension (must be power of 2)
        out_features: Output feature dimension (must be power of 2)
        bias: Whether to use bias (default: False)
        device: Device to use
        dtype: Data type to use
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        # Verify dimensions are powers of 2
        if not (in_features & (in_features - 1) == 0):
            raise ValueError(f"in_features must be a power of 2, got {in_features}")
        if not (out_features & (out_features - 1) == 0):
            raise ValueError(f"out_features must be a power of 2, got {out_features}")

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.register_buffer('weight_scale', torch.ones(1, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # Add Layer Normalization
        self.layer_norm = nn.LayerNorm(in_features, **factory_kwargs)

        # Initialize weights using standard initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _weight_quantize(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to ternary values (-1, 0, 1).
        
        Args:
            w: Weight tensor to quantize
            
        Returns:
            Tuple of (quantized weights, scale factor)
        """
        
        
        # Calculate the scaling factor (mean of absolute values)
        scale = w.abs().mean()
        # Store scale for dequantization
        self.weight_scale = scale
        # Ternary quantization: -1, 0, or 1
        w_quantized = torch.zeros_like(w)
        w_quantized[w > 0.5 * scale] = 1.0
        w_quantized[w < -0.5 * scale] = -1.0
        return w_quantized, scale

    def _activation_quantize(self, x: torch.Tensor, bits: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to specified bit width with numerical stability.
        
        Args:
            x: Activation tensor to quantize
            bits: Number of bits for quantization (default: 4 for H-BitLinear)
            
        Returns:
            Tuple of (quantized activations, scale factor)
        """
        if bits is None:
            bits = 4  # Default for H-BitLinear
            
        # Calculate scaling factor with improved numerical stability
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6, max=1e6)
        
        # Scale to target bit range with bounds checking
        max_val = (1 << (bits - 1)) - 1
        scale_factor = max_val / scale
        
        # Clamp scale factor to prevent extreme values
        scale_factor = scale_factor.clamp(min=1e-6, max=1e6)
        
        x_scaled = (x * scale_factor).round().clamp(-max_val, max_val)
        
        # Return quantized values and scale for dequantization
        return x_scaled, scale

    

    def forward(self, x: torch.Tensor, bits: int = 4, ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with Layer Normalization, Hadamard transformation, and quantization.
        
        Args:
            x: Input tensor
            bits: Number of bits for activation quantization (default: 4 for H-BitLinear)
            return_quantization_info: Whether to return quantization loss information
            
        Returns:
            Output tensor, or (output_tensor, quantization_info) if return_quantization_info=True
        """
        # Store original shape for reshaping
        original_shape = x.shape
        
        # Calculate expected output shape
        expected_output_shape = list(original_shape)
        expected_output_shape[-1] = self.out_features  # Change last dimension to out_features
        expected_output_shape = tuple(expected_output_shape)
        
        # Apply Layer Normalization first
        x_ln = self.layer_norm(x)
        
        # Quantize activations before Hadamard transform
        x_q, x_scale = self._activation_quantize(x_ln, bits)
        
        # Apply Hadamard transform to quantized input
        x_h = hadamard_transform(x_q)



        # Quantize weights during forward pass (training with STE)
        if bool(self.training):
            # During training, use Straight-Through Estimator for backprop
            w_q, w_scale = self._weight_quantize(self.weight)
            # Store original weights for backward pass
            w_original = self.weight.data.clone()
            # Replace weights with quantized version for forward pass
            self.weight.data = w_q * w_scale
            

                
            
            # Perform linear transformation
            # Reshape for linear layer: (batch_size * seq_length, in_features) -> (batch_size * seq_length, out_features)
            x_h_flat = x_h.view(-1, x_h.shape[-1])
            output_flat = F.linear(x_h_flat, self.weight, self.bias)
            # Reshape back to expected output shape
            try:
                output = output_flat.view(expected_output_shape)
            except RuntimeError as e:
                print(f"DEBUG: Reshape error in H-BitLinear training")
                print(f"  x_h shape: {x_h.shape}, elements: {x_h.numel()}")
                print(f"  x_h_flat shape: {x_h_flat.shape}, elements: {x_h_flat.numel()}")
                print(f"  output_flat shape: {output_flat.shape}, elements: {output_flat.numel()}")
                print(f"  original_shape: {original_shape}")
                print(f"  expected_output_shape: {expected_output_shape}")
                print(f"  expected elements: {torch.tensor(expected_output_shape).prod().item()}")
                raise e
            # Restore original weights for backward pass
            self.weight.data = w_original
        else:
            # During inference, use quantized weights directly
            w_q, w_scale = self._weight_quantize(self.weight)
            
            
            
            # Matrix multiplication with quantized values
            # Reshape for linear layer: (batch_size * seq_length, in_features) -> (batch_size * seq_length, out_features)
            x_h_flat = x_h.view(-1, x_h.shape[-1])
            output_flat = F.linear(x_h_flat, w_q * w_scale, self.bias)
            # Reshape back to expected output shape
            try:
                output = output_flat.view(expected_output_shape)
            except RuntimeError as e:
                print(f"DEBUG: Reshape error in H-BitLinear inference")
                print(f"  x_h shape: {x_h.shape}, elements: {x_h.numel()}")
                print(f"  x_h_flat shape: {x_h_flat.shape}, elements: {x_h_flat.numel()}")
                print(f"  output_flat shape: {output_flat.shape}, elements: {output_flat.numel()}")
                print(f"  original_shape: {original_shape}")
                print(f"  expected_output_shape: {expected_output_shape}")
                print(f"  expected elements: {torch.tensor(expected_output_shape).prod().item()}")
                raise e

        # Apply inverse Hadamard transform to output
        output = hadamard_transform(output)
        
        return output