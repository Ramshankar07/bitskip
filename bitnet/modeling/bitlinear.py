"""
BitLinear implementation with quantization and Straight-Through Estimator.
"""

import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import (
    bitnet_kernels,
    ternary_quantize_cuda,
    activation_quantize_cuda,
    bitlinear_forward_cuda,
    squared_relu_cuda
)


def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Squared ReLU activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    if bitnet_kernels.is_available:
        return squared_relu_cuda(x)
    return torch.relu(x) ** 2


class BitLinear(nn.Module):
    """
    BitLinear layer with weight and activation quantization.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to use bias (default: False)
        activation_bits: Number of bits for activation quantization (default: 8)
        device: Device to use
        dtype: Data type to use
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        
        # Initialize weights as standard nn.Linear for training
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.register_buffer('weight_scale', torch.ones(1, **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
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
        if bitnet_kernels.is_available:
            return ternary_quantize_cuda(w)
            
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
        Quantize activations to specified bit width.
        
        Args:
            x: Activation tensor to quantize
            bits: Number of bits for quantization (default: uses self.activation_bits)
            
        Returns:
            Tuple of (quantized activations, scale factor)
        """
        if bits is None:
            bits = self.activation_bits
            
        if bitnet_kernels.is_available:
            return activation_quantize_cuda(x, bits)
            
        # Calculate scaling factor (max of absolute values per token)
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        # Scale to target bit range
        max_val = (1 << (bits - 1)) - 1
        x_scaled = (x * (max_val / scale)).round().clamp(-max_val, max_val)
        # Return quantized values and scale for dequantization
        return x_scaled, scale
    
    def compute_quantization_loss(self, original_weights: torch.Tensor, quantized_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute quantization reconstruction error: ||W - W̃||²
        
        Args:
            original_weights: Original full-precision weights
            quantized_weights: Quantized weights
            
        Returns:
            Quantization loss tensor
        """
        return F.mse_loss(original_weights, quantized_weights)
    
    def forward(self, x: torch.Tensor, bits: Optional[int] = None, return_quantization_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            bits: Number of bits for activation quantization (default: uses self.activation_bits)
            return_quantization_info: Whether to return quantization loss information
            
        Returns:
            Output tensor, or (output_tensor, quantization_info) if return_quantization_info=True
        """
        if bits is None:
            bits = self.activation_bits
        
        # Quantize input activations for both training and inference
        x_q, x_scale = self._activation_quantize(x, bits)
        
        quantization_info = {}
        
        if self.training:
            # During training, use Straight-Through Estimator for backprop
            w_q, w_scale = self._weight_quantize(self.weight)
            w_original = self.weight.data.clone()
            self.weight.data = w_q * w_scale
            
            # Debug scaling factors
            print(f"DEBUG: BitLinear - w_scale: {w_scale:.6f}, x_scale: {x_scale:.6f}, ratio: {w_scale/x_scale:.6f}")
            print(f"DEBUG: BitLinear - x_q stats: min={x_q.min():.4f}, max={x_q.max():.4f}, mean={x_q.mean():.4f}")
            print(f"DEBUG: BitLinear - weight stats: min={self.weight.min():.4f}, max={self.weight.max():.4f}, mean={self.weight.mean():.4f}")
            
            # Check for extreme scaling
            if w_scale / x_scale > 1000:
                print(f"WARNING: Extreme scaling ratio detected! w_scale/x_scale = {w_scale/x_scale:.2f}")
            
            # Compute quantization loss if requested
            if return_quantization_info:
                quant_loss = self.compute_quantization_loss(w_original, w_q * w_scale)
                quantization_info['weight_quantization_loss'] = quant_loss
                quantization_info['original_weights'] = w_original
                quantization_info['quantized_weights'] = w_q * w_scale
                
            
            # Perform linear transformation with quantized input
            if bitnet_kernels.is_available:
                output = bitlinear_forward_cuda(x_q, self.weight, self.bias) / x_scale
            else:
                output = F.linear(x_q, self.weight, self.bias) / x_scale
            
            print(f"DEBUG: BitLinear - output before squared_relu: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"ERROR: NaN/Inf detected in BitLinear output before squared_relu!")
                print(f"ERROR: NaN count: {torch.isnan(output).sum()}")
                print(f"ERROR: Inf count: {torch.isinf(output).sum()}")
            
            self.weight.data = w_original
        else:
            # During inference, use quantized weights directly
            w_q, w_scale = self._weight_quantize(self.weight)
            
            # Compute quantization loss if requested (for evaluation)
            if return_quantization_info:
                quant_loss = self.compute_quantization_loss(self.weight, w_q * w_scale)
                quantization_info['weight_quantization_loss'] = quant_loss
                quantization_info['original_weights'] = self.weight
                quantization_info['quantized_weights'] = w_q * w_scale
            
            # Matrix multiplication with quantized values
            if bitnet_kernels.is_available:
                output = bitlinear_forward_cuda(x_q, w_q * w_scale, self.bias) / x_scale
            else:
                output = F.linear(x_q, w_q * w_scale, self.bias) / x_scale
        
        # Apply Squared ReLU activation
        output = squared_relu(output)
        
        if return_quantization_info:
            return output, quantization_info
        else:
            return output 