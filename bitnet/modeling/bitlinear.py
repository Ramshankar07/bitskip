"""
BitLinear implementation with quantization and Straight-Through Estimator.
"""

import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Squared ReLU activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    
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
        
        # Initialize weights with conservative scaling to prevent extreme values
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale down weights to prevent extreme quantization scales
        self.weight.data *= 0.1
    
    def _weight_quantize(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to ternary values (-1, 0, 1).
        
        Args:
            w: Weight tensor to quantize
            
        Returns:
            Tuple of (quantized weights, scale factor)
        """    
        # Calculate the scaling factor (mean of absolute values) with stability
        scale = w.abs().mean()
        # Clamp scale to prevent extreme values
        scale = scale.clamp(min=1e-6, max=1e6)
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
            bits: Number of bits for quantization (default: uses self.activation_bits)
            
        Returns:
            Tuple of (quantized activations, scale factor)
        """
        if bits is None:
            bits = self.activation_bits
            
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
    

    def forward(self, x: torch.Tensor, bits: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with quantization-aware training (QAT) using Straight-Through Estimator (STE).
        Does NOT modify parameters in-place, preserving autograd.
        
        Args:
            x: Input tensor
            bits: Number of bits for activation quantization (default: uses self.activation_bits)
        Returns:
            Output tensor
        """
        if bits is None:
            bits = self.activation_bits

        # --- Activation fake-quant with STE ---
        # Per-token dynamic scale
        x_scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        max_val = (1 << (bits - 1)) - 1
        x_int = (x * max_val / x_scale).round().clamp(-max_val, max_val)
        x_q = x_int * x_scale / max_val

        if bool(self.training):
            # STE: use quantized values in forward, full-precision gradients in backward
            x_q = x - x.detach() + x_q.detach()

        # --- Weight fake-quant with STE (ternary) ---
        w_scale = self.weight.abs().mean().clamp(min=1e-6)
        w_q = torch.zeros_like(self.weight)
        w_q[self.weight > 0.5 * w_scale] = 1.0
        w_q[self.weight < -0.5 * w_scale] = -1.0
        w_q = w_q * w_scale

        if bool(self.training):
            # STE for weights
            w_q = self.weight - self.weight.detach() + w_q.detach()

        # Linear with fake-quantized tensors (no in-place .data changes)
        output = F.linear(x_q, w_q, self.bias)

        # Squared ReLU activation
        output = squared_relu(output)
        
        return output