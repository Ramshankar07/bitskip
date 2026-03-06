
import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernels import fwht


def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** 2


def _next_power_of_2(n: int) -> int:
    """Return the next power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    if (n & (n - 1)) == 0:
        return n  # Already a power of 2
    return 1 << (n - 1).bit_length()


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard transform with CUDA acceleration (falls back to CPU)."""
    return fwht(x)


class HBitLinear(nn.Module):
    """
    H-BitLinear layer with Hadamard transformation, Layer Normalization, and quantization.
    
    Supports arbitrary dimensions by automatically padding to the next power of 2 for FWHT.
    
    Args:
        in_features: Input feature dimension (will be padded to next power of 2 if needed)
        out_features: Output feature dimension (will be padded to next power of 2 if needed)
        bias: Whether to use bias (default: False)
        device: Device to use
        dtype: Data type to use
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 4,
        disable_quantization: bool = False,
        disable_hadamard: bool = False,
        init_scale: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.disable_quantization = disable_quantization
        self.disable_hadamard = disable_hadamard

        # Padded input dimension for FWHT (must be power of 2)
        self.in_features_padded = _next_power_of_2(in_features)
        self.in_pad = self.in_features_padded - in_features

        # FWHT only on input; weights map from padded-input → out_features
        weight_in = in_features if disable_hadamard else self.in_features_padded
        self.weight = nn.Parameter(torch.empty((out_features, weight_in), **factory_kwargs))
        self.register_buffer('weight_scale', torch.ones(1, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # Layer Normalization on original (unpadded) dimension
        self.layer_norm = nn.LayerNorm(self.in_features, **factory_kwargs)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if init_scale != 1.0:
            self.weight.data *= init_scale

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
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5, max=1e5)
        
        # Scale to target bit range with bounds checking
        max_val = (1 << (bits - 1)) - 1
        scale_factor = max_val / scale
        
        # Clamp scale factor to prevent extreme values
        scale_factor = scale_factor.clamp(min=1e-5, max=1e5)
        
        x_scaled = (x * scale_factor).round().clamp(-max_val, max_val)
        
        # Return quantized values and scale for dequantization
        return x_scaled, scale.clamp(min=1e-5)

    

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with Layer Normalization, Hadamard transformation, and QAT using STE.
        Automatically handles padding/unpadding for non-power-of-2 dimensions.
        Does not modify parameters in-place.
        """
        # Store original shape for reshaping
        original_shape = x.shape
        expected_output_shape = list(original_shape)
        expected_output_shape[-1] = self.out_features
        expected_output_shape = tuple(expected_output_shape)

        # LayerNorm on original (unpadded) dimensions
        x_ln = self.layer_norm(x)

        if self.disable_hadamard:
            x_flat = x_ln.view(-1, x_ln.shape[-1])
            output_flat = F.linear(x_flat, self.weight, self.bias)
            return squared_relu(output_flat.view(*original_shape[:-1], self.out_features))

        # Pad to power of 2 after LayerNorm
        if self.in_pad > 0:
            x_padded = F.pad(x_ln, (0, self.in_pad), mode='constant', value=0.0)
        else:
            x_padded = x_ln

        # Hadamard transform (smooths distribution for better quantization)
        x_h = hadamard_transform(x_padded)

        if self.disable_quantization:
            x_q = x_h
            w_q = self.weight
        else:
            # Activation fake-quant (per-token) with STE
            bits = self.activation_bits
            x_scale = x_h.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5, max=1e5)
            max_val = float((1 << (bits - 1)) - 1)
            x_int = (x_h * max_val / x_scale).round().clamp(-max_val, max_val)
            x_q = x_int * x_scale / max_val
            if bool(self.training):
                x_q = x_h + (x_q - x_h).detach()

            # Weight fake-quant (ternary) with STE
            w_scale = self.weight.abs().mean().clamp(min=1e-5, max=1e5)
            w_q = torch.zeros_like(self.weight)
            w_q[self.weight > 0.5 * w_scale] = 1.0
            w_q[self.weight < -0.5 * w_scale] = -1.0
            w_q = w_q * w_scale
            if bool(self.training):
                w_q = self.weight + (w_q - self.weight).detach()

        # Linear on flattened last-dim, then reshape
        x_q_flat = x_q.view(-1, x_q.shape[-1])
        output_flat = F.linear(x_q_flat, w_q, self.bias)
        output = output_flat.view(*original_shape[:-1], self.out_features)

        return squared_relu(output)