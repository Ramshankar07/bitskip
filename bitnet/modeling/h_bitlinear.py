
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
    # Prefer CUDA kernel if available
    try:
        from .kernels import fwht as fwht_cuda, is_available as kernels_available
        if x.is_cuda and kernels_available():
            return fwht_cuda(x)
    except Exception:
        pass

    # Fallback: iterative FWHT on last dimension
    original_shape = x.shape
    n = x.shape[-1]
    if (n & (n - 1)) != 0:
        raise ValueError(f"FWHT requires power-of-two length, got {n}")
    y = x.contiguous().view(-1, n)
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            a = y[:, start:start + h]
            b = y[:, start + h:start + 2 * h]
            y[:, start:start + h] = a + b
            y[:, start + h:start + 2 * h] = a - b
        h <<= 1
    return y.view(original_shape)


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
        activation_bits: int = 4,
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
        self.activation_bits = activation_bits

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

    

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with Layer Normalization, Hadamard transformation, and QAT using STE.
        Does not modify parameters in-place.
        """
        # Store original shape for reshaping
        original_shape = x.shape
        expected_output_shape = list(original_shape)
        expected_output_shape[-1] = self.out_features
        expected_output_shape = tuple(expected_output_shape)

        # LayerNorm
        x_ln = self.layer_norm(x)

        # Activation fake-quant (per-token) with STE
        bits = self.activation_bits
        x_scale = x_ln.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        max_val = (1 << (bits - 1)) - 1
        x_int = (x_ln * max_val / x_scale).round().clamp(-max_val, max_val)
        x_q = x_int * x_scale / max_val
        if bool(self.training):
            x_q = x_ln - x_ln.detach() + x_q.detach()

        # Hadamard transform on fake-quant activations
        x_h = hadamard_transform(x_q)

        # Weight fake-quant (ternary) with STE
        w_scale = self.weight.abs().mean().clamp(min=1e-6)
        w_q = torch.zeros_like(self.weight)
        w_q[self.weight > 0.5 * w_scale] = 1.0
        w_q[self.weight < -0.5 * w_scale] = -1.0
        w_q = w_q * w_scale
        if bool(self.training):
            w_q = self.weight - self.weight.detach() + w_q.detach()

        # Linear on flattened last-dim, then reshape
        x_h_flat = x_h.view(-1, x_h.shape[-1])
        output_flat = F.linear(x_h_flat, w_q, self.bias)
        try:
            output = output_flat.view(expected_output_shape)
        except RuntimeError as e:
            print(f"DEBUG: Reshape error in H-BitLinear")
            print(f"  x_h shape: {x_h.shape}, elements: {x_h.numel()}")
            print(f"  x_h_flat shape: {x_h_flat.shape}, elements: {x_h_flat.numel()}")
            print(f"  output_flat shape: {output_flat.shape}, elements: {output_flat.numel()}")
            print(f"  original_shape: {original_shape}")
            print(f"  expected_output_shape: {expected_output_shape}")
            print(f"  expected elements: {torch.tensor(expected_output_shape).prod().item()}")
            raise e

        # Inverse Hadamard
        output = hadamard_transform(output)
        return output