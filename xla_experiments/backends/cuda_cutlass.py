"""CUTLASS CUDA backend - custom INT8 GEMM kernels."""

from typing import Dict, Any
import torch
import torch.nn as nn

from .base import Backend, BackendType
from .native_pytorch import NativePyTorchBackend
from ..kernels import is_cutlass_available, int8_gemm


class CutlassBitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation_bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations to INT8
        x_scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        max_val = 127
        x_int8 = (x * max_val / x_scale).round().clamp(-127, 127).to(torch.int8)
        
        # Quantize weights to ternary
        w_scale = self.weight.abs().mean().clamp(min=1e-6)
        w_q = torch.zeros_like(self.weight)
        w_q[self.weight > 0.5 * w_scale] = 1.0
        w_q[self.weight < -0.5 * w_scale] = -1.0
        w_int8 = w_q.to(torch.int8)
        
        # Use CUTLASS kernel or fallback
        if is_cutlass_available() and x.is_cuda:
            out = int8_gemm(x_int8, w_int8.t(), x_scale / max_val, w_scale)
        else:
            out = (x_int8.float() @ w_int8.t().float()) * (x_scale / max_val) * w_scale
        
        return torch.relu(out) ** 2


class CUDACutlassBackend(Backend):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__(device, dtype)
        self._native = NativePyTorchBackend(device, dtype)
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA_CUTLASS
    
    def is_available(self) -> bool:
        return torch.cuda.is_available()
    
    def get_bitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return CutlassBitLinear(in_features, out_features)
    
    def get_hbitlinear(self, in_features: int, out_features: int) -> nn.Module:
        # For HBitLinear, use native with custom forward - Hadamard is already CUDA optimized
        return self._native.get_hbitlinear(in_features, out_features)
    
    def get_model(self, config: Dict[str, Any], use_hadamard: bool = False) -> nn.Module:
        # Full model uses native for now - CUTLASS integration would replace BitLinear layers
        return self._native.get_model(config, use_hadamard)

