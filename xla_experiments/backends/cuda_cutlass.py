# CUTLASS/cuTile CUDA backend 

from typing import Dict, Any
import torch
import torch.nn as nn
from .base import Backend, BackendType
from .native_pytorch import NativePyTorchBackend
from ..kernels import is_cutile_available, is_cuda_kernel_available, int8_gemm, ternary_gemm


class CutileBitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation_bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        max_val = 127
        x_int8 = (x * max_val / x_scale).round().clamp(-127, 127).to(torch.int8)
        
        w_scale = self.weight.abs().mean().clamp(min=1e-6)
        w_q = torch.zeros_like(self.weight)
        w_q[self.weight > 0.5 * w_scale] = 1.0
        w_q[self.weight < -0.5 * w_scale] = -1.0
        
        out = ternary_gemm(x_int8.float() * (x_scale / max_val), w_q, w_scale)
        
        return torch.relu(out) ** 2


class CUDACutlassBackend(Backend):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__(device, dtype)
        self._native = NativePyTorchBackend(device, dtype)
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA_CUTLASS
    
    def is_available(self) -> bool:
        return torch.cuda.is_available() and (is_cutile_available() or is_cuda_kernel_available())
    
    def get_bitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return CutileBitLinear(in_features, out_features)
    
    def get_hbitlinear(self, in_features: int, out_features: int) -> nn.Module:
        # HBitLinear already uses CUDA-optimized Hadamard
        return self._native.get_hbitlinear(in_features, out_features)
    
    def get_model(self, config: Dict[str, Any], use_hadamard: bool = False) -> nn.Module:
        return self._native.get_model(config, use_hadamard)
