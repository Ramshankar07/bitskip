"""torch.compile backend - wraps existing modules with TorchDynamo."""

from typing import Dict, Any
import torch
import torch.nn as nn

from .base import Backend, BackendType
from .native_pytorch import NativePyTorchBackend


class TorchCompileBackend(Backend):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, mode: str = "max-autotune"):
        super().__init__(device, dtype)
        self.mode = mode
        self._native = NativePyTorchBackend(device, dtype)
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.TORCH_COMPILE
    
    def is_available(self) -> bool:
        return hasattr(torch, 'compile')
    
    def _compile(self, module: nn.Module) -> nn.Module:
        return torch.compile(module, mode=self.mode)
    
    def get_bitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return self._compile(self._native.get_bitlinear(in_features, out_features))
    
    def get_hbitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return self._compile(self._native.get_hbitlinear(in_features, out_features))
    
    def get_model(self, config: Dict[str, Any], use_hadamard: bool = False) -> nn.Module:
        return self._compile(self._native.get_model(config, use_hadamard))

