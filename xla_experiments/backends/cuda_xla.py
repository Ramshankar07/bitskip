"""CUDA + XLA backend - uses PyTorch/XLA or JAX with custom CUDA ops."""

from typing import Dict, Any
import torch
import torch.nn as nn

from .base import Backend, BackendType
from .native_pytorch import NativePyTorchBackend

_torch_xla_available = False
try:    
    import torch_xla
    import torch_xla.core.xla_model as xm
    _torch_xla_available = True
except ImportError:
    pass


class CUDAXLABackend(Backend):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__(device, dtype)
        self._native = NativePyTorchBackend(device, dtype)
        if _torch_xla_available:
            self.xla_device = xm.xla_device()
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA_XLA
    
    def is_available(self) -> bool:
        return _torch_xla_available
    
    def _to_xla(self, module: nn.Module) -> nn.Module:
        if _torch_xla_available:
            return module.to(self.xla_device)
        return module
    
    def get_bitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return self._to_xla(self._native.get_bitlinear(in_features, out_features))
    
    def get_hbitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return self._to_xla(self._native.get_hbitlinear(in_features, out_features))
    
    def get_model(self, config: Dict[str, Any], use_hadamard: bool = False) -> nn.Module:
        return self._to_xla(self._native.get_model(config, use_hadamard))
    
    def benchmark(self, module: nn.Module, x: torch.Tensor, warmup: int = 10, iters: int = 100) -> Dict[str, float]:
        if not _torch_xla_available:
            return self._native.benchmark(module, x, warmup, iters)
        
        x = x.to(self.xla_device)
        module = module.to(self.xla_device)
        module.eval()
        
        import time
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = module(x)
                xm.mark_step()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = module(x)
                xm.mark_step()
                times.append((time.perf_counter() - t0) * 1000)
        
        times.sort()
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": times[0],
            "max_ms": times[-1],
            "p50_ms": times[len(times) // 2],
            "p99_ms": times[int(len(times) * 0.99)],
            "peak_memory_mb": 0,
        }

