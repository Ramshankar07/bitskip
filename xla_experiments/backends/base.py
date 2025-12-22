"""Abstract base for backends."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any
import time
import torch
import torch.nn as nn


class BackendType(Enum):
    NATIVE_PYTORCH = "native_pytorch"
    TORCH_COMPILE = "torch_compile"
    CUDA_CUTLASS = "cuda_cutlass"
    CUDA_XLA = "cuda_xla"
    PURE_XLA = "pure_xla"


class Backend(ABC):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def get_bitlinear(self, in_features: int, out_features: int) -> nn.Module:
        pass
    
    @abstractmethod
    def get_hbitlinear(self, in_features: int, out_features: int) -> nn.Module:
        pass
    
    @abstractmethod
    def get_model(self, config: Dict[str, Any], use_hadamard: bool = False) -> nn.Module:
        pass
    
    def benchmark(self, module: nn.Module, x: torch.Tensor, warmup: int = 10, iters: int = 100) -> Dict[str, float]:
        module = module.to(self.device)
        module.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = module(x)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        if self.device == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                for _ in range(iters):
                    start_evt.record()
                    _ = module(x)
                    end_evt.record()
                    torch.cuda.synchronize()
                    times.append(start_evt.elapsed_time(end_evt))
        else:
            with torch.no_grad():
                for _ in range(iters):
                    t0 = time.perf_counter()
                    _ = module(x)
                    times.append((time.perf_counter() - t0) * 1000)
        
        times.sort()
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": times[0],
            "max_ms": times[-1],
            "p50_ms": times[len(times) // 2],
            "p99_ms": times[int(len(times) * 0.99)],
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if self.device == "cuda" else 0,
        }
