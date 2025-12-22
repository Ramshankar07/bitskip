"""Configuration for XLA backend experiments."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List
import torch


class BackendType(Enum):
    NATIVE_PYTORCH = "native_pytorch"
    TORCH_COMPILE = "torch_compile"
    CUDA_CUTLASS = "cuda_cutlass"
    CUDA_XLA = "cuda_xla"
    PURE_XLA = "pure_xla"


@dataclass
class BenchmarkConfig:
    backends: List[BackendType] = field(default_factory=lambda: list(BackendType))
    
    # Layer test sizes
    layer_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 32])
    
    # Model config (matches bitnet defaults)
    hidden_size: int = 2048
    num_layers: int = 16
    num_heads: int = 16
    vocab_size: int = 128256
    seq_length: int = 512
    
    # Benchmark params
    warmup: int = 10
    iterations: int = 100
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
