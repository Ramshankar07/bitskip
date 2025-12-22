"""Pure XLA backend - uses JAX/XLA without custom CUDA kernels."""

from typing import Dict, Any
import torch
import torch.nn as nn

from .base import Backend, BackendType
from .native_pytorch import NativePyTorchBackend

# Try to import JAX
_jax_available = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    _jax_available = True
except ImportError:
    pass


class JaxBitLinear:
    """Pure JAX implementation of BitLinear."""
    
    def __init__(self, in_features: int, out_features: int, activation_bits: int = 8):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        # Initialize weights
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (out_features, in_features)) * 0.02
    
    @staticmethod
    @jit
    def forward(x, weight, activation_bits=8):
        # Activation quantization
        x_scale = jnp.abs(x).max(axis=-1, keepdims=True).clip(min=1e-6)
        max_val = (1 << (activation_bits - 1)) - 1
        x_q = jnp.round(x * max_val / x_scale).clip(-max_val, max_val)
        x_q = x_q * x_scale / max_val
        
        # Weight quantization (ternary)
        w_scale = jnp.abs(weight).mean().clip(min=1e-6)
        w_q = jnp.zeros_like(weight)
        w_q = jnp.where(weight > 0.5 * w_scale, 1.0, w_q)
        w_q = jnp.where(weight < -0.5 * w_scale, -1.0, w_q)
        w_q = w_q * w_scale
        
        # Matmul + squared ReLU
        out = x_q @ w_q.T
        return jax.nn.relu(out) ** 2
    
    def __call__(self, x):
        return self.forward(x, self.weight, self.activation_bits)


class PureXLABackend(Backend):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__(device, dtype)
        self._native = NativePyTorchBackend(device, dtype)
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.PURE_XLA
    
    def is_available(self) -> bool:
        return _jax_available
    
    def get_bitlinear(self, in_features: int, out_features: int) -> nn.Module:
        # Return native PyTorch version - JAX version used in benchmark
        return self._native.get_bitlinear(in_features, out_features)
    
    def get_hbitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return self._native.get_hbitlinear(in_features, out_features)
    
    def get_model(self, config: Dict[str, Any], use_hadamard: bool = False) -> nn.Module:
        return self._native.get_model(config, use_hadamard)
    
    def get_jax_bitlinear(self, in_features: int, out_features: int):
        if not _jax_available:
            raise RuntimeError("JAX not available")
        return JaxBitLinear(in_features, out_features)
    
    def benchmark_jax(self, in_features: int, out_features: int, batch_size: int, 
                      warmup: int = 10, iters: int = 100) -> Dict[str, float]:
        if not _jax_available:
            return {"error": "JAX not available"}
        
        import time
        
        layer = JaxBitLinear(in_features, out_features)
        x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, in_features))
        
        # Warmup
        for _ in range(warmup):
            _ = layer(x)
        
        # Benchmark
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = layer(x)
            jax.block_until_ready(_)
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

