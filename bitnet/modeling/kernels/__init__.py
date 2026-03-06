import os
import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Backend 1: Triton (fastest — fully parallel butterfly operations)
# ---------------------------------------------------------------------------
try:
    from .triton_fwht import fwht_triton as _fwht_triton
    from .triton_fwht import is_available as _triton_is_available
except ImportError:
    _fwht_triton = None
    _triton_is_available = lambda: False

# ---------------------------------------------------------------------------
# Backend 2: Custom CUDA extension (fallback)
# ---------------------------------------------------------------------------
_ext: Optional[object] = None


def _load_extension() -> Optional[object]:
    global _ext
    if _ext is not None:
        return _ext
    try:
        from torch.utils.cpp_extension import load
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="fwht_ext",
            sources=[
                os.path.join(src_dir, "fwht.cpp"),
                os.path.join(src_dir, "fwht.cu"),
            ],
            verbose=False,
        )
        return _ext
    except Exception:
        _ext = None
        return None


def is_available() -> bool:
    return _triton_is_available() or _load_extension() is not None


def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform along the last dimension (power-of-two length).

    Backend priority: Triton > CUDA extension > CPU fallback.
    Includes 1/sqrt(n) scaling for orthogonality.
    """
    if x.dim() == 0:
        return x
    n = x.shape[-1]
    if (n & (n - 1)) != 0:
        raise ValueError(f"FWHT requires power-of-two length, got {n}")

    # --- Triton (with proper autograd) ---
    if _fwht_triton is not None and x.is_cuda and _triton_is_available():
        return _fwht_triton(x)

    # --- CUDA extension ---
    scaling = 1.0 / math.sqrt(n)
    ext = _load_extension()
    if ext is not None and x.is_cuda:
        return ext.fwht(x) * scaling

    # --- CPU / PyTorch fallback ---
    y = x.detach().clone().contiguous().view(-1, n)
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            a = y[:, start:start + h].clone()
            b = y[:, start + h:start + 2 * h].clone()
            y[:, start:start + h] = a + b
            y[:, start + h:start + 2 * h] = a - b
        h <<= 1

    return (y.view(x.shape) * scaling).to(x.dtype)
