"""Custom CUDA kernels for quantized ops."""

import os
from typing import Optional
import torch

_ext: Optional[object] = None


def _load_extension() -> Optional[object]:
    global _ext
    if _ext is not None:
        return _ext
    try:
        from torch.utils.cpp_extension import load
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="cutlass_int8_ext",
            sources=[
                os.path.join(src_dir, "cutlass_wrapper.cpp"),
                os.path.join(src_dir, "int8_gemm.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _ext
    except Exception:
        return None


def is_cutlass_available() -> bool:
    return _load_extension() is not None


def int8_gemm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    ext = _load_extension()
    if ext is not None and a.is_cuda:
        return ext.int8_gemm(a, b, scale_a, scale_b)
    return (a.float() @ b.float()) * scale_a * scale_b
