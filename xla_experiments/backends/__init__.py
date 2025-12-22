"""Backend implementations wrapping existing bitnet modules."""

from .base import Backend, BackendType
from .native_pytorch import NativePyTorchBackend
from .torch_compile import TorchCompileBackend
from .cuda_cutlass import CUDACutlassBackend
from .cuda_xla import CUDAXLABackend
from .pure_xla import PureXLABackend

ALL_BACKENDS = {
    BackendType.NATIVE_PYTORCH: NativePyTorchBackend,
    BackendType.TORCH_COMPILE: TorchCompileBackend,
    BackendType.CUDA_CUTLASS: CUDACutlassBackend,
    BackendType.CUDA_XLA: CUDAXLABackend,
    BackendType.PURE_XLA: PureXLABackend,
}

__all__ = ["Backend", "BackendType", "ALL_BACKENDS"]
