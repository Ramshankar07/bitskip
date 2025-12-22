"""Native PyTorch backend - uses existing bitnet modules directly."""

from typing import Dict, Any
import torch.nn as nn

from .base import Backend, BackendType
from bitnet.modeling import BitLinear, HBitLinear, BitNetModel
from bitnet.modeling.model2 import BitNetModel2
from bitnet.utils.default_config import DefaultConfig


class NativePyTorchBackend(Backend):
    @property
    def backend_type(self) -> BackendType:
        return BackendType.NATIVE_PYTORCH
    
    def is_available(self) -> bool:
        return True
    
    def get_bitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return BitLinear(in_features, out_features)
    
    def get_hbitlinear(self, in_features: int, out_features: int) -> nn.Module:
        return HBitLinear(in_features, out_features)
    
    def get_model(self, config: Dict[str, Any], use_hadamard: bool = False) -> nn.Module:
        cfg = DefaultConfig()
        for k, v in config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return BitNetModel2(cfg) if use_hadamard else BitNetModel(cfg)

