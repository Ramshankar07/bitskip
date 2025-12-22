"""PyTorch/XLA custom ops."""

torch_xla_available = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    torch_xla_available = True
except ImportError:
    pass


def get_xla_device():
    if torch_xla_available:
        return xm.xla_device()
    return None


def mark_step():
    if torch_xla_available:
        xm.mark_step()

