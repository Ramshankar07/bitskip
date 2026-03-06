"""
Muon optimizer: MomentUm Orthogonalized by Newton-schulz.

Standalone implementation for PyTorch < 2.10 (which lacks torch.optim.Muon).
Applies Newton-Schulz orthogonalization to momentum updates for 2D weight matrices.
Non-2D params (embeddings, LM head, LayerNorm, biases, routing MLPs) use AdamW.

Reference: https://arxiv.org/abs/2502.16982
"""

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute the zeroth power / orthogonalization of G via 5 Newton-Schulz iterations.

    Uses the quintic polynomial from the Muon reference implementation.
    Operates in bfloat16 on CUDA for speed, float32 elsewhere (MPS lacks bf16 matmul).
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"

    # Use bfloat16 on CUDA for speed; float32 on MPS/CPU
    if G.device.type == "cuda":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    original_dtype = G.dtype
    X = G.to(compute_dtype)
    X = X / (X.norm() + 1e-7)

    # Quintic Newton-Schulz coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # a]I + bA + cA^2 (I term applied below)
        X = a * X + B @ X

    return X.to(original_dtype)


class SingleDeviceMuon(Optimizer):
    """Non-distributed Muon optimizer.

    Applies Nesterov momentum followed by Newton-Schulz orthogonalization.
    All params must be 2D (enforced at construction).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0, ns_steps: int = 5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

        # Validate all params are 2D
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"SingleDeviceMuon requires all params to be 2D, got shape {p.shape}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # Nesterov momentum
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                # Newton-Schulz orthogonalization
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)

                # Scale by aspect ratio (makes update scale-invariant)
                h, w = p.shape
                g = g * max(1, h / w) ** 0.5

                # Apply update
                p.add_(g, alpha=-lr)

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

        return loss


class MuonWithAdamW:
    """Wrapper holding SingleDeviceMuon + AdamW for unified optimizer interface.

    Muon handles 2D hidden weight matrices, AdamW handles everything else.
    """

    def __init__(self, muon: SingleDeviceMuon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        """Combined param_groups for gradient clipping compatibility."""
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_muon_param_groups(model, verbose: bool = False):
    """Split model params into Muon-eligible (2D BitLinear/HBitLinear weights) and AdamW (rest).

    Returns:
        (muon_params, adamw_params) — two lists ready for optimizer constructors.
    """
    from bitnet.modeling.bitlinear import BitLinear
    from bitnet.modeling.h_bitlinear import HBitLinear

    muon_params = []
    adamw_params = []
    seen_ids = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            pid = id(param)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            if (
                isinstance(module, (BitLinear, HBitLinear))
                and param_name == "weight"
                and param.ndim == 2
            ):
                muon_params.append(param)
                if verbose:
                    print(f"  Muon: {module_name}.{param_name} {tuple(param.shape)}")
            else:
                adamw_params.append(param)
                if verbose:
                    print(f"  AdamW: {module_name}.{param_name} {tuple(param.shape)}")

    if verbose:
        print(f"\nTotal: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params")

    return muon_params, adamw_params
