"""
Shared early exit loss utilities for BitNet models.
"""

from typing import Optional, Tuple, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def validate_tensor(tensor: torch.Tensor, name: str, expected_shape: Optional[Tuple[int, ...]] = None, expected_dtype: Optional[torch.dtype] = None) -> None:
    """
    Validate tensor properties and log them.

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for logging
        expected_shape: Expected shape of the tensor
        expected_dtype: Expected dtype of the tensor
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    logger.debug(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

    if expected_shape is not None and tensor.shape != expected_shape:
        raise ValueError(f"{name} has shape {tensor.shape}, expected {expected_shape}")

    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")


def compute_early_exit_loss_per_layer(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    lm_head: nn.Module,
    layer_idx: int,
    skip_mask: torch.Tensor,
    curriculum_mask: torch.Tensor  # Which layers have early exit enabled
) -> Optional[torch.Tensor]:
    """Compute early exit loss for samples that didn't skip this layer."""

    if not curriculum_mask[layer_idx].item():
        return None

    # Only compute loss for non-skipped samples
    active_mask = ~skip_mask
    if not active_mask.any().item():
        return None

    # Get logits for active samples
    active_states = hidden_states[active_mask]
    active_targets = target_ids[active_mask]
    logits = lm_head(active_states)

    if logits is not None and (torch.isnan(logits).any().item() or torch.isinf(logits).any().item()):
        print(f"ERROR: NaN/Inf detected in early exit logits for layer {layer_idx}!")
        return None

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        active_targets.view(-1),
        reduction='mean'
    )

    if loss is not None and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
        print(f"ERROR: NaN/Inf detected in early exit loss for layer {layer_idx}!")
        return None

    return loss


def _chunked_cross_entropy(hidden_states: torch.Tensor, lm_head: nn.Module,
                           targets: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Compute cross-entropy in chunks to avoid materializing (B*T, V) logits at once.

    With batch=128, seq=512, vocab=50257 the full projection is 6.14 GiB.
    Chunking at 4096 tokens keeps peak allocation under ~0.4 GiB per chunk.
    """
    hidden_flat = hidden_states.view(-1, hidden_states.size(-1))
    targets_flat = targets.view(-1)
    total_tokens = hidden_flat.size(0)

    loss_sum = torch.tensor(0.0, device=hidden_flat.device)
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        chunk_logits = lm_head(hidden_flat[start:end])
        loss_sum = loss_sum + F.cross_entropy(
            chunk_logits, targets_flat[start:end], reduction='sum'
        )
    return loss_sum / total_tokens


def compute_early_exit_loss(
    hidden_states_list: List[torch.Tensor],
    targets: torch.Tensor,
    lm_head: nn.Module,
    iteration: int,
    curriculum_fn,
    escale: float = 1.0
) -> torch.Tensor:
    """Safe early exit loss computation with NaN protection and chunked vocab projection."""

    # Add input validation
    if targets is None or targets.numel() == 0:
        return torch.tensor(0.0, device=targets.device if targets is not None else 'cpu', requires_grad=True)

    L = len(hidden_states_list)
    # Paper formula: w_i = (i+1)/L, then normalize
    weights = [(l + 1) / L for l in range(L)]
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    total_loss = 0.0
    valid_loss_count = 0

    for l, hidden_states in enumerate(hidden_states_list):
        if not curriculum_fn(l, iteration):
            continue

        # Check for NaN in hidden states
        if hidden_states is not None and (torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any()):
            continue

        try:
            layer_loss = _chunked_cross_entropy(hidden_states, lm_head, targets)

            if torch.isfinite(layer_loss):
                weighted_loss = normalized_weights[l] * layer_loss
                total_loss += weighted_loss
                valid_loss_count += 1
        except Exception:
            continue

    if valid_loss_count == 0:
        return torch.tensor(0.0, device=targets.device, requires_grad=True)

    final_loss = total_loss * escale

    # Final safety check
    if not torch.isfinite(final_loss):
        return torch.tensor(0.0, device=targets.device, requires_grad=True)

    return final_loss
