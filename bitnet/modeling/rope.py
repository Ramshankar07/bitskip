"""
Rotary Position Embeddings (RoPE) implementation.
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Rotated tensor of same shape
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, position_ids):
    with torch.autograd.profiler.record_function("RoPE.apply"):
        batch_size = x.shape[0]
        if x.dim() == 4:
            seq_len = x.shape[2]
            num_heads = x.shape[1]
            head_dim = x.shape[3]
        elif x.dim() == 3:
            seq_len = x.shape[1]
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        max_pos = cos.shape[0]
        if position_ids is not None:
            min_id = position_ids.min().item() if position_ids.numel() > 0 else 0
            max_id = position_ids.max().item() if position_ids.numel() > 0 else 0
            if min_id < 0 or max_id >= max_pos:
                raise ValueError(
                    f"Position id(s) out of bounds: min={min_id}, max={max_id}, allowed=[0, {max_pos-1}]."
                )
            if position_ids.dim() == 2:
                if position_ids.shape[0] != batch_size or position_ids.shape[1] != seq_len:
                    raise ValueError(f"position_ids shape {position_ids.shape} does not match (batch_size, seq_len)=({batch_size}, {seq_len})")
                if x.dim() == 4:
                    cos_indexed = cos[position_ids]
                    sin_indexed = sin[position_ids]
                    cos_indexed = cos_indexed.unsqueeze(1)
                    sin_indexed = sin_indexed.unsqueeze(1)
                else:
                    cos_indexed = cos[position_ids]
                    sin_indexed = sin[position_ids]
            elif position_ids.dim() == 1:
                if position_ids.shape[0] != seq_len:
                    raise ValueError(f"1D position_ids length {position_ids.shape[0]} does not match seq_len {seq_len}")
                cos_indexed = cos[position_ids]
                sin_indexed = sin[position_ids]
                if x.dim() == 4:
                    cos_indexed = cos_indexed.unsqueeze(0).unsqueeze(0)
                    sin_indexed = sin_indexed.unsqueeze(0).unsqueeze(0)
                else:
                    cos_indexed = cos_indexed.unsqueeze(0)
                    sin_indexed = sin_indexed.unsqueeze(0)
            else:
                raise ValueError(f"Unsupported position_ids shape: {position_ids.shape}")
        else:
            cos_indexed = cos[:seq_len]
            sin_indexed = sin[:seq_len]
            if x.dim() == 4:
                cos_indexed = cos_indexed.unsqueeze(0).unsqueeze(0)
                sin_indexed = sin_indexed.unsqueeze(0).unsqueeze(0)
            else:
                cos_indexed = cos_indexed.unsqueeze(0)
                sin_indexed = sin_indexed.unsqueeze(0)
        cos_indexed = torch.cat([cos_indexed, cos_indexed], dim=-1)
        sin_indexed = torch.cat([sin_indexed, sin_indexed], dim=-1)
        x_rotated = rotate_half(x)
        output = x * cos_indexed + x_rotated * sin_indexed
        return output


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Args:
        dim: Dimension of the embeddings (head_dim)
        max_position_embeddings: Maximum sequence length
        base: Base for the frequency computation
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Log initialization
        logger = logging.getLogger(__name__)
        logger.info(f"RotaryEmbedding initialized with dim={dim}, max_position_embeddings={max_position_embeddings}")
        
        # Generate and cache rotary embeddings
        self._update_cos_sin_cache(max_position_embeddings)
    
    def _update_cos_sin_cache(self, seq_len: int) -> None:
        """
        Update the cached cos and sin embeddings.
        
        Args:
            seq_len: Sequence length to generate embeddings for
        """
        # Generate position indices
        position = torch.arange(seq_len, dtype=torch.float)
        
        # Generate dimension indices (only need half since we rotate pairs)
        dim_indices = torch.arange(0, self.dim, 2, dtype=torch.float)
        
        # Compute frequencies
        freqs = 1.0 / (self.base ** (dim_indices / self.dim))
        
        # Compute angles: position * frequency
        # position shape: (seq_len, 1)
        # freqs shape: (dim/2,)
        angles = position.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, dim/2)
        
        # Compute cos and sin
        cos = torch.cos(angles)  # (seq_len, dim/2)
        sin = torch.sin(angles)  # (seq_len, dim/2)
        
        # Cache the embeddings
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        import logging
        
        # Store original shape and dtype
        orig_shape = x.shape
        orig_dtype = x.dtype
        
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch_size, num_heads, seq_len, head_dim), got {x.dim()}D tensor with shape {x.shape}")
        
        # Validate that the last dimension matches our embedding dimension
        if x.shape[-1] != self.dim:
            raise ValueError(f"Last dimension {x.shape[-1]} does not match rotary embedding dimension {self.dim}")
        
        # Get actual sequence length from input
        actual_seq_len = x.shape[2]
        
        # CRITICAL FIX: Validate position_ids before using them
        if position_ids is not None:
            max_valid_pos = self.max_position_embeddings - 1
            if position_ids.max() > max_valid_pos:
                logging.warning(
                    f"Clamping position_ids from max {position_ids.max().item()} "
                    f"to {max_valid_pos}"
                )
                position_ids = position_ids.clamp(0, max_valid_pos)
        
        # Update cache if needed (but this shouldn't happen with max_pos=128)
        if actual_seq_len > self.max_position_embeddings:
            logging.warning(
                f"Sequence length {actual_seq_len} exceeds max_position_embeddings "
                f"{self.max_position_embeddings}. This should not happen!"
            )
            self._update_cos_sin_cache(actual_seq_len)
            self.max_position_embeddings = actual_seq_len
        
        # Get cached embeddings
        cos = self.cos_cached
        sin = self.sin_cached
        
        # Move to same device and dtype as input
        if cos.device != x.device:
            cos = cos.to(x.device)
            sin = sin.to(x.device)
        if cos.dtype != x.dtype:
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)
        
        # Apply rotary embeddings with validated position_ids
        output = apply_rotary_pos_emb(x, cos, sin, position_ids)
        
        return output