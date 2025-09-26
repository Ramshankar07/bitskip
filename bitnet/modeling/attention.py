"""
BitNet attention implementation with H-BitLinear layers and RoPE.
"""

import math
import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear
from .rope import RotaryEmbedding
from .kernels import (
    bitnet_kernels,
    bitlinear_forward_cuda,
    attention_scores_cuda,
    attention_output_cuda
)


class BitNetAttention(nn.Module):
    """
    Multi-head attention using BitLinear and H-BitLinear layers with RoPE.
    
    Args:
        hidden_size: Hidden size of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
        activation_bits: Number of bits for activation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        activation_bits: int = 8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.activation_bits = activation_bits
        
        # Ensure hidden_size is divisible by num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        # Log dimensions for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"BitNetAttention initialized with hidden_size={hidden_size}, num_heads={num_heads}, head_dim={self.head_dim}, activation_bits={self.activation_bits}")
        
        # QKV projections using BitLinear
        self.q_proj = BitLinear(hidden_size, hidden_size, bias=False, activation_bits=self.activation_bits)
        self.k_proj = BitLinear(hidden_size, hidden_size, bias=False, activation_bits=self.activation_bits)
        self.v_proj = BitLinear(hidden_size, hidden_size, bias=False, activation_bits=self.activation_bits)
        
        # Output projection using BitLinear instead of H-BitLinear
        # H-BitLinear can cause issues with dimension handling, so we use regular BitLinear
        self.o_proj = BitLinear(hidden_size, hidden_size, bias=False, activation_bits=self.activation_bits)
        
        # Rotary position embeddings - initialize with head_dim
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.head_dim)
    
    def collect_quantization_losses(self) -> Dict[str, torch.Tensor]:
        """
        Collect quantization losses from BitLinear layers.
        
        Returns:
            Dictionary containing quantization loss information
        """
        quantization_info = {}
        
        # Collect from QKV projections
        for name, proj in [('q_proj', self.q_proj), ('k_proj', self.k_proj), ('v_proj', self.v_proj)]:
            if hasattr(proj, 'compute_quantization_loss'):
                # Get the current weights and compute quantization loss
                original_weights = proj.weight
                w_q, w_scale = proj._weight_quantize(original_weights)
                quant_loss = proj.compute_quantization_loss(original_weights, w_q * w_scale)
                quantization_info[f"{name}_quantization_loss"] = quant_loss
        
        # Collect from output projection
        if hasattr(self.o_proj, 'compute_quantization_loss'):
            original_weights = self.o_proj.weight
            w_q, w_scale = self.o_proj._weight_quantize(original_weights)
            quant_loss = self.o_proj.compute_quantization_loss(original_weights, w_q * w_scale)
            quantization_info['o_proj_quantization_loss'] = quant_loss
        
        return quantization_info
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention scores and apply to values.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
            v: Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Attention output of shape (batch_size, num_heads, seq_len_q, head_dim)
        """
        # Debug shapes
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_k, _ = k.shape
        _, _, seq_len_v, _ = v.shape
        
        assert seq_len_k == seq_len_v, f"Key and value sequence lengths must match: {seq_len_k} vs {seq_len_v}"
        
        print(f"DEBUG: bitnet_kernels.is_available = {bitnet_kernels.is_available}")
        
        if bitnet_kernels.is_available:
            print(f"DEBUG: Using CUDA kernel for attention computation")
            print(f"DEBUG: q stats before CUDA: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
            print(f"DEBUG: k stats before CUDA: min={k.min().item():.4f}, max={k.max().item():.4f}, mean={k.mean().item():.4f}")
            print(f"DEBUG: scale = {self.scale}")
            
            # Use CUDA kernel for attention computation
            attn_scores = attention_scores_cuda(q, k, self.scale)
            print(f"DEBUG: After attention_scores_cuda - attn_scores stats: min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}, mean={attn_scores.mean().item():.4f}")
            
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                print(f"ERROR: NaN/Inf detected in attn_scores after CUDA kernel!")
                print(f"ERROR: NaN count: {torch.isnan(attn_scores).sum()}")
                print(f"ERROR: Inf count: {torch.isinf(attn_scores).sum()}")
            
            if attention_mask is not None:
                # Prepare attention mask for CUDA kernel
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                attn_scores = attn_scores + attention_mask
                print(f"DEBUG: After attention mask - attn_scores stats: min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}, mean={attn_scores.mean().item():.4f}")
            
            # Apply softmax and dropout
            print(f"DEBUG: Before softmax - attn_scores stats: min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}, mean={attn_scores.mean().item():.4f}")
            attn_weights = F.softmax(attn_scores, dim=-1)
            print(f"DEBUG: After softmax - attn_weights stats: min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}, mean={attn_weights.mean().item():.4f}")
            
            if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                print(f"ERROR: NaN/Inf detected in attn_weights after softmax!")
                print(f"ERROR: NaN count: {torch.isnan(attn_weights).sum()}")
                print(f"ERROR: Inf count: {torch.isinf(attn_weights).sum()}")
            
            attn_weights = self.dropout(attn_weights)
            print(f"DEBUG: After dropout - attn_weights stats: min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}, mean={attn_weights.mean().item():.4f}")
            
            # Compute attention output using CUDA kernel
            print(f"DEBUG: Before attention_output_cuda - v stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
            result = attention_output_cuda(attn_weights, v)
            print(f"DEBUG: After attention_output_cuda - result stats: min={result.min().item():.4f}, max={result.max().item():.4f}, mean={result.mean().item():.4f}")
            
            if torch.isnan(result).any() or torch.isinf(result).any():
                print(f"ERROR: NaN/Inf detected in result after attention_output_cuda!")
                print(f"ERROR: NaN count: {torch.isnan(result).sum()}")
                print(f"ERROR: Inf count: {torch.isinf(result).sum()}")
            
            return result
        else:
            print(f"DEBUG: Using CPU implementation for attention computation")
            print(f"DEBUG: q stats before CPU: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
            print(f"DEBUG: k stats before CPU: min={k.min().item():.4f}, max={k.max().item():.4f}, mean={k.mean().item():.4f}")
            print(f"DEBUG: scale = {self.scale}")
            
            # Calculate attention scores
            # q shape: (batch_size, num_heads, seq_len_q, head_dim)
            # k shape: (batch_size, num_heads, seq_len_k, head_dim)
            # attn_scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
            
            # Check for extreme values before matmul
            print(f"DEBUG: Q max abs: {q.abs().max():.4f}, K max abs: {k.abs().max():.4f}")
            print(f"DEBUG: Q std: {q.std():.4f}, K std: {k.std():.4f}")
            print(f"DEBUG: head_dim: {self.head_dim}, scale: {self.scale:.6f}")
            
            # Check if values are too large
            if q.abs().max() > 1000 or k.abs().max() > 1000:
                print(f"WARNING: Very large Q/K values detected! Q max: {q.abs().max():.4f}, K max: {k.abs().max():.4f}")
            
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            print(f"DEBUG: After matmul - attn_scores stats: min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}, mean={attn_scores.mean().item():.4f}")
            
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                print(f"ERROR: NaN/Inf detected in attn_scores after matmul!")
                print(f"ERROR: NaN count: {torch.isnan(attn_scores).sum()}")
                print(f"ERROR: Inf count: {torch.isinf(attn_scores).sum()}")
            
            # Verify attention scores shape
            expected_scores_shape = (batch_size, num_heads, seq_len_q, seq_len_k)
            assert attn_scores.shape == expected_scores_shape, f"Attention scores shape mismatch: {attn_scores.shape} vs {expected_scores_shape}"
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Get the key sequence length (which should match mask's last dimension)
                seq_len_k = k.size(-2)
                
                # Ensure attention mask has the right shape
                if attention_mask.dim() == 2:
                    # attention_mask shape: (batch_size, seq_len)
                    # We need shape: (batch_size, 1, 1, seq_len)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    # attention_mask shape: (batch_size, 1, seq_len)
                    # We need shape: (batch_size, 1, 1, seq_len)
                    attention_mask = attention_mask.unsqueeze(1)
                
                # Ensure mask matches the key sequence length
                if attention_mask.size(-1) != seq_len_k:
                    # This can happen with cached keys - adjust mask
                    attention_mask = F.pad(attention_mask, (0, seq_len_k - attention_mask.size(-1)), value=1.0)
                
                # Apply mask (use large negative value for masked positions)
                # Note: attention_mask is 1 for positions to attend and 0 for positions to mask
                attn_scores = attn_scores + (1.0 - attention_mask) * -10000.0
            
            # Apply softmax and dropout
            print(f"DEBUG: Before softmax (CPU) - attn_scores stats: min={attn_scores.min().item():.4f}, max={attn_scores.max().item():.4f}, mean={attn_scores.mean().item():.4f}")
            attn_weights = F.softmax(attn_scores, dim=-1)
            print(f"DEBUG: After softmax (CPU) - attn_weights stats: min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}, mean={attn_weights.mean().item():.4f}")
            
            if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                print(f"ERROR: NaN/Inf detected in attn_weights after softmax (CPU)!")
                print(f"ERROR: NaN count: {torch.isnan(attn_weights).sum()}")
                print(f"ERROR: Inf count: {torch.isinf(attn_weights).sum()}")
            
            attn_weights = self.dropout(attn_weights)
            print(f"DEBUG: After dropout (CPU) - attn_weights stats: min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}, mean={attn_weights.mean().item():.4f}")
            
            # Verify attention weights shape before matmul
            expected_weights_shape = (batch_size, num_heads, seq_len_q, seq_len_k)
            assert attn_weights.shape == expected_weights_shape, f"Attention weights shape mismatch: {attn_weights.shape} vs {expected_weights_shape}"
            
            # Apply attention to values
            # attn_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
            # v shape: (batch_size, num_heads, seq_len_v, head_dim)
            # output shape: (batch_size, num_heads, seq_len_q, head_dim)
            print(f"DEBUG: Before final matmul (CPU) - attn_weights stats: min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}, mean={attn_weights.mean().item():.4f}")
            print(f"DEBUG: Before final matmul (CPU) - v stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
            attn_output = torch.matmul(attn_weights, v)
            print(f"DEBUG: After final matmul (CPU) - attn_output stats: min={attn_output.min().item():.4f}, max={attn_output.max().item():.4f}, mean={attn_output.mean().item():.4f}")
            
            if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
                print(f"ERROR: NaN/Inf detected in attn_output after final matmul (CPU)!")
                print(f"ERROR: NaN count: {torch.isnan(attn_output).sum()}")
                print(f"ERROR: Inf count: {torch.isinf(attn_output).sum()}")
            
            # Verify output shape
            expected_output_shape = (batch_size, num_heads, seq_len_q, head_dim)
            assert attn_output.shape == expected_output_shape, f"Attention output shape mismatch: {attn_output.shape} vs {expected_output_shape}"
            
            return attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the attention layer.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            layer_past: Optional past key-value pairs for generation
            use_cache: Whether to cache key-value pairs
            position_ids: Optional position IDs for RoPE of shape (batch_size, seq_len)
            
        Returns:
            Attention output of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        print(f"DEBUG: Attention input - hidden_states stats: min={hidden_states.min().item():.4f}, max={hidden_states.max().item():.4f}, mean={hidden_states.mean().item():.4f}")
        
        # Normalize before projections
        normed = self.norm(hidden_states) if hasattr(self, 'norm') else nn.LayerNorm(self.hidden_size).to(hidden_states.device)(hidden_states)
        print(f"DEBUG: After norm - normed stats: min={normed.min().item():.4f}, max={normed.max().item():.4f}, mean={normed.mean().item():.4f}")
        
        if torch.isnan(normed).any() or torch.isinf(normed).any():
            print(f"ERROR: NaN/Inf detected in normed!")
            print(f"ERROR: NaN count: {torch.isnan(normed).sum()}")
            print(f"ERROR: Inf count: {torch.isinf(normed).sum()}")
        
        # Assert normalized shape
        assert normed.shape == (batch_size, seq_length, self.hidden_size), f"Normed shape mismatch: {normed.shape} vs ({batch_size}, {seq_length}, {self.hidden_size})"
        
        # Project queries, keys, values with normalized input
        print(f"DEBUG: Before projections")
        q = self.q_proj(normed)
        print(f"DEBUG: After q_proj - q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
        
        if torch.isnan(q).any() or torch.isinf(q).any():
            print(f"ERROR: NaN/Inf detected in q!")
            print(f"ERROR: NaN count: {torch.isnan(q).sum()}")
            print(f"ERROR: Inf count: {torch.isinf(q).sum()}")
        
        k = self.k_proj(normed)
        print(f"DEBUG: After k_proj - k stats: min={k.min().item():.4f}, max={k.max().item():.4f}, mean={k.mean().item():.4f}")
        
        if torch.isnan(k).any() or torch.isinf(k).any():
            print(f"ERROR: NaN/Inf detected in k!")
            print(f"ERROR: NaN count: {torch.isnan(k).sum()}")
            print(f"ERROR: Inf count: {torch.isinf(k).sum()}")
        
        v = self.v_proj(normed)
        print(f"DEBUG: After v_proj - v stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
        
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"ERROR: NaN/Inf detected in v!")
            print(f"ERROR: NaN count: {torch.isnan(v).sum()}")
            print(f"ERROR: Inf count: {torch.isinf(v).sum()}")
        # Assert projection shapes
        assert q.shape == (batch_size, seq_length, self.hidden_size), f"Q projection shape mismatch: {q.shape}"
        assert k.shape == (batch_size, seq_length, self.hidden_size), f"K projection shape mismatch: {k.shape}"
        assert v.shape == (batch_size, seq_length, self.hidden_size), f"V projection shape mismatch: {v.shape}"
        
        # Reshape for multi-head attention
        # From (batch_size, seq_len, hidden_size) to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Verify shapes after reshape
        assert q.shape == (batch_size, self.num_heads, seq_length, self.head_dim), f"Q shape mismatch: {q.shape}"
        assert k.shape == (batch_size, self.num_heads, seq_length, self.head_dim), f"K shape mismatch: {k.shape}"
        assert v.shape == (batch_size, self.num_heads, seq_length, self.head_dim), f"V shape mismatch: {v.shape}"
        
        # Apply rotary embeddings
        # The rotary embeddings expect input of shape (batch_size, num_heads, seq_len, head_dim)
        # Use cache_position for rotary embedding if provided, else use position_ids
        rotary_pos = cache_position if cache_position is not None else position_ids
        print(f"DEBUG: Before RoPE - q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
        q = self.rotary_emb(q, seq_len=seq_length, position_ids=rotary_pos)
        print(f"DEBUG: After RoPE q - q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
        
        if torch.isnan(q).any() or torch.isinf(q).any():
            print(f"ERROR: NaN/Inf detected in q after RoPE!")
            print(f"ERROR: NaN count: {torch.isnan(q).sum()}")
            print(f"ERROR: Inf count: {torch.isinf(q).sum()}")
        
        k = self.rotary_emb(k, seq_len=seq_length, position_ids=rotary_pos)
        print(f"DEBUG: After RoPE k - k stats: min={k.min().item():.4f}, max={k.max().item():.4f}, mean={k.mean().item():.4f}")
        
        if torch.isnan(k).any() or torch.isinf(k).any():
            print(f"ERROR: NaN/Inf detected in k after RoPE!")
            print(f"ERROR: NaN count: {torch.isnan(k).sum()}")
            print(f"ERROR: Inf count: {torch.isinf(k).sum()}")
        
        # Verify shapes after RoPE
        assert q.shape == (batch_size, self.num_heads, seq_length, self.head_dim), f"Q shape after RoPE mismatch: {q.shape}"
        assert k.shape == (batch_size, self.num_heads, seq_length, self.head_dim), f"K shape after RoPE mismatch: {k.shape}"
        
        # Handle past key-values for generation
        if layer_past is not None and use_cache:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Compute attention
        print(f"DEBUG: Before attention computation - q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
        print(f"DEBUG: Before attention computation - k stats: min={k.min().item():.4f}, max={k.max().item():.4f}, mean={k.mean().item():.4f}")
        print(f"DEBUG: Before attention computation - v stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
        
        with torch.autograd.profiler.record_function("Attention.compute"):
            attn_output = self._compute_attention(q, k, v, attention_mask)
        
        print(f"DEBUG: After attention computation - attn_output stats: min={attn_output.min().item():.4f}, max={attn_output.max().item():.4f}, mean={attn_output.mean().item():.4f}")
        
        if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
            print(f"ERROR: NaN/Inf detected in attn_output!")
            print(f"ERROR: NaN count: {torch.isnan(attn_output).sum()}")
            print(f"ERROR: Inf count: {torch.isinf(attn_output).sum()}")
        
        # Verify attention output shape
        assert attn_output.shape == (batch_size, self.num_heads, seq_length, self.head_dim), f"Attention output shape mismatch: {attn_output.shape}"
        
        # Reshape back to (batch_size, seq_len, hidden_size)
        # First transpose from (batch_size, num_heads, seq_len, head_dim) to (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Then reshape to (batch_size, seq_len, hidden_size)
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # Verify final shape
        assert attn_output.shape == (batch_size, seq_length, self.hidden_size), \
            f"Final attention output shape mismatch: {attn_output.shape} vs expected ({batch_size}, {seq_length}, {self.hidden_size})"
        
        # Final projection with BitLinear
        # Always use the standard forward method to avoid any shape issues
        attn_output = self.o_proj(attn_output)
        
        # Final shape verification
        assert attn_output.shape == (batch_size, seq_length, self.hidden_size), \
            f"Output projection changed shape: expected ({batch_size}, {seq_length}, {self.hidden_size}), got {attn_output.shape}"
        
        # Return past key-values for generation
        if use_cache:
            present = (k, v)
            return attn_output, present
        else:
            return attn_output