"""
Grouped Query Attention (GQA) implementation for BitNet models.
GQA reduces memory usage by having multiple query heads share the same key and value heads.
"""

import math
import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear
from .rope import RotaryEmbedding


class BitNetGQA(nn.Module):
    """
    Grouped Query Attention (GQA) using BitLinear layers with RoPE.
    
    GQA reduces memory usage by having multiple query heads share the same key and value heads.
    This is particularly useful for large models where KV cache memory becomes a bottleneck.
    
    Args:
        hidden_size: Hidden size of the model
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads)
        dropout: Dropout probability
        activation_bits: Number of bits for activation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
        activation_bits: int = 8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.activation_bits = activation_bits
        
        # Ensure hidden_size is divisible by num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        # Ensure num_heads is divisible by num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        
        # Log dimensions for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"BitNetGQA initialized with hidden_size={hidden_size}, num_heads={num_heads}, "
                   f"num_kv_heads={num_kv_heads}, head_dim={self.head_dim}, "
                   f"queries_per_kv={self.num_queries_per_kv}, activation_bits={self.activation_bits}")
        
        # Query projections - one for each query head
        self.q_proj = BitLinear(hidden_size, hidden_size, bias=False, activation_bits=self.activation_bits)
        
        # Key and Value projections - shared across query heads
        self.k_proj = BitLinear(hidden_size, num_kv_heads * self.head_dim, bias=False, activation_bits=self.activation_bits)
        self.v_proj = BitLinear(hidden_size, num_kv_heads * self.head_dim, bias=False, activation_bits=self.activation_bits)
        
        # Output projection
        self.o_proj = BitLinear(hidden_size, hidden_size, bias=False, activation_bits=self.activation_bits)
        
        # Rotary position embeddings - initialize with head_dim
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.head_dim)
    
    
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads to match the number of query heads.
        
        Args:
            x: Key or value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            n_rep: Number of repetitions (num_queries_per_kv)
            
        Returns:
            Repeated tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim).reshape(
            batch_size, num_kv_heads * n_rep, seq_len, head_dim
        )
    
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
            k: Key tensor of shape (batch_size, num_kv_heads, seq_len_k, head_dim)
            v: Value tensor of shape (batch_size, num_kv_heads, seq_len_v, head_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Attention output of shape (batch_size, num_heads, seq_len_q, head_dim)
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, num_kv_heads, seq_len_k, _ = k.shape
        _, _, seq_len_v, _ = v.shape
        
        assert seq_len_k == seq_len_v, f"Key and value sequence lengths must match: {seq_len_k} vs {seq_len_v}"
        
        # Repeat keys and values to match number of query heads
        k = self._repeat_kv(k, self.num_queries_per_kv)
        v = self._repeat_kv(v, self.num_queries_per_kv)
        
        # Now k and v have shape (batch_size, num_heads, seq_len, head_dim)
        assert k.shape == (batch_size, num_heads, seq_len_k, head_dim)
        assert v.shape == (batch_size, num_heads, seq_len_v, head_dim)
        
        # Calculate attention scores using standard PyTorch operations
        # q shape: (batch_size, num_heads, seq_len_q, head_dim)
        # k shape: (batch_size, num_heads, seq_len_k, head_dim)
        # attn_scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Get the key sequence length
            seq_len_k = k.size(-2)
            
            # Ensure attention mask has the right shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Ensure mask matches the key sequence length
            if attention_mask.size(-1) != seq_len_k:
                attention_mask = F.pad(attention_mask, (0, seq_len_k - attention_mask.size(-1)), value=1.0)
            
            # Apply mask (use large negative value for masked positions)
            attn_scores = attn_scores + (1.0 - attention_mask) * -10000.0
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
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
        Forward pass of the GQA layer.
        
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
        
        # Normalize before projections
        normed = self.norm(hidden_states) if hasattr(self, 'norm') else nn.LayerNorm(self.hidden_size).to(hidden_states.device)(hidden_states)
        
        # Project queries, keys, values
        q = self.q_proj(normed)  # (batch_size, seq_len, hidden_size)
        k = self.k_proj(normed)  # (batch_size, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(normed)  # (batch_size, seq_len, num_kv_heads * head_dim)
        
        # Reshape for multi-head attention
        # Query: (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Key: (batch_size, seq_len, num_kv_heads * head_dim) -> (batch_size, num_kv_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Value: (batch_size, seq_len, num_kv_heads * head_dim) -> (batch_size, num_kv_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        rotary_pos = cache_position if cache_position is not None else position_ids
        q = self.rotary_emb(q, seq_len=seq_length, position_ids=rotary_pos)
        k = self.rotary_emb(k, seq_len=seq_length, position_ids=rotary_pos)
        
        # Handle past key-values for generation
        if layer_past is not None and use_cache:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Compute attention
        with torch.autograd.profiler.record_function("GQA.compute"):
            attn_output = self._compute_attention(q, k, v, attention_mask)
        
        # Reshape back to (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # Final projection
        attn_output = self.o_proj(attn_output)
        
        # Return past key-values for generation
        if use_cache:
            present = (k, v)
            return attn_output, present
        else:
            return attn_output
