"""
Grouped Query Attention (GQA) implementation for BitNet models with H-BitLinear layers.
GQA reduces memory usage by having multiple query heads share the same key and value heads.
"""

import math
import logging
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .h_bitlinear import HBitLinear
from .rope import RotaryEmbedding


class BitNetGQA2(nn.Module):
    """
    Grouped Query Attention (GQA) using H-BitLinear layers with RoPE.
    
    GQA reduces memory usage by having multiple query heads share the same key and value heads.
    This is particularly useful for large models where KV cache memory becomes a bottleneck.
    
    Args:
        hidden_size: Hidden size of the model
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads)
        dropout: Dropout probability
        activation_bits: Number of bits for activation
        weight_bits: Number of bits for weights
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
        activation_bits: int = 8,
        weight_bits: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        
        # Ensure hidden_size is divisible by num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        # Ensure num_heads is divisible by num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        
        self.q_proj = HBitLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            activation_bits=activation_bits,
        )
        
        self.k_proj = HBitLinear(
            in_features=hidden_size,
            out_features=num_kv_heads * self.head_dim,
            bias=False,
            activation_bits=activation_bits,
        )
        
        self.v_proj = HBitLinear(
            in_features=hidden_size,
            out_features=num_kv_heads * self.head_dim,
            bias=False,
            activation_bits=activation_bits,
        )
        
        self.o_proj = HBitLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            activation_bits=activation_bits,
        )
        
        # RoPE for positional encoding
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads for grouped query attention.

        Args:
            x: Key or value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            n_rep: Number of repetitions (num_queries_per_kv)

        Returns:
            Repeated tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.shape

        if n_rep == 1:
            return x

        # Repeat along the head dimension
        return x[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim).reshape(
            batch_size, num_kv_heads * n_rep, seq_len, head_dim
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the GQA attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to cache key-value pairs
            
        Returns:
            Attention output (and optionally cached key-values if use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape and transpose to (batch, heads, seq, dim) for RoPE and attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (returns single rotated tensor, expects (batch, heads, seq, dim))
        query_states = self.rotary_emb(query_states, seq_len=seq_len, position_ids=position_ids)
        key_states = self.rotary_emb(key_states, seq_len=seq_len, position_ids=position_ids)

        # Handle past key-value states for generation (dim=2 is seq dim in transposed layout)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Repeat key and value for grouped query attention (already in (batch, heads, seq, dim) format)
        key_states = self._repeat_kv(key_states, self.num_queries_per_kv)
        value_states = self._repeat_kv(value_states, self.num_queries_per_kv)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Causal mask: prevent attending to future positions
        if seq_len > 1:
            seq_len_k = key_states.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len_k, dtype=torch.bool, device=query_states.device),
                diagonal=seq_len_k - seq_len + 1,
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        # Apply attention mask
        if attention_mask is not None:
            # Reshape 2D/3D mask to 4D for broadcasting with (B, heads, seq_q, seq_k)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            # Ensure mask covers key sequence length (may differ with past_key_value)
            seq_len_k = key_states.size(2)
            if attention_mask.size(-1) != seq_len_k:
                attention_mask = F.pad(attention_mask, (0, seq_len_k - attention_mask.size(-1)), value=1.0)

            # Convert binary (1=attend, 0=mask) to additive format
            attn_weights = attn_weights + (1.0 - attention_mask) * -10000.0
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Project output
        attn_output = self.o_proj(attn_output)
        
        
        
        
        # Return with optional cached key-values
        if use_cache:
            present_key_value = (key_states, value_states)
            return attn_output, present_key_value
        else:
            return attn_output
    
    


# Alias for backward compatibility
GQAAttention2 = BitNetGQA2
