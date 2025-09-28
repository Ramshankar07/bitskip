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
        
        # Ensure head_dim is power of 2 for H-BitLinear
        if not self._is_power_of_2(self.head_dim):
            raise ValueError(f"head_dim ({self.head_dim}) must be a power of 2 for H-BitLinear")
        
        # Query projection using H-BitLinear
        self.q_proj = HBitLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )
        
        # Key projection using H-BitLinear (smaller output for GQA)
        self.k_proj = HBitLinear(
            in_features=hidden_size,
            out_features=num_kv_heads * self.head_dim,
            bias=False
        )
        
        # Value projection using H-BitLinear (smaller output for GQA)
        self.v_proj = HBitLinear(
            in_features=hidden_size,
            out_features=num_kv_heads * self.head_dim,
            bias=False
        )
        
        # Output projection using H-BitLinear
        self.o_proj = HBitLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )
        
        # RoPE for positional encoding
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _is_power_of_2(self, n: int) -> bool:
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads for grouped query attention.
        
        Args:
            x: Key or value tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            n_rep: Number of repetitions (num_queries_per_kv)
            
        Returns:
            Repeated tensor of shape (batch_size, seq_len, num_heads, head_dim)
        """
        batch_size, seq_len, num_kv_heads, head_dim = x.shape
        
        if n_rep == 1:
            return x
        
        # Repeat along the head dimension
        x = x.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        return x.reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    
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
            cache_position: Optional cache position for generation
            return_quantization_info: Whether to return quantization loss information
            
        Returns:
            Attention output (and optionally cached key-values if use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states = self.rotary_emb.apply_rotary_pos_emb(query_states, cos, sin)
            key_states = self.rotary_emb.apply_rotary_pos_emb(key_states, cos, sin)
        
        # Handle past key-value states for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)
        
        # Repeat key and value for grouped query attention
        key_states = self._repeat_kv(key_states, self.num_queries_per_kv)
        value_states = self._repeat_kv(value_states, self.num_queries_per_kv)
        
        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key_states = key_states.transpose(1, 2)       # (batch_size, num_heads, seq_len, head_dim)
        value_states = value_states.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
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
