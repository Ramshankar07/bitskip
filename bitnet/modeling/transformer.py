"""
Transformer block implementation for BitNet with SublayerNorm.
"""

from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from ..utils.default_config import DefaultConfig
from .gqa_attention import BitNetGQA
from .feed_forward import BitFeedForward
from .layer_skipping import LayerSkipping
from .subln import SublayerNormWithResidual
from .routing import RoutingModule


class BitTransformerBlock(nn.Module):
    """
    Transformer block with BitLinear layers and SublayerNorm.
    
    Args:
        config: BitNet configuration
    """
    
    def __init__(self, config: DefaultConfig):
        super().__init__()
        self.config = config
        self.activation_bits = config.activation_bits  # Get from config
        
        # Self-attention with SublayerNorm using GQA
        self.self_attn = BitNetGQA(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.attention_probs_dropout_prob,
            activation_bits=self.activation_bits  # Pass activation bits
        )
        self.self_attn_norm = SublayerNormWithResidual(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps
        )
        
        # Feed-forward with SublayerNorm
        self.feed_forward = BitFeedForward(config)
        self.feed_forward_norm = SublayerNormWithResidual(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Routing module for early exit decisions
        self.routing_module = RoutingModule(
            hidden_size=config.hidden_size,
            dropout=config.hidden_dropout_prob
        )
    

    def get_routing_decision(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get routing decision for early exit.
        
        Args:
            hidden_states: Input hidden states
            training: Whether in training mode
            
        Returns:
            p_exit: Exit probabilities
            z_exit: Binary exit decisions
        """
        return self.routing_module(hidden_states, training=training)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the transformer block.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            layer_past: Optional past key-value pairs for generation
            use_cache: Whether to cache key-value pairs
            position_ids: Optional position IDs for RoPE
            return_quantization_info: Whether to return quantization loss information
            
        Returns:
            Layer output (and optionally cached key-values if use_cache=True)
        """
        with torch.autograd.profiler.record_function("TransformerBlock.forward"):
            # Store residual for later
            residual = hidden_states
            # Self attention
            attn_outputs = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                position_ids=position_ids,
                cache_position=cache_position
            )
            
            # Handle both cases: with and without cached key-values
            if isinstance(attn_outputs, tuple):
                # When use_cache=True, attention returns (output, cached_keys_values)
                attn_output, present_key_value = attn_outputs
            else:
                # When use_cache=False, attention returns just the output
                attn_output = attn_outputs
                present_key_value = None
            
            if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
                print(f"ERROR: NaN/Inf detected in attn_output!")
            
            # Apply dropout to attention output
            attn_output = self.dropout(attn_output)
            
            # Apply sublayer norm with residual connection
            hidden_states = self.self_attn_norm(attn_output, residual)
            
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(f"ERROR: NaN/Inf detected in hidden_states after self_attn_norm!")
            
            # Store new residual
            residual = hidden_states
            
            # Feed forward
            ff_output = self.feed_forward(hidden_states)
            
            if torch.isnan(ff_output).any() or torch.isinf(ff_output).any():
                print(f"ERROR: NaN/Inf detected in ff_output!")
            
            # Apply dropout to feed-forward output
            ff_output = self.dropout(ff_output)
            
            # Apply sublayer norm with residual connection
            hidden_states = self.feed_forward_norm(ff_output, residual)
            
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(f"ERROR: NaN/Inf detected in final hidden_states!")
            
            
            
            # Return output with optional cached key-values
            if use_cache:
                return hidden_states, present_key_value
            else:
                return hidden_states