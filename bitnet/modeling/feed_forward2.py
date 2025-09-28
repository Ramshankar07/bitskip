"""
BitNet feed-forward network implementation with H-BitLinear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

from .h_bitlinear import HBitLinear

logger = logging.getLogger(__name__)


class BitFeedForward2(nn.Module):
    """
    Feed-forward network using H-BitLinear layers.
    
    Args:
        config: BitNet configuration
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp_ratio = getattr(config, 'mlp_ratio', 2.0)  # Default to 2.0 for smaller model (power of 2)
        self.intermediate_size = int(self.hidden_size * self.mlp_ratio)
        self.activation_bits = config.activation_bits  # Get from config
        
        # Ensure dimensions are powers of 2 for H-BitLinear
        if not (self.hidden_size & (self.hidden_size - 1) == 0):
            raise ValueError(f"hidden_size ({self.hidden_size}) must be a power of 2 for H-BitLinear")
        if not (self.intermediate_size & (self.intermediate_size - 1) == 0):
            raise ValueError(f"intermediate_size ({self.intermediate_size}) must be a power of 2 for H-BitLinear")
        
        # Log dimensions
        logger.info(f"BitFeedForward2 initialized with hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size} (mlp_ratio={self.mlp_ratio}), activation_bits={self.activation_bits}")
        
        # H-BitLinear layer for up projection
        self.up_proj = HBitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            activation_bits=self.activation_bits
        )
        
        # H-BitLinear layer for down projection
        self.down_proj = HBitLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            activation_bits=self.activation_bits
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    
    
    def forward(self, hidden_states: torch.Tensor, return_quantization_info: bool = False) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            return_quantization_info: Whether to return quantization loss information
            
        Returns:
            Output hidden states of shape (batch_size, seq_len, hidden_size)
        """
        # Store input shape for verification
        input_shape = hidden_states.shape
        batch_size, seq_len, hidden_size = input_shape
        
        # Up projection with H-BitLinear
        hidden_states = self.up_proj(hidden_states)
        # Assert up_proj output shape
        assert hidden_states.shape == (batch_size, seq_len, self.intermediate_size), f"Up projection output shape mismatch: {hidden_states.shape}"
        
        # Activation function (Squared ReLU)
        hidden_states = torch.relu(hidden_states) ** 2
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Down projection with H-BitLinear
        hidden_states = self.down_proj(hidden_states)
        
        # Verify output shape matches input shape
        assert hidden_states.shape == input_shape, \
            f"Feed-forward output shape {hidden_states.shape} doesn't match input shape {input_shape}"
        
        
        
        return hidden_states 