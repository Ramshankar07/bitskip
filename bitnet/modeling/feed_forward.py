"""
BitNet feed-forward network implementation with BitLinear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

from .bitlinear import BitLinear
from .kernels import bitnet_kernels, squared_relu_cuda

logger = logging.getLogger(__name__)


class BitFeedForward(nn.Module):
    """
    Feed-forward network using BitLinear layers.
    
    Args:
        config: BitNet configuration
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp_ratio = getattr(config, 'mlp_ratio', 2.0)  # Default to 2.0 for smaller model (power of 2)
        self.intermediate_size = int(self.hidden_size * self.mlp_ratio)
        self.activation_bits = config.activation_bits  # Get from config
        
        # Log dimensions
        logger.info(f"BitFeedForward initialized with hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size} (mlp_ratio={self.mlp_ratio}), activation_bits={self.activation_bits}")
        
        # BitLinear layer for up projection
        self.up_proj = BitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            activation_bits=self.activation_bits  # Pass activation bits
        )
        
        # BitLinear layer for down projection (not H-BitLinear to avoid shape issues)
        self.down_proj = BitLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            activation_bits=self.activation_bits  # Pass activation bits
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def collect_quantization_losses(self) -> Dict[str, torch.Tensor]:
        """
        Collect quantization losses from BitLinear layers.
        
        Returns:
            Dictionary containing quantization loss information
        """
        quantization_info = {}
        
        # Collect from up projection
        if hasattr(self.up_proj, 'compute_quantization_loss'):
            original_weights = self.up_proj.weight
            w_q, w_scale = self.up_proj._weight_quantize(original_weights)
            quant_loss = self.up_proj.compute_quantization_loss(original_weights, w_q * w_scale)
            quantization_info['up_proj_quantization_loss'] = quant_loss
        
        # Collect from down projection
        if hasattr(self.down_proj, 'compute_quantization_loss'):
            original_weights = self.down_proj.weight
            w_q, w_scale = self.down_proj._weight_quantize(original_weights)
            quant_loss = self.down_proj.compute_quantization_loss(original_weights, w_q * w_scale)
            quantization_info['down_proj_quantization_loss'] = quant_loss
        
        return quantization_info
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output hidden states of shape (batch_size, seq_len, hidden_size)
        """
        # Store input shape for verification
        input_shape = hidden_states.shape
        batch_size, seq_len, hidden_size = input_shape
        
        # Up projection
        hidden_states = self.up_proj(hidden_states)
        # Assert up_proj output shape
        assert hidden_states.shape == (batch_size, seq_len, self.intermediate_size), f"Up projection output shape mismatch: {hidden_states.shape}"
        
        # Activation function (Squared ReLU)
        if bitnet_kernels.is_available:
            hidden_states = squared_relu_cuda(hidden_states)
        else:
            hidden_states = torch.relu(hidden_states) ** 2
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Down projection
        hidden_states = self.down_proj(hidden_states)
        
        # Verify output shape matches input shape
        assert hidden_states.shape == input_shape, \
            f"Feed-forward output shape {hidden_states.shape} doesn't match input shape {input_shape}"
        
        return hidden_states