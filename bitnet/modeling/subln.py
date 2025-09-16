"""
Sublayer Normalization (subln) implementation with debugging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SublayerNorm(nn.Module):
    """
    Sublayer Normalization (subln) with learnable scale and bias.
    
    Args:
        hidden_size: Size of the hidden dimension
        eps: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of sublayer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.weight * x_norm + self.bias


class SublayerNormWithResidual(nn.Module):
    """
    Sublayer Normalization with residual connection.
    
    Args:
        hidden_size: Size of the hidden dimension
        eps: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5
    ):
        super().__init__()
        self.norm = SublayerNorm(hidden_size, eps)
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (output from sublayer)
            residual: Residual tensor (original input to sublayer)
            
        Returns:
            Normalized tensor with residual connection
        """
        # Debug logging
        logger.debug(f"SublayerNormWithResidual forward:")
        logger.debug(f"  x shape: {x.shape}, dtype: {x.dtype}")
        logger.debug(f"  residual shape: {residual.shape}, dtype: {residual.dtype}")
        
        # Check if shapes match
        if x.shape != residual.shape:
            logger.error(f"Shape mismatch in residual connection!")
            logger.error(f"  x shape: {x.shape}")
            logger.error(f"  residual shape: {residual.shape}")
            
            # Try to provide more context
            if len(x.shape) == 3 and len(residual.shape) == 3:
                batch_x, seq_x, hidden_x = x.shape
                batch_r, seq_r, hidden_r = residual.shape
                
                if batch_x != batch_r:
                    logger.error(f"  Batch size mismatch: {batch_x} vs {batch_r}")
                if seq_x != seq_r:
                    logger.error(f"  Sequence length mismatch: {seq_x} vs {seq_r}")
                if hidden_x != hidden_r:
                    logger.error(f"  Hidden size mismatch: {hidden_x} vs {hidden_r}")
                    
                # Check if dimensions got swapped
                if hidden_x == seq_r and seq_x == hidden_r:
                    logger.error("  WARNING: It looks like sequence length and hidden size are swapped!")
                    logger.error("  This often happens when tensor dimensions are transposed incorrectly.")
            
            raise RuntimeError(f"Cannot add tensors with shapes {x.shape} and {residual.shape}")
        
        # Perform residual connection
        output = self.norm(x + residual)
        
        logger.debug(f"  output shape: {output.shape}")
        
        return output