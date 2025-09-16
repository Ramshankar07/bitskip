"""
Routing Module for learnable early exit decisions.

This module implements a gating network that produces routing probabilities
for early exit decisions at each layer. Uses straight-through estimator
for differentiable sampling during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RoutingModule(nn.Module):
    """
    Routing module that produces exit probabilities for early exit decisions.
    
    Architecture: Gate_l(x) = σ(Linear(ReLU(Linear(LayerNorm(x)))))
    
    Args:
        hidden_size: Input feature dimension
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Two-layer MLP for gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # First linear layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),  # Output single probability
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.gate_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the routing module.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            training: Whether in training mode (affects sampling strategy)
            
        Returns:
            p_exit: Exit probabilities of shape (batch_size, seq_len, 1)
            z_exit: Binary exit decisions of shape (batch_size, seq_len, 1)
        """
        # Normalize input
        x = self.layer_norm(hidden_states)
        
        # Compute exit probabilities
        p_exit_logits = self.gate_mlp(x)  # (batch_size, seq_len, 1)
        p_exit = torch.sigmoid(p_exit_logits)  # (batch_size, seq_len, 1)
        
        if training:
            # During training: Sample from Bernoulli distribution
            # Use straight-through estimator for differentiable sampling
            z_exit = self._straight_through_sampling(p_exit)
        else:
            # During inference: Threshold-based decision
            z_exit = (p_exit > 0.5).float()
        
        return p_exit, z_exit
    
    def _straight_through_sampling(self, p: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator for differentiable sampling from Bernoulli distribution.
        
        Args:
            p: Bernoulli probabilities of shape (batch_size, seq_len, 1)
            
        Returns:
            Sampled binary decisions with straight-through gradients
        """
        # Sample from Bernoulli distribution
        z_hard = torch.bernoulli(p)
        
        # Straight-through estimator: use hard samples in forward pass,
        # but soft probabilities for gradients
        z_soft = p
        
        # During forward pass, use hard samples
        # During backward pass, gradients flow through soft probabilities
        z = z_hard.detach() + z_soft - z_soft.detach()
        
        return z


class RoutingLoss(nn.Module):
    """
    Routing loss that encourages efficient early exit decisions.
    
    Implements target cost loss to encourage the model to achieve
    a desired average exit layer.
    """
    
    def __init__(self, target_exit_layer: float = 6.0, num_layers: int = 12):
        super().__init__()
        self.target_exit_layer = target_exit_layer
        self.num_layers = num_layers
        
    def forward(self, p_exit_list: list, layer_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute routing loss based on exit probabilities.
        
        Args:
            p_exit_list: List of exit probabilities for each layer
            layer_weights: Optional weights for each layer
            
        Returns:
            Routing loss tensor
        """
        if not p_exit_list:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Compute expected exit layer
        expected_exit_layer = self._compute_expected_exit_layer(p_exit_list, layer_weights)
        
        # Target cost loss: encourage expected exit layer to be close to target
        routing_loss = F.mse_loss(expected_exit_layer, torch.tensor(self.target_exit_layer, 
                                                                   device=expected_exit_layer.device))
        
        return routing_loss
    
    def _compute_expected_exit_layer(self, p_exit_list: list, layer_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute expected exit layer based on exit probabilities.
        
        The probability of NOT exiting by layer l is the product of staying:
        p_continue = ∏(1 - p_exit_i) for i < l
        
        The probability of exiting at layer l is:
        p_exit_at_l = p_exit_l * p_continue
        
        Expected exit layer = Σ(l * p_exit_at_l)
        """
        if not p_exit_list:
            return torch.tensor(0.0)
        
        # Stack all exit probabilities: (num_layers, batch_size, seq_len, 1)
        p_exit_tensor = torch.stack(p_exit_list, dim=0)  # (num_layers, batch_size, seq_len, 1)
        batch_size, seq_len = p_exit_tensor.shape[1], p_exit_tensor.shape[2]
        
        # Compute probability of continuing to each layer
        p_continue = torch.ones_like(p_exit_tensor[0])  # (batch_size, seq_len, 1)
        p_exit_at_layer = []
        
        for l in range(len(p_exit_list)):
            if l > 0:
                p_continue = p_continue * (1 - p_exit_tensor[l-1])
            
            p_exit_at_l = p_exit_tensor[l] * p_continue
            p_exit_at_layer.append(p_exit_at_l)
        
        # Stack probabilities: (num_layers, batch_size, seq_len, 1)
        p_exit_at_layer = torch.stack(p_exit_at_layer, dim=0)
        
        # Create layer indices: (num_layers, 1, 1, 1)
        layer_indices = torch.arange(len(p_exit_list), dtype=torch.float32, 
                                   device=p_exit_tensor.device).view(-1, 1, 1, 1)
        
        # Apply layer weights if provided
        if layer_weights is not None:
            layer_weights = layer_weights.view(-1, 1, 1, 1)
            p_exit_at_layer = p_exit_at_layer * layer_weights
        
        # Compute expected exit layer
        expected_exit_layer = torch.sum(layer_indices * p_exit_at_layer, dim=0)  # (batch_size, seq_len, 1)
        
        # Average over sequence length and batch
        expected_exit_layer = expected_exit_layer.mean()
        
        return expected_exit_layer
