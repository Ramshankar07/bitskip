"""
Layer skipping mechanism for BitNet.

This module implements the LayerSkip training recipe, which allows for efficient
training and inference by selectively skipping layers based on a curriculum.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass

from ..utils.default_config import DefaultConfig

logger = logging.getLogger(__name__)

@dataclass
class LayerSkipState:
    """State information for layer skipping."""
    hidden_states: torch.Tensor
    skip_mask: torch.Tensor
    layer_outputs: List[torch.Tensor]
    exit_layer: Optional[int]

class LayerSkipping(nn.Module):
    """
    Layer skipping mechanism that implements the LayerSkip training recipe.
    
    This module manages layer skipping during both training and inference,
    implementing an exponential dropout schedule and early exit capabilities.
    
    Args:
        num_layers: Number of layers in the model
        config: BitNet configuration containing layer dropout parameters
    """
    
    def __init__(self, num_layers: int, config: DefaultConfig):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"Number of layers must be positive, got {num_layers}")
            
        self.num_layers = num_layers
        self.config = config
        
        # Initialize layer skipping parameters
        self.base_dropout_rate = getattr(config, 'skip_probability', 0.1)
        self.max_dropout_rate = getattr(config, 'skip_probability', 0.1)
        self.early_exit_scale = getattr(config, 'early_exit_threshold', 0.95)
        
        # Compute LayerSkip dropout probabilities
        self.dropout_probs = self._compute_dropout_probs()
        
        # Initialize curriculum state (simplified)
        self.curriculum_type = "fixed"  # Use fixed curriculum for simplicity
        self.curriculum_settings = {"fixed": {}}  # Empty settings for fixed curriculum
        self.current_curriculum_step = 0
        
        # Register skip masks as buffers
        self.register_buffer('skip_masks', torch.ones((num_layers,), dtype=torch.bool))
        
        logger.info(f"Initialized LayerSkip with {num_layers} layers")
        logger.debug(f"Layer dropout probabilities: {self.dropout_probs}")
    
    def _compute_dropout_probs(self) -> List[float]:
        """
        Compute quadratic layer dropout probabilities using LayerSkip formula.
        Returns:
            List of dropout probabilities for each layer
        """
        probs = []
        for i in range(self.num_layers):
            # Quadratic dropout: p_l = p_max * (l/L)^2
            prob = self.max_dropout_rate * ((i / max(self.num_layers - 1, 1)) ** 2)
            probs.append(prob)
        return probs
    
    def _update_curriculum(self, step: int) -> None:
        """
        Update the curriculum based on the current training step.
        
        Args:
            step: Current training step
        """
        self.current_curriculum_step = step
        if self.curriculum_type == "rotational":
            interval = self.curriculum_settings["rotational"]["rotation_interval"]
            if step % interval == 0:
                self._rotate_curriculum()
        elif self.curriculum_type == "gradual":
            interval = self.curriculum_settings["gradual"]["increase_interval"]
            if step % interval == 0:
                self._increase_curriculum()
    
    def _rotate_curriculum(self) -> None:
        """Rotate the curriculum by shifting dropout probabilities."""
        self.dropout_probs = self.dropout_probs[1:] + [self.dropout_probs[0]]
        logger.debug(f"Rotated curriculum, new probabilities: {self.dropout_probs}")
    
    def _increase_curriculum(self) -> None:
        """Gradually increase the number of active layers."""
        start = self.curriculum_settings["gradual"]["start_layers"]
        end = self.curriculum_settings["gradual"]["end_layers"]
        current = min(start + self.current_curriculum_step // self.curriculum_settings["gradual"]["increase_interval"], end)
        self.dropout_probs = [0.0] * current + self.dropout_probs[current:]
        logger.debug(f"Increased curriculum to {current} layers")
    
    def _generate_skip_masks(self, batch_size: int) -> torch.Tensor:
        """
        Generate per-sample skip masks for the current batch.
        
        Args:
            batch_size: Size of the current batch
            
        Returns:
            Tensor of shape (batch_size, num_layers) containing skip masks
        """
        # Generate skip decisions using standard PyTorch operations
        skip_masks = torch.ones((batch_size, self.num_layers), dtype=torch.bool)
        for i, prob in enumerate(self.dropout_probs):
            if i < self.num_layers - 1:  # Never skip last layer
                skip_masks[:, i] = (torch.rand(batch_size) < prob)
        
        return skip_masks
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        layer_fn: callable,
        return_quantization_info: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with layer skipping.
        
        Args:
            hidden_states: Input hidden states
            layer_idx: Current layer index
            layer_fn: Layer forward function
            **kwargs: Additional arguments for the layer
            
        Returns:
            Tuple of (output_states, skip_mask)
        """
        batch_size = hidden_states.size(0)
        
        # Generate skip decisions for this layer
        if bool(self.training) and layer_idx < self.num_layers - 1:  # FIX: Added bool() and never skip last layer
            skip_prob = self.dropout_probs[layer_idx]
            skip_mask = torch.rand(batch_size, device=hidden_states.device) < skip_prob
        else:
            skip_mask = torch.zeros(batch_size, dtype=torch.bool, device=hidden_states.device)
        
        # Check if any samples should be skipped - FIX: Added .item() and bool()
        if skip_mask.any().item() and bool(self.training):
            # Process only non-skipped samples
            non_skip_indices = (~skip_mask).nonzero(as_tuple=True)[0]
            
            if non_skip_indices.numel() > 0:
                # Extract non-skipped samples
                non_skip_states = hidden_states[non_skip_indices]
                
                # Also extract non-skipped samples from kwargs if they contain batch dimension
                filtered_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.size(0) == batch_size:
                        # This tensor has batch dimension, so we need to filter it
                        if key == 'attention_mask' and value.dim() == 2:
                            # Attention mask: extract non-skipped samples
                            filtered_kwargs[key] = value[non_skip_indices]
                        elif key == 'position_ids' and value.dim() == 2:
                            # Position IDs: extract non-skipped samples
                            filtered_kwargs[key] = value[non_skip_indices]
                        else:
                            # For other tensors with batch dimension, filter them
                            filtered_kwargs[key] = value[non_skip_indices]
                    else:
                        # Keep non-tensor arguments or tensors without batch dimension as-is
                        filtered_kwargs[key] = value
                
                # Process through layer with filtered inputs
                filtered_kwargs['return_quantization_info'] = return_quantization_info
                processed_states = layer_fn(non_skip_states, **filtered_kwargs)
                
                # Handle different return types from layer_fn
                if isinstance(processed_states, tuple):
                    # If layer returns multiple values (e.g., with cache), handle first element
                    processed_states = processed_states[0]
                
                # Reassemble batch: skipped samples keep original states
                output_states = hidden_states.clone()
                output_states[non_skip_indices] = processed_states
                
                return output_states, skip_mask
            else:
                # All samples skipped
                return hidden_states, skip_mask
        else:
            # No skipping or inference mode
            kwargs['return_quantization_info'] = return_quantization_info
            output = layer_fn(hidden_states, **kwargs)
            
            # Handle different return types
            if isinstance(output, tuple):
                # If layer returns multiple values, take first element as hidden states
                return output[0], skip_mask
            else:
                return output, skip_mask

class BitNetWithLayerSkip(nn.Module):
    """
    BitNet model with integrated layer skipping.
    
    This wrapper class manages layer skipping at the model level,
    ensuring proper handling of batch dimensions and KV cache.
    """
    
    def __init__(self, model: nn.Module, config: DefaultConfig):
        super().__init__()
        self.model = model
        self.layer_skipping = LayerSkipping(config.num_hidden_layers, config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        training_step: Optional[int] = None,
        exit_layer: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with layer skipping.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional past key-value pairs
            use_cache: Whether to cache key-value pairs
            training_step: Current training step for curriculum
            exit_layer: Optional layer to exit at
            **kwargs: Additional model arguments
            
        Returns:
            Model outputs with skip information
        """
        # Get initial hidden states
        hidden_states = self.model.embeddings(input_ids)
        
        # Process through layers with skipping
        all_hidden_states = []
        all_skip_masks = []
        present_key_values = []
        
        for i, layer in enumerate(self.model.layers):
            # Get past key-value for this layer
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Process through layer with skipping
            layer_output = self.layer_skipping(
                hidden_states,
                i,
                layer,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                position_ids=position_ids,
                exit_layer=exit_layer,
                training_step=training_step,
                **kwargs
            )
            
            # Handle layer output
            if isinstance(layer_output, tuple):
                hidden_states, present = layer_output
                present_key_values.append(present)
            else:
                hidden_states = layer_output
            
            # Store states and skip masks
            all_hidden_states.append(hidden_states)
            
            # Early exit if specified
            if exit_layer is not None and i >= exit_layer:
                break
        
        # Final layer norm
        hidden_states = self.model.final_layer_norm(hidden_states)
        
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "past_key_values": present_key_values if use_cache else None
        }

def apply_layer_dropout(model: nn.Module, config: DefaultConfig) -> nn.Module:
    """
    Apply layer dropout by wrapping the model with BitNetWithLayerSkip.
    
    Args:
        model: The BitNet model
        config: BitNet configuration
        
    Returns:
        Model wrapped with layer skipping
    """
    return BitNetWithLayerSkip(model, config)