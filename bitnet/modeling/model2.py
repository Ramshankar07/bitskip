"""
BitNet model implementation with H-BitLinear layers and layer skipping.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from .layer_skipping import LayerSkipping, LayerSkipState
from .transformer2 import BitTransformerBlock2
from ..utils.default_config import DefaultConfig

# Configure logging
logger = logging.getLogger(__name__)

def validate_tensor(tensor: torch.Tensor, name: str, expected_shape: Optional[Tuple[int, ...]] = None, expected_dtype: Optional[torch.dtype] = None) -> None:
    """
    Validate tensor properties and log them.
    
    Args:
        tensor: Tensor to validate
        name: Name of the tensor for logging
        expected_shape: Expected shape of the tensor
        expected_dtype: Expected dtype of the tensor
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    logger.debug(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
    
    if expected_shape is not None and tensor.shape != expected_shape:
        raise ValueError(f"{name} has shape {tensor.shape}, expected {expected_shape}")
    
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")


def compute_early_exit_loss_per_layer(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    lm_head: nn.Module,
    layer_idx: int,
    skip_mask: torch.Tensor,
    curriculum_mask: torch.Tensor  # Which layers have early exit enabled
) -> Optional[torch.Tensor]:
    """Compute early exit loss for samples that didn't skip this layer."""
    
    if not curriculum_mask[layer_idx].item():
        return None
        
    # Only compute loss for non-skipped samples
    active_mask = ~skip_mask
    if not active_mask.any().item():
        return None
        
    # Get logits for active samples
    active_states = hidden_states[active_mask]
    active_targets = target_ids[active_mask]
    logits = lm_head(active_states)
    
    if torch.isnan(logits).any().item() or torch.isinf(logits).any().item():
        print(f"ERROR: NaN/Inf detected in early exit logits for layer {layer_idx}!")
        return None
    
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        active_targets.view(-1),
        reduction='mean'
    )
    
    if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
        print(f"ERROR: NaN/Inf detected in early exit loss for layer {layer_idx}!")
        return None
    
    return loss


def compute_early_exit_loss(
    hidden_states_list: List[torch.Tensor],
    targets: torch.Tensor,
    lm_head: nn.Module,
    iteration: int,
    curriculum_fn,
    escale: float = 1.0
) -> torch.Tensor:
    """Safe early exit loss computation with NaN protection."""
    
    # Add input validation
    if targets is None or targets.numel() == 0:
        return torch.tensor(0.0, device=targets.device if targets is not None else 'cpu', requires_grad=True)
    
    L = len(hidden_states_list)
    weights = [sum(k+1 for k in range(l+1)) for l in range(L)]
    weight_sum = sum(weights)
    normalized_weights = [w/weight_sum for w in weights]
    
    total_loss = 0.0
    valid_loss_count = 0
    
    for l, hidden_states in enumerate(hidden_states_list):
        if not curriculum_fn(l, iteration):
            continue
            
        # Check for NaN in hidden states
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            continue
            
        logits = lm_head(hidden_states)
        
        # Comprehensive NaN check
        if (torch.isnan(logits).any() or torch.isinf(logits).any() or 
            logits.abs().max() > 1e6):  # Also check for extreme values
            continue
        
        # Safe cross entropy with clamping
        logits = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Skip if targets contain invalid values
        if targets_flat.min() < 0 or targets_flat.max() >= logits.size(-1):
            continue
            
        try:
            layer_loss = F.cross_entropy(logits, targets_flat, reduction='mean')
            
            if torch.isfinite(layer_loss):
                weighted_loss = normalized_weights[l] * layer_loss
                total_loss += weighted_loss
                valid_loss_count += 1
        except:
            continue
    
    if valid_loss_count == 0:
        return torch.tensor(0.0, device=targets.device, requires_grad=True)
    
    final_loss = total_loss * escale
    
    # Final safety check
    if not torch.isfinite(final_loss):
        return torch.tensor(0.0, device=targets.device, requires_grad=True)
    
    return final_loss


class BitNetModel2(nn.Module):
    """
    BitNet model with H-BitLinear layers, layer skipping capabilities and early exit loss.
    
    This model implements a transformer architecture with H-BitLinear layers, layer skipping and
    early exit capabilities, optimized for efficient training and inference.
    
    Args:
        config: BitNet configuration
    """
    
    def __init__(self, config: DefaultConfig):
        super().__init__()
        self.config = config
        logger.info(f"Initializing BitNet model with H-BitLinear layers and config: {config}")
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        logger.debug(f"Initialized token embeddings with shape: ({config.vocab_size}, {config.hidden_size})")
        
        # Position embeddings
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        logger.debug(f"Initialized position embeddings with shape: ({config.max_position_embeddings}, {config.hidden_size})")
        
        # Transformer layers with H-BitLinear
        self.layers = nn.ModuleList([
            BitTransformerBlock2(config)
            for _ in range(config.num_hidden_layers)
        ])
        logger.info(f"Initialized {config.num_hidden_layers} transformer layers with H-BitLinear")
        
        # Layer skipping mechanism
        self.layer_skipping = LayerSkipping(
            num_layers=config.num_hidden_layers,
            config=config
        )
        logger.info("Initialized layer skipping mechanism")
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Shared LM head for early exit (using regular Linear for output projection)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        logger.debug(f"Initialized LM head with shape: ({config.hidden_size}, {config.vocab_size})")
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.info("Model weights initialized")
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
        if self.gradient_checkpointing:
            logger.info("Gradient checkpointing enabled")
        
        # Early exit curriculum mask
        self.register_buffer(
            'early_exit_curriculum',
            torch.ones(config.num_hidden_layers, dtype=torch.bool)
        )
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights with extreme caution for quantized models.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            # MUCH more conservative initialization for quantized models
            std = min(self.config.initializer_range, 0.002)  # Reduced from 0.01 to 0.002
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Even more conservative for embeddings
            std = min(self.config.initializer_range, 0.001)  # Further reduced
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm safely
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    
    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing."""
        self.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    
    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        logger.info("Gradient checkpointing disabled")
    
    def _get_layer_fn(self, layer_idx: int) -> callable:
        """
        Get the forward function for a layer, with gradient checkpointing if enabled.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Forward function for the layer
        """
        layer = self.layers[layer_idx]
        
        def layer_fn(hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
            # Filter out parameters that shouldn't be passed to the layer
            layer_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in ['exit_layer', 'training_step', 'labels']
            }
            
            if self.gradient_checkpointing and bool(getattr(self, 'training', False)):
                # Use gradient checkpointing with explicit use_reentrant=False
                return checkpoint(
                    layer,
                    hidden_states,
                    use_reentrant=False,
                    **layer_kwargs
                )
            return layer(hidden_states, **layer_kwargs)
            
        return layer_fn
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[List[Tuple[torch.FloatTensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        exit_layer: Optional[int] = None,
        training_step: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,  # NEW9
    ) -> Union[Tuple, Dict]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            layer_past: Optional past key/value states
            use_cache: Whether to use KV caching
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dictionary
            exit_layer: Optional layer to exit at
            training_step: Current training step for curriculum updates
            labels: Optional labels for computing loss
            return_quantization_info: Whether to return quantization loss information
            
        Returns:
            Model outputs
        """
        with torch.autograd.profiler.record_function("BitNetModel2.forward"):
            # Validate inputs
            validate_tensor(input_ids, "input_ids", expected_dtype=torch.long)
            if attention_mask is not None:
                validate_tensor(attention_mask, "attention_mask", expected_shape=input_ids.shape)
            if position_ids is not None:
                validate_tensor(position_ids, "position_ids", expected_shape=input_ids.shape, expected_dtype=torch.long)
            if labels is not None:
                validate_tensor(labels, "labels", expected_shape=input_ids.shape, expected_dtype=torch.long)
            
            # Get sequence length
            batch_size, seq_length = input_ids.shape

            # Defensive check: ensure sequence length does not exceed max_position_embeddings
            if seq_length > self.config.max_position_embeddings:
                raise ValueError(f"Input sequence length {seq_length} exceeds model's max_position_embeddings {self.config.max_position_embeddings}. Reduce your input length or retrain the model.")
            
            # Generate position IDs if not provided
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Get embeddings
            inputs_embeds = self.embed_tokens(input_ids)
            position_embeddings = self.embed_positions(position_ids)
            hidden_states = inputs_embeds + position_embeddings
            
            if torch.isnan(hidden_states).any().item() or torch.isinf(hidden_states).any().item():
                print(f"ERROR: NaN/Inf detected in hidden_states after embeddings!")
            
            # Initialize layer outputs
            all_hidden_states = []  # Always collect for early exit loss
            early_exit_losses = []
            
            # Process each layer
            for layer_idx in range(self.config.num_hidden_layers):
                if torch.isnan(hidden_states).any().item() or torch.isinf(hidden_states).any().item():
                    print(f"ERROR: NaN/Inf detected in hidden_states before layer {layer_idx}!")
                    break
                
                # Get layer function
                layer_fn = self._get_layer_fn(layer_idx)
                
                # Apply layer skipping
                try:
                    hidden_states, skip_mask = self.layer_skipping(
                        hidden_states=hidden_states,
                        layer_idx=layer_idx,
                        layer_fn=layer_fn,
                        attention_mask=attention_mask,
                        layer_past=layer_past[layer_idx] if layer_past is not None else None,
                        use_cache=use_cache,
                        position_ids=position_ids,
                        exit_layer=exit_layer,
                        training_step=training_step,
                    )
                except ValueError as e:
                    if "too many values to unpack" in str(e):
                        print(f"DEBUG: Layer skipping error at layer {layer_idx}: {e}")
                        # Get the actual return value to debug
                        result = self.layer_skipping(
                            hidden_states=hidden_states,
                            layer_idx=layer_idx,
                            layer_fn=layer_fn,
                            attention_mask=attention_mask,
                            layer_past=layer_past[layer_idx] if layer_past is not None else None,
                            use_cache=use_cache,
                            position_ids=position_ids,
                            exit_layer=exit_layer,
                            training_step=training_step,
                        )
                        print(f"DEBUG: Actual return type: {type(result)}")
                        if isinstance(result, tuple):
                            print(f"DEBUG: Tuple length: {len(result)}")
                        raise e
                    else:
                        raise e
                
                if torch.isnan(hidden_states).any().item() or torch.isinf(hidden_states).any().item():
                    print(f"ERROR: NaN/Inf detected in hidden_states after layer {layer_idx}!")
                    break  # Stop processing to prevent further NaN propagation
                
                # Store hidden states if requested
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
                
                # Compute early exit loss if not skipped and early exit is enabled
                if bool(getattr(self, 'training', False)) and bool(getattr(self.config, 'use_early_exit', False)) and labels is not None:
                    # Add safety check for labels tensor
                    if isinstance(labels, torch.Tensor) and labels.numel() > 0:
                        with torch.autograd.profiler.record_function("EarlyExitLoss"):
                            layer_loss = compute_early_exit_loss_per_layer(
                                hidden_states=hidden_states,
                                target_ids=labels,
                                lm_head=self.lm_head,
                                layer_idx=layer_idx,
                                skip_mask=skip_mask,
                                curriculum_mask=self.early_exit_curriculum
                            )
                            if layer_loss is not None:
                                early_exit_losses.append((layer_idx, layer_loss))
                
                # Early exit if requested
                if exit_layer is not None and layer_idx >= exit_layer:
                    break
            
            # Apply final layer norm
            hidden_states = self.layer_norm(hidden_states)
            
            # Compute logits
            logits = self.lm_head(hidden_states)
            
            if torch.isnan(logits).any().item() or torch.isinf(logits).any().item():
                print(f"ERROR: NaN/Inf detected in logits!")
            
            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                    print(f"ERROR: NaN/Inf detected in main loss!")
                
                # Add early exit losses if any and early exit is enabled
                if bool(getattr(self, 'training', False)) and early_exit_losses and bool(getattr(self.config, 'use_early_exit', False)):
                    early_exit_loss = compute_early_exit_loss(
                        all_hidden_states, # Pass all hidden states for loss computation
                        labels,
                        self.lm_head,
                        training_step, # Pass training_step for curriculum
                        lambda l, _: self.early_exit_curriculum[l], # Pass curriculum_fn
                        self.config.early_exit_threshold
                    )
                    loss = loss + early_exit_loss
            
            if not return_dict:
                return tuple(v for v in [logits, all_hidden_states, loss] if v is not None)
            
            # Only return all_hidden_states if output_hidden_states is True
            if not output_hidden_states:
                all_hidden_states = None

            # Create the output object
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                hidden_states=all_hidden_states,
                past_key_values=None  # We don't use KV cache in this implementation
            )
            
            
            return output
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length of generated sequence
            num_return_sequences: Number of sequences to return
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        self.eval()
        with torch.no_grad():
            return self._generate(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                **kwargs
            )
    
    def _generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        num_return_sequences: int,
        **kwargs
    ) -> torch.LongTensor:
        """
        Internal generation method.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length of generated sequence
            num_return_sequences: Number of sequences to return
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        # Implementation of generation logic
        # This is a placeholder - actual implementation would depend on specific requirements
        raise NotImplementedError("Generation not implemented")

    def early_exit_inference(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        exit_layer: Optional[int] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Efficient early exit inference.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            exit_layer: Layer to exit at (default: use all layers)
            use_cache: Whether to use KV caching
            
        Returns:
            Model logits
        """
        self.eval()
        with torch.no_grad():
            # Forward pass up to exit_layer
            output = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                exit_layer=exit_layer,
                training=False,
                use_cache=use_cache,
            )
            return output.logits
    
    def self_speculative_decode(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        exit_layer: int = 4,
        num_speculations: int = 4,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Self-speculative decoding with early exit.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            exit_layer: Layer to use for draft generation
            num_speculations: Number of tokens to speculate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeated tokens
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end of sequence token
            
        Returns:
            Generated token IDs
        """
        logger.info(f"Starting self-speculative decoding with exit_layer={exit_layer}, num_speculations={num_speculations}")
        
        # Use default values if not provided
        if pad_token_id is None:
            pad_token_id = 0  # Default padding token ID
        if eos_token_id is None:
            eos_token_id = 2  # Default EOS token ID
            
        generated = input_ids
        batch_size = input_ids.shape[0]
        
        # Initialize attention mask
        attention_mask = torch.ones_like(generated)
        
        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            # Draft stage with early exit
            draft_tokens = []
            draft_logits = []
            
            for _ in range(num_speculations):
                with torch.no_grad():
                    # Get draft predictions
                    outputs = self.forward(
                        generated,
                        attention_mask=attention_mask,
                        exit_layer=exit_layer,
                        training=False
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        for i in range(batch_size):
                            for token_id in set(generated[i].tolist()):
                                next_token_logits[i, token_id] /= repetition_penalty
                    
                    # Apply top-k filtering
                    if top_k is not None:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p filtering
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    draft_tokens.append(next_token)
                    draft_logits.append(next_token_logits)
                    generated = torch.cat([generated, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Verify stage with full model
            with torch.no_grad():
                outputs = self.forward(
                    generated,
                    attention_mask=attention_mask,
                    training=False
                )
                target_logits = outputs.logits[:, -num_speculations:, :]
                
                # Verify and accept/reject draft tokens
                accepted_tokens = []
                for i in range(num_speculations):
                    draft_logit = draft_logits[i]
                    target_logit = target_logits[:, i, :]
                    
                    # Compute acceptance probability
                    draft_prob = torch.softmax(draft_logit, dim=-1)
                    target_prob = torch.softmax(target_logit, dim=-1)
                    
                    # Accept if target probability is higher
                    accept_prob = torch.minimum(
                        torch.ones_like(target_prob),
                        target_prob / (draft_prob + 1e-8)
                    )
                    
                    # Sample whether to accept
                    accept = torch.rand_like(accept_prob) < accept_prob
                    
                    if accept.any().item():
                        # Accept token
                        accepted_tokens.append(draft_tokens[i])
                    else:
                        # Reject and resample from target
                        probs = torch.softmax(target_logit, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        accepted_tokens.append(next_token)
                        break  # Stop after first rejection
                
                # Update generated sequence
                if len(accepted_tokens) > 0:
                    generated = torch.cat([generated[:, :-num_speculations]] + accepted_tokens, dim=1)
                    attention_mask = torch.ones_like(generated)
            
            # Check for EOS token
            if (generated == eos_token_id).any().item():
                logger.info("EOS token generated, stopping")
                break
        
        logger.info(f"Self-speculative decoding completed, generated {generated.shape[1] - input_ids.shape[1]} tokens")
        return generated 