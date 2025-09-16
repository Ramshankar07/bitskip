"""
BitNet model implementation WITHOUT layer skipping and early exit.
Clean baseline model for ablation studies.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from .transformer import BitTransformerBlock
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


class BitNetModel0(nn.Module):
    """
    BitNet model WITHOUT layer skipping and early exit capabilities.
    
    This is a clean baseline model for ablation studies, implementing
    a standard transformer architecture without any optimization features.
    
    Args:
        config: BitNet configuration
    """
    
    def __init__(self, config: DefaultConfig):
        super().__init__()
        self.config = config
        logger.info(f"Initializing BitNet Model0 (baseline) with config: {config}")
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        logger.debug(f"Initialized token embeddings with shape: ({config.vocab_size}, {config.hidden_size})")
        
        # Position embeddings
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        logger.debug(f"Initialized position embeddings with shape: ({config.max_position_embeddings}, {config.hidden_size})")
        
        # Transformer layers (standard, no skipping)
        self.layers = nn.ModuleList([
            BitTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        logger.info(f"Initialized {config.num_hidden_layers} transformer layers (no skipping)")
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Standard LM head (no early exit)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        logger.debug(f"Initialized LM head with shape: ({config.hidden_size}, {config.vocab_size})")
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.info("Model weights initialized")
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
        if self.gradient_checkpointing:
            logger.info("Gradient checkpointing enabled")
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of a module.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def collect_quantization_losses(self) -> Dict[str, torch.Tensor]:
        """
        Collect quantization losses from all BitLinear layers in the model.
        
        Returns:
            Dictionary containing quantization loss information
        """
        quantization_info = {}
        
        # Collect from transformer layers
        for layer_idx, layer in enumerate(self.layers):
            layer_quant_info = layer.collect_quantization_losses()
            if layer_quant_info:
                for key, value in layer_quant_info.items():
                    quantization_info[f"layer_{layer_idx}_{key}"] = value
        
        return quantization_info
    
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
            
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing with explicit use_reentrant=False
                return torch.utils.checkpoint.checkpoint(
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
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[List[Tuple[torch.FloatTensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        exit_layer: Optional[int] = None,  # Kept for compatibility but ignored
        training_step: Optional[int] = None,  # Kept for compatibility but ignored
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        return_quantization_info: bool = False,  # NEW: for joint loss computation
    ) -> Union[Tuple, Dict]:
        """
        Forward pass of the baseline model (no layer skipping, no early exit).
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            layer_past: Optional past key/value states (ignored in baseline)
            use_cache: Whether to use KV caching (ignored in baseline)
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dictionary
            exit_layer: Ignored in baseline model
            training_step: Ignored in baseline model
            labels: Optional labels for computing loss
            cache_position: Ignored in baseline model
            past_key_values: Ignored in baseline model
            return_quantization_info: Whether to return quantization loss information
            
        Returns:
            Model outputs
        """
        with torch.autograd.profiler.record_function("BitNetModel0.forward"):
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
            
            # Initialize layer outputs
            all_hidden_states = [] if output_hidden_states else None
            
            # Process ALL layers sequentially (no skipping, no early exit)
            logger.debug(f"Processing all {self.config.num_hidden_layers} layers sequentially")
            
            for layer_idx in range(self.config.num_hidden_layers):
                # Get layer function
                layer_fn = self._get_layer_fn(layer_idx)
                
                # Apply layer (no skipping logic)
                hidden_states = layer_fn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
                
                # Store hidden states if requested
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
                
                # No early exit loss computation in baseline model
            
            # Apply final layer norm
            hidden_states = self.layer_norm(hidden_states)
            
            # Compute logits
            logits = self.lm_head(hidden_states)
            
            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # No early exit losses in baseline model
            
            if not return_dict:
                return tuple(v for v in [logits, all_hidden_states, loss] if v is not None)

            # Collect quantization information if requested
            quantization_info = None
            if return_quantization_info:
                quantization_info = self.collect_quantization_losses()

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                hidden_states=all_hidden_states,
                past_key_values=None  # No KV cache in baseline implementation
            )
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text using the baseline model.
        
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

    def standard_inference(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Standard inference without any optimizations.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            use_cache: Whether to use KV caching (ignored in baseline)
            
        Returns:
            Model logits
        """
        self.eval()
        with torch.no_grad():
            # Forward pass through all layers
            output = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False,
                use_cache=use_cache,
            )
            return output.logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the baseline model.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "BitNetModel0 (Baseline)",
            "num_layers": self.config.num_hidden_layers,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "features": {
                "layer_skipping": False,
                "early_exit": False,
                "gradient_checkpointing": self.gradient_checkpointing,
                "optimizations": "None (baseline)"
            }
        }
    
    def benchmark_forward_pass(
        self,
        input_ids: torch.LongTensor,
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark the forward pass performance of the baseline model.
        
        Args:
            input_ids: Input token IDs
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        self.eval()
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.forward(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
            
            with torch.no_grad():
                _ = self.forward(input_ids)
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
            else:
                # CPU timing
                import time
                start = time.time()
                with torch.no_grad():
                    _ = self.forward(input_ids)
                end = time.time()
                times.append(end - start)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate tokens per second
        batch_size, seq_length = input_ids.shape
        tokens_per_sec = (batch_size * seq_length) / avg_time if avg_time > 0 else 0
        
        return {
            "model_type": "BitNetModel0 (Baseline)",
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "batch_size": batch_size,
            "sequence_length": seq_length,
            "timing": {
                "average_time": avg_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "times": times
            },
            "performance": {
                "tokens_per_second": tokens_per_sec,
                "throughput": f"{tokens_per_sec:.2f} tokens/sec"
            }
        }
