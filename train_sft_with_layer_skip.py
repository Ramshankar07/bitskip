#!/usr/bin/env python3
"""
CUDA-optimized SFT script for microsoft/bitnet-b1.58-2B-4T-bf16 with layer skipping.
Optimized for single GPU training with efficient memory management.
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import json
import numpy as np
import logging
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    LlamaConfig,
    LlamaForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset, Dataset, IterableDataset
from huggingface_hub import snapshot_download
from pathlib import Path
import safetensors.torch

# Import the existing streaming loader
from bitnet.data.streaming_loader import create_streaming_dataloader, StreamingConfig

# CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bitnet_training.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BitNetLayerSkipConfig:
    """Configuration for BitNet layer skip SFT training."""
    
    # Model specific
    model_name: str = "1BitLLM/bitnet_b1_58-3B"
    
    # Layer skipping parameters (following paper methodology)
    enable_layer_skip: bool = True
    max_skip_rate: float = 0.2  # p_max in paper
    use_exponential_skip: bool = True  # Exponential increase across layers
    skip_curriculum: str = "constant"  # "constant" or "exponential" over time
    
    # Early exit training
    enable_early_exit: bool = True
    exit_loss_scale: float = 0.5  # e_scale in paper
    exit_curriculum: str = "rotational"  # "rotational", "gradual", or "all"
    rotation_interval: int = 4  # For rotational curriculum
    
    # Training parameters
    output_dir: str = "./bitnet-layerskip-sft"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2  # Increased for better GPU utilization
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Reduced since batch size increased
    num_train_epochs: int = 1
    num_epochs: int = 1  # Alias for streaming trainer
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    bf16: bool = True  # Use BF16 for better performance on modern GPUs
    tf32: bool = True  # Enable TF32 for better performance
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch_fused"  # Fused optimizer for better performance
    
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    dataset_text_field: str = "messages"  # Messages field name in dataset
    max_samples: Optional[int] = 10000  # Limit for testing
    preprocessing_num_workers: int = 4  # Parallel data processing
    dataloader_num_workers: int = 2  # DataLoader workers
    
    use_streaming: bool = True  # Enable streaming for large datasets
    streaming_buffer_size: int = 1000  # Buffer size for streaming
    eval_samples: int = 500  # Number of samples for evaluation
    
    max_grad_norm: float = 1.0
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2  # Limit saved checkpoints to save disk space
    seed: int = 42
    
    cuda_device: int = 0  # Which GPU to use
    pin_memory: bool = True  # Pin memory for faster transfer
    non_blocking: bool = True  # Non-blocking memory transfers


class BitNetModelWrapper(PreTrainedModel):
    """
    Custom wrapper for BitNet models that handles non-standard architectures.
    """
    
    config_class = LlamaConfig  # Use LlamaConfig as base
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        self.model = LlamaForCausalLM(config)
        
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        logger.info(f"Initialized BitNet wrapper with {self.num_layers} layers")
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Custom loading method for BitNet models.
        """
        logger.info(f"Loading BitNet model from {model_name_or_path}")
        
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                **kwargs
            )
            logger.info("Successfully loaded BitNet model using AutoModel")
            return model
            
        except Exception as e:
            logger.warning(f"Standard loading failed: {e}")
            logger.info("Attempting custom loading method...")
            
            # Fallback: Create a LLaMA-based model and load weights manually
            return cls._custom_load_bitnet(model_name_or_path, **kwargs)
    
    @classmethod
    def _custom_load_bitnet(cls, model_name_or_path: str, **kwargs):
        """
        Custom loading for BitNet models without proper config files.
        """
        # Download model files
        cache_dir = kwargs.get('cache_dir', None)
        local_path = snapshot_download(
            repo_id=model_name_or_path,
            cache_dir=cache_dir,
            ignore_patterns=["*.md", "*.txt"]
        )
        
        # Try to load config.json
        config_path = Path(local_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            # Use default BitNet 2B configuration
            logger.warning("No config found, using default BitNet 2B configuration")
            config_dict = {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 2048,
                "intermediate_size": 5504,
                "num_attention_heads": 32,
                "num_hidden_layers": 24,
                "num_key_value_heads": 32,
                "vocab_size": 128256,
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-05,
                "rope_scaling": None,
                "rope_theta": 10000.0,
                "torch_dtype": "bfloat16",
            }
        
        # Create config
        config = LlamaConfig(**config_dict)
        
        # Initialize model
        model = cls(config)
        
        # Load weights if available
        weight_files = list(Path(local_path).glob("*.safetensors"))
        if weight_files:
            logger.info(f"Loading weights from {weight_files[0]}")
            state_dict = safetensors.torch.load_file(weight_files[0])
            model.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("No weight files found, using random initialization")
        
        return model
    
    def forward(self, **kwargs):
        """Forward pass through the underlying model."""
        return self.model(**kwargs)
    
    def get_input_embeddings(self):
        """Get input embeddings from the underlying model."""
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        """Get output embeddings from the underlying model."""
        return self.model.get_output_embeddings()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()


class BitNetLayerSkipModel(nn.Module):
    """
    CUDA-optimized BitNet model wrapper with layer skipping capabilities and RoPE support.
    """
    
    def __init__(self, base_model, config: BitNetLayerSkipConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        self.model_config = base_model.config
        self.num_layers = self.model_config.num_hidden_layers
        self.hidden_size = self.model_config.hidden_size
        self.vocab_size = self.model_config.vocab_size
        self.device = torch.device(f'cuda:{config.cuda_device}' if torch.cuda.is_available() else 'cpu')
        
        # RoPE configuration
        self.rope_theta = getattr(self.model_config, 'rope_theta', 10000.0)
        self.max_position_embeddings = getattr(self.model_config, 'max_position_embeddings', 2048)
        self.rope_scaling = getattr(self.model_config, 'rope_scaling', None)
        
        # Initialize RoPE cache
        self._rope_cache = {}
        
        if torch.cuda.is_available():
            self.base_model = self.base_model.to(self.device)
            logger.info(f"Model moved to {self.device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available Memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
            logger.info(f"Total Memory: {torch.cuda.mem_get_info()[1] / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available, using CPU")
        
        self.precomputed_skip_probs = self._precompute_skip_probabilities()
        
        # Training step counter
        self.register_buffer('current_step', torch.tensor(0, dtype=torch.long))
        self.total_steps = 10000  # Will be updated during training
        
        # Pre-allocated tensors
        self.register_buffer('skip_decisions', torch.zeros(self.num_layers, dtype=torch.bool))
        
        # Cache for exit weights
        self.exit_weights_cache = self._precompute_exit_weights()
        
        # using the same LM head for all layers (final logits)
        if hasattr(base_model, 'lm_head'):
            self.lm_head = base_model.lm_head
        else:
            # using the output embeddings
            self.lm_head = base_model.get_output_embeddings()

        # per-layer exit heads for early-exit loss
        if self.config.enable_early_exit:
            self.exit_heads = nn.ModuleList([
                nn.Linear(self.hidden_size, self.vocab_size) for _ in range(self.num_layers)
            ])
        
        logger.info(f"Initialized BitNet with {self.num_layers} layers, hidden_size={self.hidden_size}")
        logger.info(f"Model size: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M parameters")
        logger.info(f"Layer skip enabled: {config.enable_layer_skip}")
        logger.info(f"Early exit enabled: {config.enable_early_exit}")
        logger.info(f"Max skip rate: {config.max_skip_rate}")
        logger.info(f"Exit loss scale: {config.exit_loss_scale}")
        logger.info(f"Skip curriculum: {config.skip_curriculum}")
        logger.info(f"Exit curriculum: {config.exit_curriculum}")
        logger.info(f"RoPE theta: {self.rope_theta}")
        logger.info(f"Max position embeddings: {self.max_position_embeddings}")
    
    def _precompute_skip_probabilities(self) -> torch.Tensor:
        """Pre-compute skip probabilities for all layers to avoid repeated calculation."""
        probs = torch.zeros(self.num_layers)
        if self.config.enable_layer_skip and self.config.use_exponential_skip and self.num_layers > 1:
            for i in range(self.num_layers):
                D_l = np.exp(i * np.log(2) / (self.num_layers - 1)) - 1
                probs[i] = min(D_l * self.config.max_skip_rate, 1.0)
            logger.info(f"Pre-computed skip probabilities: {probs.tolist()}")
        return probs
    
    def _precompute_exit_weights(self) -> torch.Tensor:
        """Pre-compute exit weights for all layers."""
        weights = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                weights[i] = self.config.exit_loss_scale * sum(range(i + 1))
            else:
                weights[i] = (self.num_layers - 1) + self.config.exit_loss_scale * sum(range(self.num_layers - 1))
        return weights
    
    def _get_rope_position_embeddings(self, position_ids: torch.Tensor, hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate RoPE position embeddings for LLaMA compatibility.
        
        Args:
            position_ids: Position IDs tensor of shape (batch_size, seq_len)
            hidden_size: Hidden dimension size
            
        Returns:
            Tuple of (cos, sin) tensors for RoPE
        """
        # Create cache key
        cache_key = (position_ids.shape, position_ids.device, hidden_size)
        
        if cache_key not in self._rope_cache:
            try:
                # Generate inv_freq for RoPE
                inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, hidden_size, 2, dtype=torch.float32, device=position_ids.device) / hidden_size))
                logger.debug(f"inv_freq shape: {inv_freq.shape}")
                
                # Get position values - flatten to 1D for torch.outer
                t = position_ids.float().flatten()  # Shape: (batch_size * seq_len,)
                logger.debug(f"t shape after flatten: {t.shape}")
                
                # Compute frequencies using outer product
                freqs = torch.outer(t, inv_freq)  # Shape: (batch_size * seq_len, hidden_size // 2)
                logger.debug(f"freqs shape after outer: {freqs.shape}")
                
                # Reshape back to original batch and sequence dimensions
                batch_size, seq_len = position_ids.shape
                freqs = freqs.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, hidden_size // 2)
                logger.debug(f"freqs shape after reshape: {freqs.shape}")
                
                # Generate cos and sin
                cos = torch.cos(freqs)
                sin = torch.sin(freqs)
                
                # Cache the results
                self._rope_cache[cache_key] = (cos, sin)
                logger.debug(f"Successfully generated RoPE embeddings: cos={cos.shape}, sin={sin.shape}")
                
            except Exception as e:
                logger.error(f"Error in RoPE generation: {e}")
                logger.error(f"position_ids shape: {position_ids.shape}")
                logger.error(f"hidden_size: {hidden_size}")
                logger.error(f"rope_theta: {self.rope_theta}")
                raise
        
        return self._rope_cache[cache_key]
    
    def _prepare_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], inputs_embeds: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor:
        """
        Prepare attention mask for LLaMA compatibility.
        
        Args:
            attention_mask: Input attention mask
            input_shape: (batch_size, seq_len)
            inputs_embeds: Input embeddings
            past_key_values_length: Length of past key values
            
        Returns:
            Prepared attention mask
        """
        batch_size, seq_length = input_shape
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=inputs_embeds.device))
        
        # Expand to batch dimension
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Handle past key values length
        if past_key_values_length > 0:
            # Create a mask for past positions
            past_mask = torch.ones((batch_size, seq_length, past_key_values_length), device=inputs_embeds.device)
            # Concatenate past and current masks
            causal_mask = torch.cat([past_mask, causal_mask], dim=-1)
        
        # Apply input attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to 4D
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Apply to causal mask
            causal_mask = causal_mask * attention_mask
        
        # Convert to the format expected by LLaMA attention
        causal_mask = causal_mask[:, None, :, :]
        
        return causal_mask
    
    def _set_rope_embeddings(self, layer, cos: torch.Tensor, sin: torch.Tensor) -> None:
        """
        Set RoPE embeddings in a layer's attention module.
        
        Args:
            layer: Transformer layer
            cos: Cosine values for RoPE
            sin: Sine values for RoPE
        """
        try:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                # Set the RoPE embeddings in the attention module
                layer.self_attn.rotary_emb.cos_cached = cos
                layer.self_attn.rotary_emb.sin_cached = sin
            elif hasattr(layer, 'attention') and hasattr(layer.attention, 'rotary_emb'):
                # Alternative structure
                layer.attention.rotary_emb.cos_cached = cos
                layer.attention.rotary_emb.sin_cached = sin
        except Exception as e:
            # If setting RoPE embeddings fails, log warning but continue
            logger.debug(f"Could not set RoPE embeddings in layer: {e}")
            pass
    
    def _get_layer_output(self, layer_outputs) -> torch.Tensor:
        """
        Extract hidden states from layer outputs, handling different output formats.
        
        Args:
            layer_outputs: Output from transformer layer
            
        Returns:
            Hidden states tensor
        """
        if isinstance(layer_outputs, tuple):
            # Standard case: (hidden_states, ...)
            return layer_outputs[0]
        elif isinstance(layer_outputs, torch.Tensor):
            # Direct tensor output
            return layer_outputs
        else:
            # Try to get hidden_states attribute
            if hasattr(layer_outputs, 'hidden_states'):
                return layer_outputs.hidden_states
            elif hasattr(layer_outputs, 'last_hidden_state'):
                return layer_outputs.last_hidden_state
            else:
                # Last resort: assume it's the hidden states
                return layer_outputs
    
    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        CUDA-optimized forward pass with layer skipping and early exit loss.
        """
        
        # Ensure inputs are on the correct device
        if input_ids is not None:
            input_ids = input_ids.to(self.device, non_blocking=self.config.non_blocking)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=self.config.non_blocking)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=self.config.non_blocking)
        
        # During inference, use standard forward with a safe fallback
        if not self.training or (not self.config.enable_layer_skip and not self.config.enable_early_exit):
            try:
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Base model forward failed, falling back to custom pass: {e}")
        
        # Get embeddings - handle both wrapped and unwrapped models
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
            # Standard model structure with embed_tokens
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
        elif hasattr(self.base_model, 'model'):
            # Standard model structure without embed_tokens
            inputs_embeds = self.base_model.model.get_input_embeddings()(input_ids)
        else:
            # BitNet wrapper or direct model
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        hidden_states = inputs_embeds
        
        # Prepare attention mask efficiently
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), 
                device=self.device, 
                dtype=torch.long
            )
        
        # Prepare position embeddings and attention mask for LLaMA compatibility
        position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Generate RoPE position embeddings
        try:
            cos, sin = self._get_rope_position_embeddings(position_ids, self.hidden_size)
            logger.debug(f"Generated RoPE embeddings: cos={cos.shape}, sin={sin.shape}")
        except Exception as e:
            logger.warning(f"Failed to generate RoPE embeddings: {e}")
            logger.warning(f"Position IDs shape: {position_ids.shape}, hidden_size: {self.hidden_size}")
            # Fallback: create dummy embeddings
            cos = torch.ones((batch_size, seq_length, self.hidden_size // 2), device=self.device)
            sin = torch.zeros((batch_size, seq_length, self.hidden_size // 2), device=self.device)
        
        # Prepare attention mask using our custom method
        try:
            attention_mask = self._prepare_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, 0)
            logger.debug(f"Prepared attention mask: {attention_mask.shape}")
        except Exception as e:
            logger.warning(f"Failed to prepare attention mask: {e}")
            # Fallback: use simple attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask is not None else None
        
        # Pre-generate skip decisions for this forward pass (more efficient than per-layer)
        if self.config.enable_layer_skip and self.training:
            skip_random = torch.rand(self.num_layers, device=self.device)
            
            # Apply temporal curriculum if needed
            if self.config.skip_curriculum == "exponential" and self.total_steps > 1:
                S_t = np.exp(self.current_step.item() * np.log(2) / (self.total_steps - 1)) - 1
            else:
                S_t = 1.0
            
            skip_probs = self.precomputed_skip_probs.to(self.device) * S_t
            skip_decisions = skip_random < skip_probs
            
            # Log skip decisions periodically
            if self.current_step.item() % 100 == 0:
                skipped_layers = skip_decisions.sum().item()
                logger.info(f"Step {self.current_step.item()}: Skipped {skipped_layers}/{self.num_layers} layers")
        else:
            skip_decisions = torch.zeros(self.num_layers, dtype=torch.bool, device=self.device)
        
        # Storage for early exit losses (pre-allocate list size)
        exit_losses = []
        exit_indices = []
        
        # Determine which layers need exit loss computation
        if self.config.enable_early_exit and labels is not None:
            compute_exit_at = []
            for layer_idx in range(self.num_layers):
                if self._should_compute_exit_loss(layer_idx):
                    compute_exit_at.append(layer_idx)
        else:
            compute_exit_at = []
        
        # Process through layers with potential skipping
        layers_processed = 0
        for layer_idx in range(self.num_layers):
            if not skip_decisions[layer_idx]:
                layers_processed += 1
                # Apply the transformer layer - handle both wrapped and unwrapped models
                if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
                    layer = self.base_model.model.layers[layer_idx]
                elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model'):
                    # For BitNet wrapper, access layers through the underlying model
                    layer = self.base_model.model.model.layers[layer_idx]
                else:
                    # Direct model access
                    layer = self.base_model.layers[layer_idx]
                
                # Use gradient checkpointing if enabled
                if self.config.gradient_checkpointing and self.training:
                    def create_custom_forward(module, layer_idx):
                        def custom_forward(hidden_states, attention_mask, position_ids):
                            # Generate position embeddings for this layer
                            cos, sin = self._get_rope_position_embeddings(position_ids, self.hidden_size)
                            position_embeddings = (cos, sin)
                            
                            # Call the layer with position_embeddings
                            return module(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=None,
                                output_attentions=False,
                                use_cache=False,
                                position_embeddings=position_embeddings,  # Pass position_embeddings directly
                            )
                        return custom_forward
                    
                    # Note: Pass layer_idx to create_custom_forward
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer, layer_idx),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        use_reentrant=False
                    )
                else:
                    # Non-gradient checkpointing path
                    # Generate position embeddings for this layer
                    cos, sin = self._get_rope_position_embeddings(position_ids, self.hidden_size)
                    position_embeddings = (cos, sin)
                    
                    try:
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                            position_embeddings=position_embeddings,  # Pass position_embeddings
                        )
                    except TypeError as e:
                        # Fallback for different forward signatures
                        logger.debug(f"Standard forward failed, trying without position_embeddings: {e}")
                        try:
                            # Some models might not accept position_embeddings directly
                            layer_outputs = layer(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=None,
                                output_attentions=False,
                                use_cache=False,
                            )
                        except Exception as e2:
                            logger.debug(f"Alternative forward failed, using minimal: {e2}")
                            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
                
                hidden_states = self._get_layer_output(layer_outputs)
            
            # Compute early exit loss if needed (vectorized for efficiency)
            if layer_idx in compute_exit_at:
                # Apply RMS norm and compute logits - handle both wrapped and unwrapped models
                if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'norm'):
                    normed_hidden = self.base_model.model.norm(hidden_states)
                elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model'):
                    # For BitNet wrapper
                    normed_hidden = self.base_model.model.model.norm(hidden_states)
                else:
                    # Direct model access
                    normed_hidden = self.base_model.norm(hidden_states)

                # Prefer per-layer exit heads when available; otherwise fall back to lm_head
                if hasattr(self, 'exit_heads') and layer_idx < len(self.exit_heads):
                    exit_logits = self.exit_heads[layer_idx](normed_hidden)
                else:
                    if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model'):
                        exit_logits = self.base_model.get_output_embeddings()(normed_hidden)
                    else:
                        exit_logits = self.lm_head(normed_hidden)
                
                # Efficient loss computation
                shift_logits = exit_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Use torch.nn.functional for better performance
                exit_loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )
                
                exit_losses.append(exit_loss)
                exit_indices.append(layer_idx)
        
        # Apply final layer norm - handle both wrapped and unwrapped models
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'norm'):
            hidden_states = self.base_model.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model'):
            # For BitNet wrapper
            hidden_states = self.base_model.model.model.norm(hidden_states)
            logits = self.base_model.get_output_embeddings()(hidden_states)
        else:
            # Direct model access
            hidden_states = self.base_model.norm(hidden_states)
            logits = self.lm_head(hidden_states)
        
        # Compute total loss efficiently
        loss = None
        if labels is not None:
            # Main loss from final layer
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            
            # Combine with early exit losses if present
            if exit_losses:
                # Use pre-computed weights
                weights = self.exit_weights_cache[exit_indices].to(self.device)
                final_weight = self.exit_weights_cache[-1]
                
                # Normalize weights
                total_weight = weights.sum() + final_weight
                normalized_weights = weights / total_weight
                main_weight = final_weight / total_weight
                
                # Efficient weighted sum using torch operations
                exit_loss_tensor = torch.stack(exit_losses)
                weighted_exit_loss = (exit_loss_tensor * normalized_weights).sum()
                
                loss = main_weight * main_loss + weighted_exit_loss
            else:
                loss = main_loss
        
        # Log training progress periodically
        if self.training and self.current_step.item() % 50 == 0:
            logger.info(f"Step {self.current_step.item()}: Processed {layers_processed}/{self.num_layers} layers, "
                       f"Exit losses: {len(exit_losses)}, Main loss: {loss.item() if loss is not None else 'N/A':.4f}")
        
        # Increment step counter
        self.current_step += 1
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )
    
    def _should_compute_exit_loss(self, layer_idx: int) -> bool:
        """Optimized exit loss computation check."""
        if not self.config.enable_early_exit:
            return False
        
        if self.config.exit_curriculum == "all":
            return True
        elif self.config.exit_curriculum == "rotational":
            return (layer_idx % self.config.rotation_interval) == (self.current_step.item() % self.config.rotation_interval)
        elif self.config.exit_curriculum == "gradual":
            progress = min(1.0, self.current_step.item() / max(1, self.total_steps // 2))
            enabled_layers = int(progress * self.num_layers)
            return layer_idx >= (self.num_layers - enabled_layers)
        
        return False




class StreamingTrainer:
    """
    Custom trainer for streaming datasets with CUDA optimizations.
    """
    
    def __init__(self, model, config: BitNetLayerSkipConfig, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Create custom streaming loaders that handle message formatting
        self.train_loader = self._create_custom_streaming_loader(
            config.dataset_name,
            config.dataset_split,
            tokenizer,
            config.max_seq_length,
            config.per_device_train_batch_size,
            config.max_samples
        )
        
        self.eval_loader = self._create_custom_streaming_loader(
            config.dataset_name,
            config.dataset_split,
            tokenizer,
            config.max_seq_length,
            config.per_device_eval_batch_size,
            config.eval_samples
        )
        
        # Training parameters
        self.batch_size = config.per_device_train_batch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.max_grad_norm = config.max_grad_norm
        
        # Calculate total steps
        if config.max_samples:
            self.total_steps = config.max_samples // (self.batch_size * self.gradient_accumulation_steps) * self.num_epochs
        else:
            # Estimate for streaming
            self.total_steps = 10000  # Default estimate
        
        self.model.total_steps = self.total_steps
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon
        )
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_steps),
            num_training_steps=self.total_steps
        )
        
        # CUDA optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
            logger.info(f"CUDA optimizations enabled")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        logger.info(f"Streaming trainer initialized")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
    
    def _create_custom_streaming_loader(self, dataset_name, split, tokenizer, max_length, batch_size, max_samples):
        """Create a custom streaming loader that handles message formatting."""
        
        def format_messages(example):
            """Format messages array into training text."""
            messages = example.get("messages", [])
            if not messages:
                return None
            
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
            
            return text.strip() if text.strip() else None
        
        def batch_iterator(dataset, batch_size, max_samples=None):
            """Custom batch iterator with message formatting."""
            batch = []
            yielded = 0
            
            for example in dataset:
                if max_samples is not None and yielded >= max_samples:
                    break
                
                # Format messages
                formatted_text = format_messages(example)
                if formatted_text is None or len(formatted_text.strip()) == 0:
                    continue
                
                batch.append(formatted_text)
                
                if len(batch) == batch_size:
                    # Tokenize batch
                    tokenized = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    tokenized["labels"] = tokenized["input_ids"].clone()
                    yield tokenized
                    yielded += len(batch)
                    batch = []
            
            # Yield remaining batch
            if batch and (max_samples is None or yielded < max_samples):
                tokenized = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                tokenized["labels"] = tokenized["input_ids"].clone()
                yield tokenized
        
        # Load dataset
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True
        )
        
        return batch_iterator(dataset, batch_size, max_samples)
    
    def train(self):
        """Main training loop for streaming data."""
        logger.info("Starting streaming training...")
        
        self.model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            epoch_loss = 0
            step_count = 0
            
            # Use the streaming loader iterator
            for batch in self.train_loader:
                # Move to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Forward pass
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights
                if (step_count + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    total_loss += loss.item() * self.gradient_accumulation_steps
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        logger.info(f"Step {global_step}/{self.total_steps}, Loss: {avg_loss:.4f}")
                        total_loss = 0
                    
                    # Evaluation
                    if global_step % self.config.eval_steps == 0:
                        self.evaluate()
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step)
                
                step_count += 1
                
                # Break if we've reached max samples
                if self.config.max_samples and step_count * self.batch_size >= self.config.max_samples:
                    break
            
            logger.info(f"Epoch {epoch + 1} completed. Steps: {step_count}")
        
        logger.info("Training completed!")
    
    def evaluate(self):
        """Evaluate the model on evaluation dataset."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        eval_loss = 0
        eval_steps = 0
        
        with torch.no_grad():
            # Use the streaming loader iterator for evaluation
            for batch in self.eval_loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                eval_loss += loss.item()
                eval_steps += 1
                
                # Limit evaluation to reasonable number of batches
                if eval_steps >= 10:  # Evaluate on 10 batches max
                    break
        
        avg_eval_loss = eval_loss / max(eval_steps, 1)
        logger.info(f"Evaluation loss: {avg_eval_loss:.4f}")
        
        self.model.train()
    
    def save_checkpoint(self, step):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


class CUDAOptimizedTrainer(Trainer):
    """
    Custom trainer with CUDA optimizations for single GPU training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set total training steps in the model
        if hasattr(self.model, 'total_steps'):
            total_steps = len(self.train_dataset) // self.args.train_batch_size * self.args.num_train_epochs
            self.model.total_steps = total_steps
            logger.info(f"Set total training steps to {total_steps}")
            logger.info(f"Training dataset size: {len(self.train_dataset)}")
            logger.info(f"Batch size: {self.args.train_batch_size}")
            logger.info(f"Epochs: {self.args.num_train_epochs}")
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
            
            logger.info(f"CUDA optimizations enabled")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    def training_step(self, model, inputs):
        """Override training step with CUDA optimizations."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Use mixed precision training context
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.args.bf16 else torch.float16):
            loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()
        
        return loss.detach()
    
    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Ensure all inputs are on CUDA with non-blocking transfer."""
        device = self.args.device
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(device, non_blocking=True)
            else:
                prepared[k] = v
        return prepared


def prepare_dataset(config: BitNetLayerSkipConfig, tokenizer: AutoTokenizer) -> Tuple[IterableDataset, IterableDataset]:
    """
    Prepare training and evaluation datasets with streaming support.
    """
    logger.info(f"Loading dataset: {config.dataset_name}")
    logger.info(f"Dataset split: {config.dataset_split}")
    logger.info(f"Max samples: {config.max_samples}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"Streaming enabled: {config.use_streaming}")
    
    # Always use streaming mode to avoid disk quota issues
    if config.use_streaming:
        logger.info("Loading dataset in streaming mode with message formatting...")
        
        # Create streaming datasets with message formatting
        train_dataset = load_dataset(
            config.dataset_name, 
            split=config.dataset_split,
            streaming=True
        )
        # Try to use test split for eval, fallback to train split
        try:
            eval_dataset = load_dataset(
                config.dataset_name, 
                split="test_sft",
                streaming=True
            )
        except ValueError:
            # If test_sft doesn't exist, use train_sft for eval too
            eval_dataset = load_dataset(
                config.dataset_name, 
                split=config.dataset_split,
                streaming=True
            )
        
        # Format messages for both datasets
        def format_messages(example):
            messages = example.get("messages", [])
            if not messages:
                return {"text": ""}
            
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
            return {"text": text}
        
        train_dataset = train_dataset.map(format_messages)
        eval_dataset = eval_dataset.map(format_messages)
        
        # Filter out empty texts
        train_dataset = train_dataset.filter(lambda x: len(x["text"].strip()) > 0)
        eval_dataset = eval_dataset.filter(lambda x: len(x["text"].strip()) > 0)
        
        logger.info("Streaming dataset loaded successfully with message formatting")
        return train_dataset, eval_dataset
    
    else:
        # Non-streaming mode (original implementation)
        logger.info("Loading dataset in non-streaming mode...")
        if config.dataset_name == "HuggingFaceH4/ultrachat_200k":
            dataset = load_dataset(config.dataset_name, split=config.dataset_split)
            
            # Format for SFT
            def format_chat(example):
                messages = example.get("messages", [])
                text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        text += f"User: {content}\n"
                    elif role == "assistant":
                        text += f"Assistant: {content}\n"
                return {"text": text}
            
            dataset = dataset.map(
                format_chat,
                num_proc=config.preprocessing_num_workers,
                desc="Formatting conversations"
            )
        else:
            dataset = load_dataset(config.dataset_name, split="train")
        
        # Limit samples if specified
        if config.max_samples:
            original_size = len(dataset)
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
            logger.info(f"Limited dataset from {original_size} to {len(dataset)} samples")
        
        # Tokenize with padding for efficient batching
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=config.max_seq_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Add labels (same as input_ids for causal LM)
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized_dataset = tokenized_dataset.map(
            add_labels,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            desc="Adding labels"
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format("torch")
        
        # Split into train and eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=config.seed)
        
        logger.info(f"Dataset split completed: Train={len(split_dataset['train'])}, Eval={len(split_dataset['test'])}")
        
        return split_dataset["train"], split_dataset["test"]


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    logger.info("=" * 60)
    logger.info("Starting BitNet Layer Skip SFT Training")
    logger.info("=" * 60)
    
    # Configuration
    config = BitNetLayerSkipConfig()
    logger.info(f"Configuration loaded: {config}")
    
    # CUDA setup
    if torch.cuda.is_available():
        torch.cuda.set_device(config.cuda_device)
        logger.info(f"Using CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Enable TF32 for better performance
        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for better performance")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    logger.info(f"Random seeds set to: {config.seed}")
    
    
    # Load tokenizer - use LLaMA3 tokenizer
    logger.info("Loading LLaMA3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    tokenizer.padding_side = "left"
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer model max length: {tokenizer.model_max_length}")
    
    # Load base model with memory-efficient settings
    logger.info(f"Loading model: {config.model_name}")
    
    # Load the model directly
    logger.info("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
        device_map={'': config.cuda_device} if torch.cuda.is_available() else "auto",
        low_cpu_mem_usage=True,
    )
    logger.info("Successfully loaded model")
    
    
    
    # Enable gradient checkpointing before wrapping
    if config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Wrap with layer skip model
    logger.info("Wrapping model with layer skip capabilities...")
    model = BitNetLayerSkipModel(base_model, config)
    
    # Log memory usage
    if torch.cuda.is_available():
        logger.info(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    if config.use_streaming:
        logger.info("Using streaming trainer for large dataset...")
        # Create streaming trainer
        trainer = StreamingTrainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        logger.info("Streaming trainer created successfully")
    else:
        logger.info(f"Dataset preparation completed: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
        
        # Training arguments optimized for single GPU
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            adam_epsilon=config.adam_epsilon,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            weight_decay=config.weight_decay,
            bf16=config.bf16,
            tf32=config.tf32,
            optim=config.optim,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=config.save_total_limit,
            dataloader_drop_last=True,  # Drop last incomplete batch
            dataloader_num_workers=config.dataloader_num_workers,
            dataloader_pin_memory=config.pin_memory,
            report_to=[],
            run_name=f"bitnet-layerskip",
            remove_unused_columns=False,
            fp16_full_eval=False,  # Use bf16 for eval too
        )
        
        # Create optimized trainer
        logger.info("Creating CUDA-optimized trainer...")
        trainer = CUDAOptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # Efficient padding
            ),
        )
        logger.info("Trainer created successfully")
    
    # Clean memory before training
    cleanup_memory()
    
    # Train
    logger.info("=" * 60)
    if config.use_streaming:
        logger.info("Starting streaming layer skip SFT training...")
    else:
        logger.info("Starting CUDA-optimized layer skip SFT training...")
    logger.info("=" * 60)
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM! Try reducing batch size or sequence length")
        cleanup_memory()
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        cleanup_memory()
        raise
    
    # Save final model
    logger.info("=" * 60)
    logger.info("Saving model and training artifacts...")
    logger.info("=" * 60)
    logger.info(f"Saving model to {config.output_dir}")
    
    if config.use_streaming:
        # For streaming trainer, save manually
        os.makedirs(config.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(config.output_dir)
    else:
        # For regular trainer, use built-in save
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
    
    logger.info("Model and tokenizer saved successfully")
    
    # Save training configuration
    config_path = os.path.join(config.output_dir, "layer_skip_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    # Save training statistics
    skip_stats = {
        "num_layers": model.num_layers,
        "max_skip_rate": config.max_skip_rate,
        "exit_loss_scale": config.exit_loss_scale,
        "total_steps_trained": model.current_step.item(),
        "final_gpu_memory_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
    }
    
    stats_path = os.path.join(config.output_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(skip_stats, f, indent=2)
    logger.info(f"Training statistics saved to {stats_path}")
    
    logger.info("=" * 60)
    logger.info("Training Summary:")
    logger.info(f"  Total steps trained: {skip_stats['total_steps_trained']}")
    logger.info(f"  Peak GPU memory: {skip_stats['peak_gpu_memory_gb']:.2f} GB")
    logger.info(f"  Final GPU memory: {skip_stats['final_gpu_memory_gb']:.2f} GB")
    logger.info(f"  Model saved to: {config.output_dir}")
    logger.info("=" * 60)
    
    # Cleanup
    logger.info("Cleaning up memory...")
    cleanup_memory()
    logger.info("Cleanup completed")


if __name__ == "__main__":
    main()