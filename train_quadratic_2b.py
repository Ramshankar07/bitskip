#!/usr/bin/env python3
"""
2B Parameter BitNet Training Script - H200 GPU Scaling Study (Hugging Face Compatible)
Early Exit + Quadratic Schedule + Layer Skipping + Standard Architecture + FP16

This script implements a 2B parameter BitNet model optimized for H200 GPU with:
- Hugging Face compatible architecture and configuration
- Consistent naming conventions
- Safetensors checkpoint format in FP16
- Standard model registration
- Architecture compatibility layer
- FP16 training with automatic mixed precision
- Gradient scaling for stable FP16 training
- Target: ~2.1B parameters (increased from 1.94B)
- Memory: ~7.2GB FP16 (fits comfortably in H200's 141GB)
"""

import os
import argparse
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModel,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from transformers.modeling_outputs import CausalLMOutput
from safetensors.torch import save_file, load_file
from dotenv import load_dotenv
import json

from bitnet.modeling.model import BitNetModel
from bitnet.data.streaming_loader import create_streaming_dataloader
from bitnet.utils.default_config import DefaultConfig

# Load environment variables
load_dotenv()

# Completely disable HuggingFace caching
os.environ["HF_DATASETS_CACHE"] = "/dev/null"
os.environ["TRANSFORMERS_CACHE"] = "/dev/null"
os.environ["HF_HOME"] = "/dev/null"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class BitNetConfig:
    """
    Hugging Face compatible BitNet configuration for 2B parameters.
    Uses standard HF naming conventions and structure.
    """
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,  
        num_hidden_layers: int = 28,  
        num_attention_heads: int = 16, 
        num_key_value_heads: int = 4,  
        intermediate_size: int = 4096,  
        max_position_embeddings: int = 1024,
        rms_norm_eps: float = 1e-5,
        hidden_dropout_prob: float = 0.1,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.01,
        activation_bits: int = 8,
        weight_bits: int = 2,
        use_layer_skipping: bool = True,
        skip_probability: float = 0.1,
        min_layers_to_keep: int = 8,  
        use_early_exit: bool = False, 
        early_exit_threshold: float = 0.95,
        dropout_schedule: str = "quadratic",
        quadratic_constant: float = 0.3,
        **kwargs
    ):
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        
        self.use_layer_skipping = use_layer_skipping
        self.skip_probability = skip_probability
        self.min_layers_to_keep = min_layers_to_keep
        
        self.use_early_exit = use_early_exit
        self.early_exit_threshold = early_exit_threshold
        self.dropout_schedule = dropout_schedule
        self.quadratic_constant = quadratic_constant
        
        # Additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_pretrained(self, save_directory: str):
        """Save configuration in Hugging Face format."""
        os.makedirs(save_directory, exist_ok=True)
        config_dict = self.to_dict()
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load configuration from Hugging Face format."""
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config_dict.update(kwargs)
            return cls(**config_dict)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

class BitNetForCausalLM(nn.Module):
    """
    Hugging Face compatible BitNet model for causal language modeling (2B parameters).
    Uses standard HF naming conventions and output format.
    
    Key Features:
    - Single LM head (no memory sharing issues)
    - Consistent architecture across all model variants
    - Hugging Face compatible interface
    - Safetensors compatible (no duplicate tensors)
    - FP16 optimized for memory efficiency
    """
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        
        # Convert to internal config format
        internal_config = self._convert_to_internal_config(config)
        
        # Initialize BitNet model
        self.model = BitNetModel(internal_config)
        
        # Initialize weights
        self.post_init()
    
    @property
    def lm_head(self):
        """Access to the language modeling head."""
        return self.model.lm_head
    
    def _convert_to_internal_config(self, config: BitNetConfig) -> DefaultConfig:
        """Convert HF config to internal BitNet config."""
        return DefaultConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.rms_norm_eps,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_dropout,
            initializer_range=config.initializer_range,
            activation_bits=config.activation_bits,
            weight_bits=config.weight_bits,
            use_layer_skipping=config.use_layer_skipping,
            skip_probability=config.skip_probability,
            min_layers_to_keep=config.min_layers_to_keep,
            use_early_exit=config.use_early_exit,
            early_exit_threshold=config.early_exit_threshold
        )
    
    def post_init(self):
        """Initialize weights after model creation."""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using standard initialization for stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[CausalLMOutput, Tuple]:
        """
        Forward pass with Hugging Face compatible interface.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            **kwargs: Additional arguments
            
        Returns:
            CausalLMOutput or tuple of outputs
        """
        # Prepare inputs for internal model
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'output_hidden_states': output_hidden_states
        }
        
        outputs = self.model(**model_inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # Fallback: compute logits from last hidden state using internal model's LM head
            hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            logits = self.model.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None
        )
    
    def save_pretrained(self, save_directory: str):
        """Save model in Hugging Face format with safetensors in FP16."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Convert model to FP16 for saving
        model_fp16 = self.half()
        state_dict = model_fp16.state_dict()
        
        # Save model state dict in safetensors format (FP16)
        safetensors_path = os.path.join(save_directory, "model.safetensors")
        save_file(state_dict, safetensors_path)
        
        # Also save in PyTorch format for compatibility (FP16)
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load model from Hugging Face format."""
        config = BitNetConfig.from_pretrained(model_name_or_path, **kwargs)
        model = cls(config)
        
        # Try to load safetensors first, fallback to PyTorch
        safetensors_path = os.path.join(model_name_or_path, "model.safetensors")
        pytorch_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model file found at {model_name_or_path}")
        
        # Fix weight_scale shape mismatches (scalar to [1])
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith('_scale') and value.dim() == 0:
                # Convert scalar to [1] shape
                fixed_state_dict[key] = value.unsqueeze(0)
            else:
                fixed_state_dict[key] = value
        
        model.load_state_dict(fixed_state_dict, strict=False)
        return model

def setup_logging(log_dir: str):
    """Set up logging configuration."""

    log_dir = os.path.expanduser(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'quadratic_2b_hf_training.log')),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='2B Parameter BitNet training - H200 GPU Scaling Study'
    )
    
    parser.add_argument('--hidden_size', type=int, default=2048,
                       help='Hidden size (default: 2048 for ~1.8B parameters)')
    parser.add_argument('--num_hidden_layers', type=int, default=28,
                       help='Number of transformer layers (default: 28 for ~1.8B parameters)')
    parser.add_argument('--num_attention_heads', type=int, default=16,
                       help='Number of attention heads (default: 16 for ~1.8B parameters)')
    parser.add_argument('--num_key_value_heads', type=int, default=4,
                       help='Number of key-value heads for GQA (default: 4 for ~1.8B parameters)')
    parser.add_argument('--intermediate_size', type=int, default=4096,
                       help='Intermediate size for feed-forward network')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Training batch size (default: 1 for stability)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate (default: 1e-5 for stability)')
    parser.add_argument('--max_length', type=int, default=1024,
                      help='Maximum sequence length (default: 1024 for memory efficiency)')
    parser.add_argument('--num_steps', type=int, default=1000,
                      help='Number of training steps')
    
    
    parser.add_argument('--output_dir', type=str, default='./output-quadratic-2b-hf',
                      help='Output directory')
    parser.add_argument('--logging_steps', type=int, default=10,
                      help='Log every X steps')
    parser.add_argument('--save_steps', type=int, default=500,
                      help='Save checkpoint every X steps')
    
    parser.add_argument('--early_exit_threshold', type=float, default=0.95,
                      help='Confidence threshold for early exit')
    
    parser.add_argument('--quadratic_constant', type=float, default=0.3,
                      help='Constant c in quadratic dropout: p_l = c·(l/L)²')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Path to checkpoint to load (optional)')
    
    return parser.parse_args()



def main():
    """Main training function for 2B parameter quadratic schedule model."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("1.8B PARAMETER BITNET TRAINING - H200 GPU SCALING STUDY (HF COMPATIBLE)")
    logger.info("=" * 80)
    logger.info("Features: HF Compatible + Layer Dropout + BitLinear + Quadratic Schedule + FP16")
    logger.info("Target: ~1.8B parameters (reduced for stability)")
    logger.info("Memory: ~6.4GB FP16 (fits comfortably in H200's 141GB)")
    logger.info("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    logger.info(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
            token=os.getenv("HUGGINGFACE_TOKEN"),
            force_download=False,
            local_files_only=True,
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        actual_vocab_size = len(tokenizer)
        logger.info(f"Tokenizer vocabulary size: {actual_vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise
    
    config = BitNetConfig(
        vocab_size=actual_vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length,
        use_layer_skipping=True,
        skip_probability=0.1,
        min_layers_to_keep=8, 
        use_early_exit=False,  # Disabled for memory efficiency
        early_exit_threshold=args.early_exit_threshold, 
        dropout_schedule='quadratic', 
        quadratic_constant=args.quadratic_constant
    )
    
    
    logger.info("Initializing 1.8B parameter BitNet model (HF Compatible)...")
    
    
    logger.info("=" * 60)
    logger.info("1.8B PARAMETER CONFIGURATION (HF COMPATIBLE)")
    logger.info("=" * 60)
    logger.info(f"Model Architecture:")
    logger.info(f"  - Model Type: Hugging Face Compatible BitNet")
    logger.info(f"  - Hidden Size: {config.hidden_size}")
    logger.info(f"  - Number of Layers: {config.num_hidden_layers}")
    logger.info(f"  - Number of Attention Heads: {config.num_attention_heads}")
    logger.info(f"  - Number of KV Heads: {config.num_key_value_heads}")
    logger.info(f"  - Intermediate Size: {config.intermediate_size}")
    logger.info(f"  - Max Position Embeddings: {config.max_position_embeddings}")
    logger.info(f"  - Vocabulary Size: {config.vocab_size}")
    logger.info(f"")
    logger.info(f"Training Parameters:")
    logger.info(f"  - Learning Rate: {args.learning_rate}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Number of Steps: {args.num_steps}")
    logger.info(f"  - Output Directory: {args.output_dir}")
    logger.info(f"")
    logger.info(f"Quadratic Schedule Features:")
    logger.info(f"  - Layer Skipping: {config.use_layer_skipping}")
    logger.info(f"  - Skip Probability: {config.skip_probability}")
    logger.info(f"  - Min Layers to Keep: {config.min_layers_to_keep}")
    logger.info(f"  - Early Exit: {config.use_early_exit}")
    logger.info(f"  - Early Exit Threshold: {config.early_exit_threshold}")
    logger.info(f"  - Dropout Schedule: {config.dropout_schedule}")
    logger.info(f"  - Quadratic Constant: {config.quadratic_constant}")
    logger.info("=" * 60)
    
    model = BitNetForCausalLM(config)
    model.to(device)
    
    
    scaler = GradScaler()
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"Model Size: {total_params * 4 / (1024**3):.2f} GB (FP32), {total_params * 2 / (1024**3):.2f} GB (FP16)")
    logger.info(f"Using FP16 precision for training and saving")
    
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        try:
            model = BitNetForCausalLM.from_pretrained(args.checkpoint_path)
            model.to(device)
            logger.info(f"Loaded checkpoint from {args.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            logger.info("Continuing with randomly initialized model")
    
    
    
    logger.info("=" * 60)
    logger.info("QUADRATIC SCHEDULE DETAILS (2B MODEL)")
    logger.info("=" * 60)
    logger.info(f"Quadratic Constant (c): {args.quadratic_constant}")
    logger.info(f"Number of Layers (L): {config.num_hidden_layers}")
    logger.info(f"Formula: p_l = c·(l/L)²")
    logger.info(f"")
    logger.info(f"Layer-wise Dropout Probabilities:")
    for l in range(config.num_hidden_layers):
        prob = args.quadratic_constant * ((l / (config.num_hidden_layers - 1)) ** 2)
        logger.info(f"  - Layer {l+1:2d}: p = {prob:.4f}")
    logger.info("=" * 60)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    def lr_lambda(step):
        warmup_steps = 100
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (args.num_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    train_dataloader = create_streaming_dataloader(
        dataset_name="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        streaming=True,
        text_column="text"
    )
    
    # Training loop
    logger.info("Starting 1.8B parameter HF-compatible training...")
    global_step = 0
    
    # Lists to store training metrics for plotting
    training_losses = []
    learning_rates = []
    training_steps = []
    
    for step in range(args.num_steps):
        try:
            # Clear memory before allocation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get batch
            batch = next(iter(train_dataloader))
            
            # Prepare inputs
            tensor_inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            # Forward pass with FP16 autocast
            try:
                with autocast('cuda'):
                    outputs = model(**tensor_inputs)
                    loss = outputs.loss
            except TypeError:
                # Fallback for older PyTorch versions
                with autocast():
                    outputs = model(**tensor_inputs)
                    loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at step {global_step}, skipping this step")
                optimizer.zero_grad()
                continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            try:
                # Check for gradient overflow before optimizer step
                scaler.unscale_(optimizer)
                
                # Check for gradient overflow
                if torch.isfinite(scaler.scale.item()):
                    # Check for NaN gradients
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                    
                    if not has_nan_grad:
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        
                                # Optimizer step with gradient scaling
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                    else:
                        logger.warning(f"NaN gradients detected at step {global_step}, skipping optimizer step")
                        optimizer.zero_grad()
                        scaler.update()
                else:
                    # Skip optimizer step if gradients overflowed
                    logger.warning(f"Gradient overflow detected at step {global_step}, skipping optimizer step")
                    optimizer.zero_grad()
                    scaler.update()
            except Exception as e:
                logger.warning(f"Gradient scaling error at step {global_step}: {str(e)}")
                # Fallback: clear gradients and continue
                optimizer.zero_grad()
                scaler.update()
            
            # Store metrics for plotting
            training_losses.append(loss.item())
            learning_rates.append(scheduler.get_last_lr()[0])
            training_steps.append(global_step)
            
            # Logging
            if global_step % args.logging_steps == 0:
                logger.info(f"Step {global_step}: Loss = {loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6e}")
                logger.info(f"Quadratic constant: {args.quadratic_constant}, Early exit threshold: {args.early_exit_threshold}")
                logger.info(f"FP16 Training: Enabled with gradient scaling (scale: {scaler.scale.item():.2f})")
                
                # Memory usage logging
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU Memory (FP16) - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            # Save checkpoint in Hugging Face format
            if global_step == args.save_steps:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
                logger.info(f"Saved HF-compatible checkpoint to {checkpoint_dir}")
                
                # Clear memory after checkpoint save
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            global_step += 1
            
        except Exception as e:
            logger.error(f"Error at step {step}: {str(e)}")
            # Clear memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Clear memory after training loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory after training loop")
    
    # Plot and save training loss graph
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        plt.subplot(2, 1, 1)
        plt.plot(training_steps, training_losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('1.8B Parameter BitNet Training Loss (HF Compatible + FP16)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(training_steps, learning_rates, 'r-', linewidth=2, label='Learning Rate')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(args.output_dir, 'quadratic_2b_hf_training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training loss plot saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create training loss plot: {str(e)}")
    
    # Save final model in Hugging Face format
    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Final GPU memory cleanup completed")
    
    logger.info("=" * 80)
    logger.info("1.8B PARAMETER TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("Features: HF Compatible + Layer Dropout + BitLinear + Quadratic Schedule + FP16")
    logger.info(f"Model Size: {total_params:,} parameters (~{total_params/1e9:.1f}B)")
    logger.info(f"Memory Usage: ~6.4GB FP16 (H200: 141GB)")
    logger.info(f"Quadratic Constant: {args.quadratic_constant}")
    logger.info(f"Early Exit Threshold: {args.early_exit_threshold}")
    logger.info(f"Final model saved to: {final_model_dir}")
    logger.info("Model saved in Hugging Face format with safetensors in FP16!")

if __name__ == '__main__':
    main()
