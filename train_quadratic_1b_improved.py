#!/usr/bin/env python3
"""
1B Parameter BitNet Training Script
Clean version without debug prints and emoji
"""

import os
import argparse
import logging
import math
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from safetensors.torch import save_file, load_file
from dotenv import load_dotenv

from bitnet.modeling.model import BitNetModel
from bitnet.data.streaming_loader import create_streaming_dataloader
from bitnet.utils.default_config import DefaultConfig

# Load environment variables
load_dotenv()

# Disable HuggingFace caching
os.environ["HF_DATASETS_CACHE"] = "/dev/null"
os.environ["TRANSFORMERS_CACHE"] = "/dev/null"
os.environ["HF_HOME"] = "/dev/null"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


class BitNetConfig:
    """BitNet configuration compatible with Hugging Face format."""
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 1536,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 1024,
        rms_norm_eps: float = 1e-5,
        hidden_dropout_prob: float = 0.1,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.02,
        activation_bits: int = 8,
        weight_bits: int = 2,
        use_layer_skipping: bool = False,
        skip_probability: float = 0.1,
        use_early_exit: bool = False,
        early_exit_threshold: float = 0.95,
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
        self.use_early_exit = use_early_exit
        self.early_exit_threshold = early_exit_threshold
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_pretrained(self, save_directory: str):
        """Save configuration file."""
        os.makedirs(save_directory, exist_ok=True)
        config_dict = self.to_dict()
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load configuration from file."""
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config_dict.update(kwargs)
            return cls(**config_dict)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")


class BitNetForCausalLM(nn.Module):
    """BitNet model for causal language modeling."""
    
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
        """Convert external config to internal BitNet config."""
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
            use_layer_skipping=bool(config.use_layer_skipping),
            skip_probability=config.skip_probability,
            use_early_exit=bool(config.use_early_exit),
            early_exit_threshold=config.early_exit_threshold
        )
    
    def post_init(self):
        """Initialize weights after model creation."""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for stability."""
        if isinstance(module, nn.Linear):
            std = min(self.config.initializer_range, 0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = min(self.config.initializer_range, 0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, (nn.LayerNorm, nn.modules.normalization.LayerNorm)):
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
        """Forward pass."""
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        
        logits = outputs.logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
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
        """Save model and configuration."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Convert model to FP16 for saving
        model_fp16 = self.half()
        state_dict = model_fp16.state_dict()
        
        # Save in safetensors format
        safetensors_path = os.path.join(save_directory, "model.safetensors")
        save_file(state_dict, safetensors_path)
        
        # Also save in PyTorch format
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from saved files."""
        config = BitNetConfig.from_pretrained(model_path, **kwargs)
        model = cls(config)
        
        # Try loading safetensors first, then PyTorch format
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model file found at {model_path}")
        
        # Fix potential shape mismatches
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith('_scale') and value.dim() == 0:
                fixed_state_dict[key] = value.unsqueeze(0)
            else:
                fixed_state_dict[key] = value
        
        model.load_state_dict(fixed_state_dict, strict=False)
        return model


def setup_logging(log_dir: str):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BitNet 1B Parameter Training')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=1536)
    parser.add_argument('--num_hidden_layers', type=int, default=20)
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--num_key_value_heads', type=int, default=4)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=100)
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./output-1b')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    # Optional features
    parser.add_argument('--use_layer_skipping', action='store_true')
    parser.add_argument('--skip_probability', type=float, default=0.1)
    parser.add_argument('--use_early_exit', action='store_true')
    parser.add_argument('--early_exit_threshold', type=float, default=0.95)
    
    return parser.parse_args()


def train():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting BitNet 1B Parameter Training")
    logger.info(f"Configuration: {vars(args)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
            token=os.getenv("HUGGINGFACE_TOKEN"),
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = len(tokenizer)
        logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    # Create model configuration
    config = BitNetConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length,
        activation_bits=8,
        weight_bits=2,
        use_layer_skipping=args.use_layer_skipping,
        skip_probability=args.skip_probability,
        use_early_exit=args.use_early_exit,
        early_exit_threshold=args.early_exit_threshold
    )
    
    # Initialize or load model
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        try:
            model = BitNetForCausalLM.from_pretrained(args.checkpoint_path)
            logger.info(f"Loaded checkpoint from {args.checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            model = BitNetForCausalLM(config)
    else:
        model = BitNetForCausalLM(config)
        logger.info("Initialized new model")
    
    model.to(device)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size (FP32): {total_params * 4 / (1024**3):.2f} GB")
    logger.info(f"Model size (FP16): {total_params * 2 / (1024**3):.2f} GB")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
        eta_min=1e-6
    )
    
    # Initialize gradient scaler for FP16 training
    scaler = torch.amp.GradScaler('cuda',
        init_scale=2**10,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )
    
    # Create data loader
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
    logger.info("Starting training loop")
    global_step = 0
    training_losses = []
    
    for step in range(args.num_steps):
        try:
            # Clear cache periodically
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get batch
            batch = next(iter(train_dataloader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at step {global_step}, skipping")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Record loss
            training_losses.append(loss.item())
            
            # Logging
            if global_step % args.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {global_step}: Loss = {loss:.4f}, LR = {current_lr:.6e}")
                
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
            # Save checkpoint
            if global_step > 0 and global_step == args.save_steps:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            global_step += 1
            
        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            continue
    
    # Save final model
    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    logger.info(f"Training completed. Final model saved to {final_model_dir}")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Log summary statistics
    if training_losses:
        avg_loss = sum(training_losses) / len(training_losses)
        logger.info(f"Average training loss: {avg_loss:.4f}")
    
    logger.info("Training completed successfully")


if __name__ == '__main__':
    train()