#!/usr/bin/env python3
"""
BitNet 1B Parameter Training Script - Clean Implementation
Uses Llama 3 tokenizer and FineWeb-Edu streaming dataset
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from safetensors.torch import save_file, load_file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the original BitNet model
from bitnet.modeling.model import BitNetModel
from bitnet.utils.default_config import DefaultConfig


class BitNetForCausalLM(nn.Module):
    """
    Hugging Face compatible wrapper for the original BitNet model.
    Uses the fixed BitNetModel from model.py with boolean tensor fixes applied.
    """
    
    def __init__(self, config_dict):
        super().__init__()
        
        # Convert config dict to DefaultConfig
        self.config = self._convert_to_internal_config(config_dict)
        
        # Create the internal BitNet model
        self.model = BitNetModel(self.config)
        
        # The model already has lm_head, so we can use it directly
        self.lm_head = self.model.lm_head
    
    def _convert_to_internal_config(self, config_dict):
        """Convert HF-style config to internal DefaultConfig."""
        return DefaultConfig(
            vocab_size=config_dict.get('vocab_size', 128256),
            hidden_size=config_dict.get('hidden_size', 1536),
            num_hidden_layers=config_dict.get('num_hidden_layers', 20),
            num_attention_heads=config_dict.get('num_attention_heads', 16),
            num_kv_heads=config_dict.get('num_key_value_heads', 4),
            max_position_embeddings=config_dict.get('max_position_embeddings', 1024),
            layer_norm_eps=config_dict.get('layer_norm_eps', 1e-5),
            hidden_dropout_prob=config_dict.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config_dict.get('attention_dropout', 0.1),
            initializer_range=config_dict.get('initializer_range', 0.02),
            activation_bits=config_dict.get('activation_bits', 8),
            weight_bits=config_dict.get('weight_bits', 2),
            # Force boolean conversion to avoid tensor issues
            use_layer_skipping=bool(config_dict.get('use_layer_skipping', False)),
            skip_probability=config_dict.get('skip_probability', 0.0),
            min_layers_to_keep=config_dict.get('min_layers_to_keep', 1),
            use_early_exit=bool(config_dict.get('use_early_exit', False)),
            early_exit_threshold=config_dict.get('early_exit_threshold', 0.0),
            gradient_checkpointing=bool(config_dict.get('gradient_checkpointing', False))
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass using the original BitNet model.
        """
        # Call the internal model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        
        # Return in simple dict format for compatibility
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }


class BitNetConfig:
    """Configuration class for BitNet model."""
    
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 128256)
        self.hidden_size = kwargs.get('hidden_size', 1536)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 20)
        self.num_attention_heads = kwargs.get('num_attention_heads', 16)
        self.intermediate_size = kwargs.get('intermediate_size', 3072)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 1024)
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        
        # Store all kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert to dictionary."""
        return self.__dict__
    
    def save_pretrained(self, save_directory):
        """Save configuration."""
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_data_loader(tokenizer, batch_size, max_length, num_steps):
    """Create streaming data loader with FineWeb-Edu dataset."""
    from bitnet.data.streaming_loader import create_streaming_dataloader
    
    try:
        # Use the streaming loader with FineWeb-Edu dataset
        logger = logging.getLogger(__name__)
        logger.info("Loading FineWeb-Edu dataset with streaming...")
        
        dataloader = create_streaming_dataloader(
            dataset_name="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            streaming=True,
            text_column="text"
        )
        
        logger.info("Successfully created streaming dataloader")
        return dataloader, False
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not load FineWeb-Edu dataset: {e}")
        logger.info("Trying alternative dataset...")
        
        try:
            # Try alternative dataset
            dataloader = create_streaming_dataloader(
                dataset_name="allenai/dolmino-mix-1124",
                subset="default",
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                streaming=True,
                text_column="text"
            )
            logger.info("Successfully loaded alternative dataset")
            return dataloader, False
            
        except Exception as e2:
            logger.warning(f"Could not load alternative dataset: {e2}")
            logger.warning("Using random data for testing")
            
            # Create random data for testing
            class RandomDataLoader:
                def __init__(self, vocab_size, batch_size, max_length, num_steps):
                    self.vocab_size = vocab_size
                    self.batch_size = batch_size
                    self.max_length = max_length
                    self.num_steps = num_steps
                    self.step = 0
                
                def __iter__(self):
                    return self
                
                def __next__(self):
                    if self.step >= self.num_steps:
                        self.step = 0
                    self.step += 1
                    
                    # Generate random sequences
                    input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.max_length))
                    attention_mask = torch.ones_like(input_ids)
                    
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': input_ids.clone()
                    }
            
            return RandomDataLoader(len(tokenizer), batch_size, max_length, num_steps), True


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BitNet 1B Parameter Training')
    
    # Model configuration
    parser.add_argument('--hidden_size', type=int, default=1536)
    parser.add_argument('--num_hidden_layers', type=int, default=20)
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./output-bitnet-1b')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    return parser.parse_args()


def get_cosine_schedule(step, warmup_steps, base_lr):
    """Get cosine learning rate schedule."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    else:
        return base_lr * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * (step - warmup_steps) / (1000 - warmup_steps))))


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting BitNet 1B Parameter Training - Clean Implementation")
    logger.info(f"Configuration: {vars(args)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load Llama 3 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
            token=os.getenv("HUGGINGFACE_TOKEN"),
            use_fast=True,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
        tokenizer.padding_side = 'left'
        vocab_size = len(tokenizer)
        logger.info(f"Loaded Llama 3 tokenizer with vocabulary size: {vocab_size}")
    except Exception as e:
        logger.warning(f"Could not load Llama 3 tokenizer: {e}")
        logger.warning("Falling back to GPT-2 tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            vocab_size = len(tokenizer)
        except:
            logger.error("Could not load any tokenizer")
            return
    
    # Create configuration with Llama 3 vocab size (128256)
    config = BitNetConfig(
        vocab_size=vocab_size,  # This will be 128256 for Llama 3
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length,
        hidden_dropout_prob=0.1,
        initializer_range=0.02
    )
    
    # Create or load model
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        model = BitNetForCausalLM(config.to_dict())
        checkpoint = torch.load(os.path.join(args.checkpoint_path, "model.pt"))
        model.load_state_dict(checkpoint, strict=False)
    else:
        logger.info("Creating new BitNet model")
        model = BitNetForCausalLM(config.to_dict())
    
    model.to(device)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size (FP32): {total_params * 4 / (1024**3):.2f} GB")
    logger.info(f"Model size (FP16): {total_params * 2 / (1024**3):.2f} GB")
    
    # Create data loader with Llama 3 tokenizer
    if not tokenizer:
        logger.error("No tokenizer available, cannot create data loader")
        return
    
    dataloader, is_random = create_data_loader(
        tokenizer, args.batch_size, args.max_length, args.num_steps
    )
    if is_random:
        logger.warning("Using random data for testing - this is not ideal for actual training")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    # Gradient scaler for mixed precision
    grad_scaler = GradScaler('cuda', init_scale=2**10)
    
    # Training loop - Clean implementation similar to your example
    logger.info("Starting training loop")
    
    model.train()
    
    for step in range(1, args.num_steps + 1):
        try:
            total_loss = 0
            
            # Gradient Accumulation Loop
            for micro_step in range(args.gradient_accumulation_steps):
                # Get batch
                batch = next(iter(dataloader))
                
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with mixed precision
                with autocast('cuda', dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / args.gradient_accumulation_steps
                
                # Backward pass (accumulates gradients)
                grad_scaler.scale(loss).backward()
                total_loss += loss.item()
            
            # Clip gradients after accumulation
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
            
            # Update learning rate
            current_lr = get_cosine_schedule(step, args.warmup_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Logging
            if step % args.logging_steps == 0:
                logger.info(f"Step {step}: Loss = {total_loss:.4f}, LR = {current_lr:.2e}")
                
                # Memory stats
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
            # Save checkpoint
            if step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
                
                # Save config
                config.save_pretrained(checkpoint_dir)
                
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            continue
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    config.save_pretrained(final_dir)
    logger.info(f"Training completed. Final model saved to {final_dir}")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()