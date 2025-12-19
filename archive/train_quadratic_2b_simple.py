#!/usr/bin/env python3
"""
BitNet 2B Parameter Training Script - Clean Minimal Implementation
Streamlined version for easy understanding and modification
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import BitNet components
from bitnet.modeling.model import BitNetModel
from bitnet.utils.default_config import DefaultConfig


class SimpleBitNet2B(nn.Module):
    """Simplified BitNet 2B wrapper for training."""
    
    def __init__(self, config_dict):
        super().__init__()
        self.config = self._convert_config(config_dict)
        self.model = BitNetModel(self.config)
        self.lm_head = self.model.lm_head
    
    def _convert_config(self, config_dict):
        """Convert config dict to DefaultConfig for 2B model."""
        return DefaultConfig(
            vocab_size=config_dict.get('vocab_size', 128256),
            hidden_size=config_dict.get('hidden_size', 2048),  # Larger for 2B
            num_hidden_layers=config_dict.get('num_hidden_layers', 24),  # More layers
            num_attention_heads=config_dict.get('num_attention_heads', 16),
            num_kv_heads=config_dict.get('num_key_value_heads', 4),
            max_position_embeddings=config_dict.get('max_position_embeddings', 1024),
            layer_norm_eps=config_dict.get('layer_norm_eps', 1e-5),
            hidden_dropout_prob=config_dict.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config_dict.get('attention_dropout', 0.1),
            initializer_range=config_dict.get('initializer_range', 0.02),
            activation_bits=config_dict.get('activation_bits', 8),
            weight_bits=config_dict.get('weight_bits', 2),
            use_layer_skipping=bool(config_dict.get('use_layer_skipping', False)),
            skip_probability=config_dict.get('skip_probability', 0.0),
            min_layers_to_keep=config_dict.get('min_layers_to_keep', 1),
            use_early_exit=bool(config_dict.get('use_early_exit', False)),
            early_exit_threshold=config_dict.get('early_exit_threshold', 0.0),
            gradient_checkpointing=bool(config_dict.get('gradient_checkpointing', False))
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }


def create_simple_config_2b(args):
    """Create simple configuration for 2B model."""
    return {
        'vocab_size': args.vocab_size,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'num_attention_heads': args.num_attention_heads,
        'intermediate_size': args.intermediate_size,
        'max_position_embeddings': args.max_length,
        'hidden_dropout_prob': 0.1,
        'initializer_range': 0.02,
        'weight_bits': 2,
        'activation_bits': 8,
        'use_layer_skipping': args.use_layer_skipping,
        'use_early_exit': args.use_early_exit,
    }


def create_simple_dataloader(tokenizer, batch_size, max_length, num_steps):
    """Create simple random data loader for testing."""
    class SimpleDataLoader:
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
    
    return SimpleDataLoader(len(tokenizer), batch_size, max_length, num_steps)


def setup_logging():
    """Setup simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple BitNet 2B Training')
    
    # Model configuration - 2B parameters
    parser.add_argument('--vocab_size', type=int, default=128256)
    parser.add_argument('--hidden_size', type=int, default=2048)  # Larger for 2B
    parser.add_argument('--num_hidden_layers', type=int, default=24)  # More layers
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--intermediate_size', type=int, default=4096)  # Larger FFN
    
    # Training configuration - Adjusted for 2B model
    parser.add_argument('--batch_size', type=int, default=2)  # Smaller batch for 2B
    parser.add_argument('--learning_rate', type=float, default=5e-5)  # Lower LR for larger model
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)  # More accumulation
    
    # Features
    parser.add_argument('--use_layer_skipping', action='store_true', help='Enable layer skipping')
    parser.add_argument('--use_early_exit', action='store_true', help='Enable early exit')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output-bitnet-2b-simple')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=50)
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting Simple BitNet 2B Training")
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
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
        vocab_size = len(tokenizer)
        logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}")
        logger.warning("Using default vocab size")
        vocab_size = args.vocab_size
    
    # Create model
    config = create_simple_config_2b(args)
    config['vocab_size'] = vocab_size
    
    model = SimpleBitNet2B(config)
    model.to(device)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Model size (FP32): {total_params * 4 / (1024**3):.2f} GB")
    logger.info(f"Model size (FP16): {total_params * 2 / (1024**3):.2f} GB")
    
    # Create data loader
    dataloader = create_simple_dataloader(
        tokenizer, args.batch_size, args.max_length, args.num_steps
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    # Gradient scaler
    grad_scaler = GradScaler('cuda', init_scale=2**10)
    
    # Training loop
    logger.info("Starting training loop")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    model.train()
    
    dataloader_iter = iter(dataloader)
    
    for step in range(1, args.num_steps + 1):
        try:
            total_loss = 0
            
            # Gradient accumulation
            for micro_step in range(args.gradient_accumulation_steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                with autocast('cuda', dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    
                    # Check for NaN/Inf
                    if loss is not None and (torch.isnan(loss).any() or torch.isinf(loss).any()):
                        logger.error(f"NaN/Inf loss at step {step}: {loss.item()}")
                        continue
                    
                    loss = loss / args.gradient_accumulation_steps
                
                # Backward pass
                grad_scaler.scale(loss).backward()
                total_loss += loss.item()
            
            # Gradient clipping
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
            
            # Logging
            if step % args.logging_steps == 0:
                logger.info(f"Step {step}: Loss = {total_loss:.4f}")
                
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU Memory: {allocated:.2f} GB")
            
            # Save checkpoint
            if step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'config': config,
                }, os.path.join(checkpoint_dir, "model.pt"))
                
                # Save config
                with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            continue
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, os.path.join(final_dir, "model.pt"))
    
    with open(os.path.join(final_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training completed. Final model saved to {final_dir}")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
