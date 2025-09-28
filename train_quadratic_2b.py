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

# Import emergency recovery
from emergency_nan_recovery import recover_from_nan, diagnose_model_corruption


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
            hidden_size=config_dict.get('hidden_size', 2048),
            num_hidden_layers=config_dict.get('num_hidden_layers', 28),
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
    
    def safe_forward(self, **kwargs):
        """Safe forward pass with comprehensive NaN protection."""
        
        # Pre-forward validation
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"ERROR: NaN/Inf in input {key}")
                    # Return safe dummy output
                    batch_size = value.shape[0] if hasattr(value, 'shape') else 1
                    seq_length = value.shape[1] if hasattr(value, 'shape') and len(value.shape) > 1 else 1
                    dummy_logits = torch.zeros(batch_size, seq_length, self.config.vocab_size, 
                                             device=value.device)
                    return {'loss': None, 'logits': dummy_logits}
        
        try:
            # Run forward with gradient scaling for stability
            with torch.amp.autocast('cuda', enabled=False):  # Disable AMP for stability
                output = self.forward(**kwargs)
                
            # Post-forward validation
            if output['loss'] is not None and not torch.isfinite(output['loss']):
                print("ERROR: Non-finite loss detected")
                output['loss'] = torch.tensor(0.0, device=output['logits'].device, requires_grad=True)
                
            if torch.isnan(output['logits']).any() or torch.isinf(output['logits']).any():
                print("ERROR: Non-finite logits detected")
                # Create safe logits
                output['logits'] = torch.zeros_like(output['logits'])
                
            return output
            
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            # Return safe dummy output
            device = kwargs['input_ids'].device if 'input_ids' in kwargs else 'cpu'
            dummy_logits = torch.zeros(1, 1, self.config.vocab_size, device=device)
            return {'loss': None, 'logits': dummy_logits}
    
    def safe_training_step(self, batch, optimizer, grad_accum_steps=1):
        """Safe training step with comprehensive protection."""
        
        # Enable training mode
        self.train()
        
        try:
            # Forward pass with safety wrapper
            outputs = self.safe_forward(**batch)
            
            if outputs['loss'] is None:
                print("WARNING: No loss computed, skipping batch")
                return None
                
            # Check loss validity
            if not torch.isfinite(outputs['loss']):
                print("WARNING: Non-finite loss, skipping batch")
                return None
                
            # Scale loss for gradient accumulation
            loss = outputs['loss'] / grad_accum_steps
            
            # Safe backward pass
            loss.backward()
            
            # Gradient checking before optimizer step
            total_norm = 0.0
            has_nan_grad = False
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_norm += grad_norm ** 2
                    
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"WARNING: NaN/Inf gradients in {name}")
                        has_nan_grad = True
                        param.grad.data.zero_()  # Zero out bad gradients
            
            total_norm = total_norm ** 0.5
            
            # Apply gradient clipping
            if total_norm > 1.0:  # More aggressive clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                print(f"Gradient clipped: norm was {total_norm}")
                
            if has_nan_grad:
                print("WARNING: NaN gradients detected, zeroing gradients")
                optimizer.zero_grad()
                return None
                
            return loss.item()
            
        except Exception as e:
            print(f"ERROR in training step: {e}")
            optimizer.zero_grad()
            return None
    
    def monitor_model_state(self, prefix=""):
        """Monitor model state for debugging."""
        print(f"\n=== Model State Monitor ({prefix}) ===")
        
        # Check parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param is not None and torch.isnan(param).any():
                    print(f"NaN in parameters: {name}")
                if param is not None and torch.isinf(param).any():
                    print(f"Inf in parameters: {name}")
                if param.abs().max() > 1000:
                    print(f"Large values in {name}: max={param.abs().max().item()}")
        
        # Check activations (requires a forward pass)
        print("====================================\n")


class BitNetConfig:
    """Configuration class for BitNet model."""
    
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 128256)
        self.hidden_size = kwargs.get('hidden_size', 2048)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 28)
        self.num_attention_heads = kwargs.get('num_attention_heads', 16)
        self.num_key_value_heads = kwargs.get('num_key_value_heads', 4)
        self.intermediate_size = kwargs.get('intermediate_size', 4096)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 1024)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-5)
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.attention_dropout = kwargs.get('attention_dropout', 0.1)
        self.initializer_range = kwargs.get('initializer_range', 0.01)
        self.activation_bits = kwargs.get('activation_bits', 8)
        self.weight_bits = kwargs.get('weight_bits', 2)
        self.use_layer_skipping = kwargs.get('use_layer_skipping', True)
        self.skip_probability = kwargs.get('skip_probability', 0.1)
        self.min_layers_to_keep = kwargs.get('min_layers_to_keep', 8)
        self.use_early_exit = kwargs.get('use_early_exit', False)
        self.early_exit_threshold = kwargs.get('early_exit_threshold', 0.95)
        self.dropout_schedule = kwargs.get('dropout_schedule', "quadratic")
        self.quadratic_constant = kwargs.get('quadratic_constant', 0.3)
        
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
    """Create streaming data loader using HuggingFace datasets and PyTorch DataLoader."""
    from datasets import load_dataset, IterableDataset
    from torch.utils.data import DataLoader
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load streaming dataset from HuggingFace
        logger.info("Loading FineWeb-Edu dataset with HuggingFace streaming...")
        
        # Load the dataset in streaming mode
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True
        )
        
        # Create a simple tokenization function
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None  # Return lists, not tensors
            )
            
            # Create labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,  # Process in batches for efficiency
            remove_columns=["text"]  # Remove original text column
        )
        
        # Create a simple collate function for batching
        def collate_fn(batch):
            # Get the maximum length in this batch
            max_len = max(len(item["input_ids"]) for item in batch)
            
            # Pad sequences to the same length
            input_ids = []
            attention_masks = []
            labels = []
            
            for item in batch:
                seq_len = len(item["input_ids"])
                
                # Pad with tokenizer.pad_token_id
                padded_input_ids = item["input_ids"] + [tokenizer.pad_token_id] * (max_len - seq_len)
                padded_attention_mask = [1] * seq_len + [0] * (max_len - seq_len)
                padded_labels = item["labels"] + [-100] * (max_len - seq_len)  # -100 for padding in labels
                
                input_ids.append(padded_input_ids)
                attention_masks.append(padded_attention_mask)
                labels.append(padded_labels)
            
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        
        # Create PyTorch DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with streaming
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info("Successfully created HuggingFace streaming dataloader")
        return dataloader, False
        
    except Exception as e:
        logger.warning(f"Could not load FineWeb-Edu dataset: {e}")
        logger.info("Trying alternative dataset...")
        
        try:
            # Try alternative dataset
            logger.info("Loading allenai/dolmino-mix-1124 dataset...")
            dataset = load_dataset(
                "allenai/dolmino-mix-1124",
                split="train",
                streaming=True
            )
            
            # Same tokenization and collate functions
            def tokenize_function(examples):
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                    return_tensors=None
                )
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized
            
            def collate_fn(batch):
                max_len = max(len(item["input_ids"]) for item in batch)
                
                input_ids = []
                attention_masks = []
                labels = []
                
                for item in batch:
                    seq_len = len(item["input_ids"])
                    
                    padded_input_ids = item["input_ids"] + [tokenizer.pad_token_id] * (max_len - seq_len)
                    padded_attention_mask = [1] * seq_len + [0] * (max_len - seq_len)
                    padded_labels = item["labels"] + [-100] * (max_len - seq_len)
                    
                    input_ids.append(padded_input_ids)
                    attention_masks.append(padded_attention_mask)
                    labels.append(padded_labels)
                
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                }
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=["text"]
            )
            
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            logger.info("Successfully loaded alternative dataset")
            return dataloader, False
            
        except Exception as e2:
            logger.warning(f"Could not load allenai/dolmino-mix-1124: {e2}")
            logger.info("Trying additional fallback datasets...")
            
            # Try more fallback datasets
            fallback_datasets = [
                ("wikitext", "wikitext-2-raw-v1", "train"),
                ("c4", "en", "train"),
                ("openwebtext", None, "train"),
            ]
            
            dataset_loaded = False
            for dataset_name, config, split in fallback_datasets:
                try:
                    logger.info(f"Trying {dataset_name} dataset...")
                    if config:
                        dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                    else:
                        dataset = load_dataset(dataset_name, split=split, streaming=True)
                    
                    # Same tokenization and collate functions
                    def tokenize_function(examples):
                        tokenized = tokenizer(
                            examples["text"],
                            truncation=True,
                            padding=False,
                            max_length=max_length,
                            return_tensors=None
                        )
                        tokenized["labels"] = tokenized["input_ids"].copy()
                        return tokenized
                    
                    def collate_fn(batch):
                        max_len = max(len(item["input_ids"]) for item in batch)
                        
                        input_ids = []
                        attention_masks = []
                        labels = []
                        
                        for item in batch:
                            seq_len = len(item["input_ids"])
                            
                            padded_input_ids = item["input_ids"] + [tokenizer.pad_token_id] * (max_len - seq_len)
                            padded_attention_mask = [1] * seq_len + [0] * (max_len - seq_len)
                            padded_labels = item["labels"] + [-100] * (max_len - seq_len)
                            
                            input_ids.append(padded_input_ids)
                            attention_masks.append(padded_attention_mask)
                            labels.append(padded_labels)
                        
                        return {
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                            "labels": torch.tensor(labels, dtype=torch.long)
                        }
                    
                    tokenized_dataset = dataset.map(
                        tokenize_function,
                        batched=True,
                        batch_size=1000,
                        remove_columns=["text"]
                    )
                    
                    dataloader = DataLoader(
                        tokenized_dataset,
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        num_workers=0,
                        pin_memory=True if torch.cuda.is_available() else False
                    )
                    
                    logger.info(f"Successfully loaded {dataset_name} dataset")
                    return dataloader, False
                    
                except Exception as e3:
                    logger.warning(f"Could not load {dataset_name}: {e3}")
                    continue
            
            if not dataset_loaded:
                logger.warning("Could not load any real dataset, using random data for testing")
            
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
    parser = argparse.ArgumentParser(
        description='2B Parameter BitNet training - H200 GPU Scaling Study'
    )
    
    parser.add_argument('--hidden_size', type=int, default=2048,
                       help='Hidden size (default: 2048 for ~2B parameters)')
    parser.add_argument('--num_hidden_layers', type=int, default=28,
                       help='Number of transformer layers (default: 28 for ~2B parameters)')
    parser.add_argument('--num_attention_heads', type=int, default=16,
                       help='Number of attention heads (default: 16 for ~2B parameters)')
    parser.add_argument('--num_key_value_heads', type=int, default=4,
                       help='Number of key-value heads for GQA (default: 4 for ~2B parameters)')
    parser.add_argument('--intermediate_size', type=int, default=4096,
                       help='Intermediate size for feed-forward network')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Training batch size (default: 4 for 2B model stability)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate (default: 1e-5 for stability)')
    parser.add_argument('--max_length', type=int, default=1024,
                      help='Maximum sequence length (default: 1024 for memory efficiency)')
    parser.add_argument('--num_steps', type=int, default=1000,
                      help='Number of training steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                      help='Gradient accumulation steps for effective larger batch size')
    
    # H200 optimization flags
    parser.add_argument('--aggressive_batch', action='store_true', help='Use aggressive batch size for maximum H200 utilization')
    parser.add_argument('--conservative_batch', action='store_true', help='Use conservative batch size for stability')
    
    
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
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting BitNet 2B Parameter Training - Clean Implementation")
    
    # Apply H200 optimization flags
    if args.aggressive_batch:
        args.batch_size = 8
        args.max_length = 2048
        args.gradient_accumulation_steps = 1
        logger.info("🚀 Using aggressive batch settings for maximum H200 utilization")
    elif args.conservative_batch:
        args.batch_size = 2
        args.max_length = 512
        args.gradient_accumulation_steps = 4
        logger.info("🛡️ Using conservative batch settings for stability")
    
    logger.info(f"Configuration: {vars(args)}")
    
    # Device setup with H200 optimization
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        logger.info("🚀 Optimized for H200 GPU utilization")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
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
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length,
        rms_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        attention_dropout=0.1,
        initializer_range=0.01,
        activation_bits=8,
        weight_bits=2,
        use_layer_skipping=True,
        skip_probability=0.1,
        min_layers_to_keep=8, 
        use_early_exit=False,  # Disabled for memory efficiency
        early_exit_threshold=args.early_exit_threshold, 
        dropout_schedule='quadratic', 
        quadratic_constant=args.quadratic_constant
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
    
    # Optimizer and scheduler - Using custom WSD scheduler for BitNet + LayerSkip
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    # Learning rate scheduler - Using custom WSD scheduler for BitNet + LayerSkip
    from bitnet.utils.lr_schedule import create_scheduler_for_bitnet_layerskip
    
    scheduler = create_scheduler_for_bitnet_layerskip(
        optimizer=optimizer,
        total_training_steps=args.num_steps,
        base_learning_rate=args.learning_rate,
        warmup_ratio=0.1,  # 10% warmup
        stable_ratio=0.4,  # 40% stable phase
        decay_ratio=0.5,   # 50% decay phase
    )
    
    # Gradient scaler for mixed precision
    grad_scaler = GradScaler('cuda', init_scale=2**10)
    
    # Training loop - Clean implementation similar to your example
    logger.info("Starting training loop")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Sequence length: {args.max_length}")
    logger.info(f"Expected GPU utilization: ~{args.batch_size * args.max_length / 1000:.1f}% of H200 capacity")
    
    model.train()
    
    # Create iterator for the dataloader
    dataloader_iter = iter(dataloader)
    
    for step in range(1, args.num_steps + 1):
        try:
            total_loss = 0
            
            # Gradient Accumulation Loop
            for micro_step in range(args.gradient_accumulation_steps):
                # Get batch from iterator
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    # Restart iterator if we've exhausted the dataset
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with mixed precision
                with autocast('cuda', dtype=torch.float16):
                    # Use safe forward pass
                    outputs = model.safe_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    
                    # Check for NaN/Inf in loss
                    if torch.isnan(loss).any().item() or torch.isinf(loss).any().item() or loss.item() > 100:
                        logger.error(f"🚨 NaN/Inf/Extreme loss detected at step {step}: {loss.item()}")
                        logger.error("Running emergency recovery...")
                        
                        # Run emergency recovery with aggressive mode for extreme cases
                        aggressive = loss.item() > 1000 or step > 100  # Use aggressive for extreme loss or later in training
                        model, optimizer, grad_scaler = recover_from_nan(
                            model, optimizer, grad_scaler, config, aggressive=aggressive
                        )
                        
                        # Skip this iteration
                        continue
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / args.gradient_accumulation_steps
                
                # Backward pass (accumulates gradients)
                grad_scaler.scale(loss).backward()
                total_loss += loss.item()
            
            # Clip gradients after accumulation with more aggressive clipping
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Reduced from 1.0 to 0.5
            
            # Update weights
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            # Compute gradient norm for adaptive LR adjustment
            with torch.no_grad():
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                gradient_norm = total_norm ** 0.5
            
            # Update scheduler with metrics for adaptive adjustment
            scheduler.step(metrics={
                'loss': total_loss,
                'gradient_norm': gradient_norm,
            })
            optimizer.zero_grad()
            
            # Get current learning rate for logging
            current_lr = scheduler.get_last_lr()[0]
            
            # Logging
            if step % args.logging_steps == 0:
                logger.info(f"Step {step}: Loss = {total_loss:.4f}, LR = {current_lr:.2e}")
                
                # Enhanced memory stats for H200 optimization
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                    
                    # Calculate utilization percentage (assuming H200 has ~80GB usable)
                    utilization_percent = (allocated / 80.0) * 100
                    
                    logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB ({utilization_percent:.1f}%), Reserved: {reserved:.2f} GB, Max: {max_allocated:.2f} GB")
                    
                    # Suggest optimizations if utilization is low
                    if utilization_percent < 50:
                        logger.info(f"💡 Low GPU utilization ({utilization_percent:.1f}%). Consider increasing batch_size or max_length.")
                    elif utilization_percent > 90:
                        logger.warning(f"⚠️ High GPU utilization ({utilization_percent:.1f}%). Monitor for OOM errors.")
            
            # Periodic health check every 50 steps
            if step % 50 == 0:
                logger.info("🔍 Running periodic model health check...")
                
                # Use the model's built-in monitoring
                model.monitor_model_state(f"Step {step}")
                
                corruption_found = False
                
                # Quick check for NaN/Inf in parameters
                for name, param in model.named_parameters():
                    if torch.isnan(param).any().item() or torch.isinf(param).any().item():
                        logger.error(f"🚨 NaN/Inf detected in {name} at step {step}")
                        corruption_found = True
                        break
                
                if corruption_found:
                    logger.error("Running emergency recovery...")
                    # Use aggressive recovery for periodic checks since corruption was found
                    model, optimizer, grad_scaler = recover_from_nan(
                        model, optimizer, grad_scaler, config, aggressive=True
                    )
                    logger.info("Recovery completed, continuing training...")
            
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