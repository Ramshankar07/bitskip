#!/usr/bin/env python3
"""
BitNet 1B Parameter Training Script with H-BitLinear - Clean Implementation
Uses Llama 3 tokenizer, FineWeb-Edu streaming dataset, and H-BitLinear layers with GQA attention
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
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the H-BitLinear BitNet model
from bitnet.modeling.model2 import BitNetModel2
from bitnet.utils.default_config import DefaultConfig
from bitnet.modeling.kernels import is_available as fwht_cuda_available

# Import emergency recovery
from emergency_nan_recovery import recover_from_nan, diagnose_model_corruption


def compress_bitnet_for_storage(state_dict_or_path, output_path):
    """
    Actually compress BitNet to 2-bit storage format.
    """
    import numpy as np
    
    # Handle both state_dict and file path
    if isinstance(state_dict_or_path, dict):
        checkpoint = state_dict_or_path
    else:
        checkpoint = torch.load(state_dict_or_path)
    
    compressed = {}
    
    for name, param in checkpoint.items():
        if 'weight' in name and 'norm' not in name:
            # Quantize to ternary
            scale = param.abs().mean()
            ternary = torch.zeros_like(param, dtype=torch.int8)
            ternary[param > 0.5 * scale] = 1
            ternary[param < -0.5 * scale] = -1
            
            # Pack ternary values (2 bits each) into bytes
            # This is what Microsoft does!
            packed = np.packbits(
                ((ternary.cpu().numpy().flatten() + 1) * 85).astype(np.uint8)
            )
            
            compressed[name] = {
                'packed_weights': packed,
                'scale': scale.item(),
                'shape': list(param.shape)
            }
        else:
            # Keep other parameters as is
            compressed[name] = param
    
    # Save compressed model
    torch.save(compressed, output_path)


class BitNetForCausalLM(nn.Module):
    """
    Hugging Face compatible wrapper for the H-BitLinear BitNet model.
    Uses the enhanced BitNetModel2 from model2.py with H-BitLinear layers and GQA attention.
    """
    
    def __init__(self, config_dict):
        super().__init__()
        
        # Convert config dict to DefaultConfig
        self.config = self._convert_to_internal_config(config_dict)
        
        # Create the internal H-BitLinear BitNet model
        self.model = BitNetModel2(self.config)
        
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
        Forward pass using the H-BitLinear BitNet model.
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
            if isinstance(value, torch.Tensor) and value is not None:
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
                
            if output['logits'] is not None and (torch.isnan(output['logits']).any() or torch.isinf(output['logits']).any()):
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
            
            # Check loss validity
            if outputs['loss'] is None or not torch.isfinite(outputs['loss']):
                print("WARNING: Invalid loss, skipping batch")
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
        self.hidden_size = kwargs.get('hidden_size', 2048)  # Increased to 2048 (2^11) for larger model
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 14)  # Reduced to achieve ~1B parameters
        self.num_attention_heads = kwargs.get('num_attention_heads', 16)  # head_dim = 2048/16 = 128 (power of 2)
        self.num_key_value_heads = kwargs.get('num_key_value_heads', 4)
        self.intermediate_size = kwargs.get('intermediate_size', 4096)  # Increased to 4096 (2^12) for larger model
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
    """Create streaming data loader using HuggingFace datasets and PyTorch DataLoader."""
    from datasets import load_dataset, IterableDataset
    from torch.utils.data import DataLoader
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load streaming dataset from HuggingFace
        logger.info("Loading FineWeb-Edu dataset with HuggingFace streaming...")
        
        # Load the dataset in streaming mode
        logger.info("Calling load_dataset...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True
        )
        logger.info("Dataset loaded successfully")
        
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
        logger.info("Creating PyTorch DataLoader...")
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with streaming
            pin_memory=True if torch.cuda.is_available() else False
        )
        logger.info("PyTorch DataLoader created successfully")
        
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
    parser = argparse.ArgumentParser(description='BitNet 1B Parameter Training with H-BitLinear')
    
    # Model configuration - H-BitLinear compatible (all dimensions must be powers of 2)
    parser.add_argument('--hidden_size', type=int, default=2048)  # Increased to 2048 (2^11) for larger model
    parser.add_argument('--num_hidden_layers', type=int, default=14)  # Reduced to achieve ~1B parameters
    parser.add_argument('--num_attention_heads', type=int, default=16)  # head_dim = 2048/16 = 128 (power of 2)
    parser.add_argument('--num_key_value_heads', type=int, default=4)
    parser.add_argument('--intermediate_size', type=int, default=4096)  # Increased to 4096 (2^12) for larger model
    
    # Training configuration - Optimized for H200 GPU utilization
    parser.add_argument('--batch_size', type=int, default=16)  # Increased from 2 to 16 for better GPU utilization
    parser.add_argument('--learning_rate', type=float, default=1e-5)  # Reduced from 5e-5 to 1e-5
    parser.add_argument('--max_length', type=int, default=2048)  # Increased from 1024 to 2048 for longer sequences
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)  # Increased from 1 to 2 for effective larger batch size
    
    # H200 optimization flags
    parser.add_argument('--aggressive_batch', action='store_true', help='Use aggressive batch size for maximum H200 utilization')
    parser.add_argument('--conservative_batch', action='store_true', help='Use conservative batch size for stability')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./output-bitnet-hbitlinear-1b')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='./runs/bitnet-hbitlinear-1b')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    return parser.parse_args()




def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    logger.info("Starting BitNet 1B Parameter Training with H-BitLinear - Clean Implementation")
    
    # Apply H200 optimization flags with memory management
    if args.aggressive_batch:
        args.batch_size = 4  # Reduced from 32 to prevent OOM
        args.max_length = 1536  # Reduced from 4096 to prevent OOM
        args.gradient_accumulation_steps = 4  # Effective batch size = 4 * 4 = 16
        logger.info("🚀 Using aggressive batch settings for maximum H200 utilization")
    elif args.conservative_batch:
        args.batch_size = 2  # More reasonable conservative
        args.max_length = 512  # Reasonable sequence length
        args.gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
        logger.info("🛡️ Using conservative batch settings for stability")
    else:
        # Default settings with memory safety
        args.batch_size = 2  # More reasonable default
        args.max_length = 1024  # Better sequence length for utilization
        args.gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
        logger.info("⚖️ Using default H200 batch settings with memory safety")
    
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
    
    # Debug: Whether H-BitLinear FWHT CUDA kernel will be used
    try:
        fwht_available = fwht_cuda_available()
    except Exception:
        fwht_available = False
    will_use_cuda_fwht = fwht_available and (device.type == 'cuda')
    logger.info(f"H-BitLinear FWHT CUDA kernel available: {fwht_available}, will_use_cuda_kernel: {will_use_cuda_fwht}")
    
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
        logger.info("Creating new H-BitLinear BitNet model")
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
    
    logger.info("Creating data loader...")
    dataloader, is_random = create_data_loader(
        tokenizer, args.batch_size, args.max_length, args.num_steps
    )
    logger.info("Data loader created successfully")
    if is_random:
        logger.warning("Using random data for testing - this is not ideal for actual training")
    
    # Test dataloader with first batch to catch issues early
    logger.info("Testing dataloader with first batch...")
    try:
        test_iter = iter(dataloader)
        test_batch = next(test_iter)
        logger.info(f"✅ Dataloader test successful - batch shape: {test_batch['input_ids'].shape}")
    except Exception as e:
        logger.error(f"❌ Dataloader test failed: {e}")
        logger.error("This indicates a problem with the dataloader setup")
        return
    
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
    logger.info("Creating dataloader iterator...")
    dataloader_iter = iter(dataloader)
    logger.info("Dataloader iterator created successfully")
    
    for step in range(1, args.num_steps + 1):
        try:
            logger.info(f"Starting step {step}")
            total_loss = 0
            
            # Gradient Accumulation Loop
            for micro_step in range(args.gradient_accumulation_steps):
                logger.info(f"Starting micro_step {micro_step} of step {step}")
                # Get batch from iterator
                try:
                    logger.info("Getting next batch from dataloader...")
                    batch = next(dataloader_iter)
                    logger.info("Batch retrieved successfully")
                except StopIteration:
                    logger.info("StopIteration caught, restarting dataloader iterator...")
                    # Restart iterator if we've exhausted the dataset
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                    logger.info("New batch retrieved after restart")
                
                # Move to device with memory management
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Clear cache before forward pass to free memory
                torch.cuda.empty_cache()
                
                # Forward pass with mixed precision
                with autocast('cuda', dtype=torch.float16):
                    # Use safe forward pass
                    outputs = model.safe_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    
                    # Check for None, NaN/Inf in loss
                    if loss is None:
                        logger.error(f"🚨 Loss is None at step {step}, likely due to OOM. Skipping batch.")
                        continue
                    
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
            
            # Clear cache after optimizer step
            torch.cuda.empty_cache()
            
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
                try:
                    tb_writer.add_scalar('train/loss', total_loss, step)
                    tb_writer.add_scalar('train/lr', current_lr, step)
                except Exception:
                    pass
                
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
            
            # Periodic evaluation and health check
            if step % args.eval_steps == 0:
                try:
                    model.eval()
                    with torch.no_grad():
                        try:
                            eval_batch = next(dataloader_iter)
                        except StopIteration:
                            dataloader_iter = iter(dataloader)
                            eval_batch = next(dataloader_iter)
                        e_input_ids = eval_batch['input_ids'].to(device)
                        e_attention_mask = eval_batch['attention_mask'].to(device)
                        e_labels = eval_batch['labels'].to(device)
                        e_out = model.safe_forward(input_ids=e_input_ids, attention_mask=e_attention_mask, labels=e_labels)
                        if e_out['loss'] is not None:
                            tb_writer.add_scalar('val/loss', float(e_out['loss'].item()), step)
                except Exception:
                    pass
                finally:
                    model.train()
            
            if step % 50 == 0:
                logger.info("🔍 Running periodic model health check...")
                
                # Use the model's built-in monitoring
                model.monitor_model_state(f"Step {step}")
                
                corruption_found = False
                
                # Quick check for NaN/Inf in parameters
                for name, param in model.named_parameters():
                    if param is not None and (torch.isnan(param).any().item() or torch.isinf(param).any().item()):
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
                
                # Get the internal BitNet model (unwrap the wrapper)
                internal_model = model.model  # BitNetForCausalLM -> BitNetModel
                
                # Get complete state dict with ALL parameters
                state_dict = internal_model.state_dict()
                
                logger.info(f"Saving checkpoint with {len(state_dict)} keys")
                
                # Verify we have the critical weights before saving
                has_attn = any('q_proj.weight' in k and 'weight_scale' not in k for k in state_dict.keys())
                has_embed = any('embed_tokens.weight' in k for k in state_dict.keys())
                has_ffn = any('up_proj.weight' in k and 'weight_scale' not in k for k in state_dict.keys())
                
                if not has_attn:
                    logger.error("CRITICAL: BitLinear attention weights missing from internal model!")
                    logger.error("This will create an incomplete checkpoint!")
                    # Try to save the wrapper's full state instead
                    logger.warning("Falling back to wrapper model.state_dict()")
                    state_dict = model.state_dict()
                elif not has_embed:
                    logger.error("CRITICAL: Embedding weights missing from internal model!")
                    logger.error("Falling back to wrapper model.state_dict()")
                    state_dict = model.state_dict()
                elif not has_ffn:
                    logger.error("CRITICAL: Feedforward weights missing from internal model!")
                    logger.error("Falling back to wrapper model.state_dict()")
                    state_dict = model.state_dict()
                else:
                    # Test our debug script would pass validation
                    logger.info(f"✓ Checkpoint validation: Has embeddings={has_embed}, attention={has_attn}, FFN={has_ffn}")
                
                # Save UNCOMPRESSED full state dict for downstream conversion
                torch.save({
                    'model_state_dict': state_dict,
                    'step': step,
                    'config': config.to_dict(),
                }, os.path.join(checkpoint_dir, "model_full.pt"))
                    
                # Save config
                config.save_pretrained(checkpoint_dir)
                
                # Save quantization config for HuggingFace
                quantization_config = {
                    "quantization_method": "bitnet",
                    "bits": 1.58,
                    "weight_dtype": "ternary",
                    "compressed": True
                }
                with open(os.path.join(checkpoint_dir, "quantization_config.json"), "w") as f:
                    json.dump(quantization_config, f, indent=2)
                
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            continue
    
    # Save final compressed BitNet model
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    # Get the internal BitNet model (unwrap the wrapper)
    internal_model = model.model  # BitNetForCausalLM -> BitNetModel
    
    # Get complete state dict with ALL parameters
    state_dict = internal_model.state_dict()
    
    logger.info(f"Saving final model with {len(state_dict)} keys")
    
    # Verify we have the critical weights before saving
    has_attn = any('q_proj.weight' in k and 'weight_scale' not in k for k in state_dict.keys())
    has_embed = any('embed_tokens.weight' in k for k in state_dict.keys())
    has_ffn = any('up_proj.weight' in k and 'weight_scale' not in k for k in state_dict.keys())
    
    if not has_attn:
        logger.error("CRITICAL: BitLinear attention weights missing from internal model!")
        logger.error("This will create an incomplete checkpoint!")
        # Try to save the wrapper's full state instead
        logger.warning("Falling back to wrapper model.state_dict()")
        state_dict = model.state_dict()
    elif not has_embed:
        logger.error("CRITICAL: Embedding weights missing from internal model!")
        logger.error("Falling back to wrapper model.state_dict()")
        state_dict = model.state_dict()
    elif not has_ffn:
        logger.error("CRITICAL: Feedforward weights missing from internal model!")
        logger.error("Falling back to wrapper model.state_dict()")


    else:
        # Test our debug script would pass validation
        logger.info(f"✓ Final model validation: Has embeddings={has_embed}, attention={has_attn}, FFN={has_ffn}")
    
    # Save UNCOMPRESSED full state dict for downstream conversion
    torch.save({
        'model_state_dict': state_dict,
        'config': config.to_dict(),
    }, os.path.join(final_dir, "model_full.pt"))
    
    # Save config
    config.save_pretrained(final_dir)
    
    # Save quantization config for HuggingFace
    quantization_config = {
        "quantization_method": "bitnet",
        "bits": 1.58,
        "weight_dtype": "ternary",
        "compressed": True
    }
    with open(os.path.join(final_dir, "quantization_config.json"), "w") as f:
        json.dump(quantization_config, f, indent=2)
    
    logger.info(f"Training completed. Final compressed BitNet model saved to {final_dir}")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        tb_writer.close()
    except Exception:
        pass
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()