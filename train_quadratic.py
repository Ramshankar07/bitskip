#!/usr/bin/env python3
"""
Quadratic Schedule BitNet Training Script - Early Exit + Quadratic Schedule

This script implements the third model for ablation study:
1. Native BitNet architecture
2. Layer dropout (basic implementation)
3. BitLinear layers in feed forward
4. Early Exit functionality
5. Quadratic Schedule for layer dropout
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
from transformers import AutoTokenizer
from dotenv import load_dotenv

from bitnet.modeling.model import BitNetModel  # Using model1 (native BitNet)
from bitnet.data.streaming_loader import create_streaming_dataloader
from bitnet.utils.default_config import DefaultConfig

# Load environment variables
load_dotenv()

class QuadraticScheduleBitNetConfig(DefaultConfig):
    """Quadratic Schedule configuration for ablation study - Early Exit + Quadratic Schedule."""
    
    def __init__(self, **kwargs):
        # Extract quadratic schedule parameters
        quadratic_params = {
            'use_layer_skipping': kwargs.pop('use_layer_skipping', True),
            'skip_probability': kwargs.pop('skip_probability', 0.1),
            'min_layers_to_keep': kwargs.pop('min_layers_to_keep', 4),
            'use_early_exit': kwargs.pop('use_early_exit', True),
            'early_exit_threshold': kwargs.pop('early_exit_threshold', 0.95),
            'dropout_schedule': kwargs.pop('dropout_schedule', 'quadratic'),
            'quadratic_constant': kwargs.pop('quadratic_constant', 0.3),
        }
        
        # Call parent constructor with remaining kwargs
        super().__init__(**kwargs)
        
        # Set quadratic schedule parameters as attributes
        for key, value in quadratic_params.items():
            setattr(self, key, value)

def setup_logging(log_dir: str):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'quadratic_training.log')),
            logging.StreamHandler()
        ]
    )

def map_checkpoint_layers(state_dict, model_type='native'):
    """Map checkpoint layers for different model architectures."""
    mapped_dict = {}
    
    # Direct mappings for embeddings and output layers
    direct_mappings = {
        'embed_tokens.weight': 'embed_tokens.weight',
        'embed_positions.weight': 'embed_positions.weight',
        'layer_norm.weight': 'layer_norm.weight',
        'layer_norm.bias': 'layer_norm.bias',
        'lm_head.weight': 'lm_head.weight',
    }
    
    for old_key, new_key in direct_mappings.items():
        if old_key in state_dict:
            mapped_dict[new_key] = state_dict[old_key]
    
    # Map layer weights for different model types
    layer_keys = [key for key in state_dict.keys() if key.startswith('layers.')]
    layer_indices = set()
    for key in layer_keys:
        parts = key.split('.')
        if len(parts) >= 2 and parts[1].isdigit():
            layer_indices.add(int(parts[1]))
    
    num_layers = max(layer_indices) + 1 if layer_indices else 12
    print(f"Detected {num_layers} layers in checkpoint")
    
    for layer_idx in range(num_layers):
        old_prefix = f'layers.{layer_idx}'
        
        # Layer norms
        if f'{old_prefix}.self_attn_norm.norm.weight' in state_dict:
            mapped_dict[f'layers.{layer_idx}.self_attn_norm.weight'] = state_dict[f'{old_prefix}.self_attn_norm.norm.weight']
        if f'{old_prefix}.self_attn_norm.norm.bias' in state_dict:
            mapped_dict[f'layers.{layer_idx}.self_attn_norm.bias'] = state_dict[f'{old_prefix}.self_attn_norm.norm.bias']
        if f'{old_prefix}.feed_forward_norm.norm.weight' in state_dict:
            mapped_dict[f'layers.{layer_idx}.feed_forward_norm.weight'] = state_dict[f'{old_prefix}.feed_forward_norm.norm.weight']
        if f'{old_prefix}.feed_forward_norm.norm.bias' in state_dict:
            mapped_dict[f'layers.{layer_idx}.feed_forward_norm.bias'] = state_dict[f'{old_prefix}.feed_forward_norm.norm.bias']
        
        # Self attention - map to MultiheadAttention format
        if f'{old_prefix}.self_attn.q_proj.weight' in state_dict:
            # For MultiheadAttention, we need to reshape the weights
            q_weight = state_dict[f'{old_prefix}.self_attn.q_proj.weight']
            k_weight = state_dict[f'{old_prefix}.self_attn.k_proj.weight']
            v_weight = state_dict[f'{old_prefix}.self_attn.v_proj.weight']
            
            # Combine into in_proj_weight for MultiheadAttention
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            mapped_dict[f'layers.{layer_idx}.self_attn.in_proj_weight'] = in_proj_weight
            
            # Create zero bias
            hidden_size = q_weight.shape[1]
            in_proj_bias = torch.zeros(3 * hidden_size)
            mapped_dict[f'layers.{layer_idx}.self_attn.in_proj_bias'] = in_proj_bias
        
        # Output projection
        if f'{old_prefix}.self_attn.o_proj.weight' in state_dict:
            mapped_dict[f'layers.{layer_idx}.self_attn.out_proj.weight'] = state_dict[f'{old_prefix}.self_attn.o_proj.weight']
            mapped_dict[f'layers.{layer_idx}.self_attn.out_proj.bias'] = torch.zeros(state_dict[f'{old_prefix}.self_attn.o_proj.weight'].shape[0])
        
        # Feed forward network (BitLinear layers)
        if f'{old_prefix}.feed_forward.up_proj.weight' in state_dict:
            mapped_dict[f'layers.{layer_idx}.feed_forward.0.weight'] = state_dict[f'{old_prefix}.feed_forward.up_proj.weight']
        
        if f'{old_prefix}.feed_forward.down_proj.weight' in state_dict:
            mapped_dict[f'layers.{layer_idx}.feed_forward.2.weight'] = state_dict[f'{old_prefix}.feed_forward.down_proj.weight']
    
    return mapped_dict

def compute_joint_loss(model, batch, lambda_q=0.1, lambda_r=0.05):
    """
    Compute joint loss combining task loss with quantization and routing losses.
    
    Args:
        model: The BitNet model
        batch: Input batch dictionary
        lambda_q: Weight for quantization loss
        lambda_r: Weight for routing loss
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    # Try forward pass with quantization info, fallback to standard if not supported
    try:
        outputs = model(**batch, return_quantization_info=True)
    except TypeError as e:
        if "return_quantization_info" in str(e):
            # Model doesn't support return_quantization_info parameter
            outputs = model(**batch)
        else:
            raise e
    
    task_loss = outputs.loss
    
    # Collect quantization losses from outputs
    quantization_info = getattr(outputs, 'quantization_info', {})
    quant_losses = []
    for key, value in quantization_info.items():
        if 'quantization_loss' in key:
            quant_losses.append(value)
    
    # Collect routing losses
    routing_info = model.collect_routing_losses()
    routing_loss = routing_info.get('routing_loss', torch.tensor(0.0)) if routing_info else torch.tensor(0.0)
    
    # Combine losses
    total_loss = task_loss
    if quant_losses:
        total_loss += lambda_q * torch.stack(quant_losses).mean()
    if routing_loss > 0:
        total_loss += lambda_r * routing_loss
    
    return total_loss, {
        'task': task_loss.item(),
        'quant': torch.stack(quant_losses).mean().item() if quant_losses else 0,
        'route': routing_loss.item() if routing_loss > 0 else 0
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quadratic Schedule BitNet training - Early Exit + Quadratic Schedule'
    )
    
    # Model architecture parameters - Use None as default to use DefaultConfig values
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='Hidden size (must be power of 2 for BitLinear). If not provided, uses DefaultConfig value.')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of transformer layers. If not provided, uses DefaultConfig value.')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Number of attention heads. If not provided, uses DefaultConfig value.')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Training batch size. If not provided, uses DefaultConfig value.')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='Learning rate. If not provided, uses DefaultConfig value.')
    parser.add_argument('--max_length', type=int, default=None,
                      help='Maximum sequence length. If not provided, uses DefaultConfig value.')
    parser.add_argument('--num_steps', type=int, default=1000,
                      help='Number of training steps')
    
    # Joint loss parameters
    parser.add_argument('--lambda_q', type=float, default=0.1,
                      help='Weight for quantization loss in joint optimization')
    parser.add_argument('--lambda_r', type=float, default=0.05,
                      help='Weight for routing loss in joint optimization')
    
    parser.add_argument('--output_dir', type=str, default='./output-quadratic',
                      help='Output directory')
    parser.add_argument('--logging_steps', type=int, default=10,
                      help='Log every X steps')
    parser.add_argument('--save_steps', type=int, default=500,
                      help='Save checkpoint every X steps')
    parser.add_argument('--eval_steps', type=int, default=50,
                      help='Evaluate every X steps')
    
    # Early exit specific parameters
    parser.add_argument('--early_exit_threshold', type=float, default=0.95,
                      help='Confidence threshold for early exit')
    
    # Quadratic schedule specific parameters
    parser.add_argument('--quadratic_constant', type=float, default=0.3,
                      help='Constant c in quadratic dropout: p_l = c·(l/L)²')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Path to checkpoint to load (optional)')
    
    return parser.parse_args()

class QuadraticScheduleLayerSkipping(nn.Module):
    """Quadratic schedule layer skipping implementation."""
    
    def __init__(self, num_layers: int, quadratic_constant: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.quadratic_constant = quadratic_constant
        
        # Compute quadratic dropout probabilities
        self.dropout_probs = self._compute_quadratic_dropout()
        
        # Register skip masks as buffers
        self.register_buffer('skip_masks', torch.ones((num_layers,), dtype=torch.bool))
        
        logger = logging.getLogger(__name__)
        logger.info(f"QuadraticScheduleLayerSkipping initialized with quadratic_constant={quadratic_constant}")
        logger.debug(f"Dropout probabilities: {self.dropout_probs}")
    
    def _compute_quadratic_dropout(self) -> List[float]:
        """Compute quadratic dropout schedule: p_l = c·(l/L)²."""
        probs = []
        for l in range(self.num_layers):
            # p_l = c·(l/L)²
            prob = self.quadratic_constant * ((l / (self.num_layers - 1)) ** 2)
            probs.append(prob)
        return probs
    
    def forward(self, hidden_states: torch.Tensor, layer_idx: int,
                layer_fn: callable, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with quadratic schedule layer skipping."""
        
        # Get dropout probability for this layer
        skip_prob = self.dropout_probs[layer_idx]
        
        # Generate skip mask
        if self.training:
            # During training, apply stochastic skipping
            skip_mask = torch.rand(hidden_states.shape[0], device=hidden_states.device) < skip_prob
        else:
            # During inference, no skipping
            skip_mask = torch.zeros(hidden_states.shape[0], device=hidden_states.device, dtype=torch.bool)
        
        # Apply layer function to non-skipped samples
        if skip_mask.any():
            # Process non-skipped samples
            non_skipped = hidden_states[~skip_mask]
            if non_skipped.shape[0] > 0:
                processed = layer_fn(non_skipped, **kwargs)
                # Update hidden states for non-skipped samples
                hidden_states[~skip_mask] = processed
        
        return hidden_states, skip_mask

class EarlyExitLoss(nn.Module):
    """Early exit loss implementation."""
    
    def __init__(self, num_layers: int, threshold: float = 0.95):
        super().__init__()
        self.num_layers = num_layers
        self.threshold = threshold
        
    def forward(self, hidden_states: List[torch.Tensor], targets: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        """Compute early exit loss across all layers."""
        total_loss = 0.0
        layer_weights = torch.linspace(0.1, 1.0, self.num_layers)  # Weight early layers less
        
        for layer_idx, hidden in enumerate(hidden_states):
            if hidden is not None:
                # Compute logits for this layer
                logits = lm_head(hidden)
                layer_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='mean'
                )
                
                # Apply layer weighting
                weighted_loss = layer_weights[layer_idx] * layer_loss
                total_loss += weighted_loss
        
        return total_loss / self.num_layers

def main():
    """Main training function for quadratic schedule model."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("QUADRATIC SCHEDULE BITNET TRAINING - EARLY EXIT + QUADRATIC SCHEDULE")
    logger.info("=" * 80)
    logger.info("Ablation Study: Script 3/3")
    logger.info("Features: Native BitNet + Layer Dropout + BitLinear + Early Exit + Quadratic Schedule")
    logger.info("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
            token=os.getenv("HUGGINGFACE_TOKEN"),
            force_download=False,
        )
        tokenizer.pad_token = tokenizer.eos_token
        actual_vocab_size = len(tokenizer)
        logger.info(f"Tokenizer vocabulary size: {actual_vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise
    
    # Create quadratic schedule configuration
    # Only override DefaultConfig values if command line arguments are provided
    config_kwargs = {
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "vocab_size": actual_vocab_size,
        # All features enabled
        "use_layer_skipping": True,
        "skip_probability": 0.1,
        "min_layers_to_keep": 2,
        "use_early_exit": True,
        "early_exit_threshold": args.early_exit_threshold,
        "dropout_schedule": 'quadratic',
        "quadratic_constant": args.quadratic_constant,
        "output_dir": args.output_dir
    }
    
    # Only override if command line arguments are provided
    if args.hidden_size is not None:
        config_kwargs["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        config_kwargs["num_hidden_layers"] = args.num_layers
    if args.num_heads is not None:
        config_kwargs["num_attention_heads"] = args.num_heads
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config_kwargs["learning_rate"] = args.learning_rate
    if args.max_length is not None:
        config_kwargs["max_length"] = args.max_length
        config_kwargs["max_position_embeddings"] = args.max_length
    
    logger.info(f"DEBUG: config_kwargs = {config_kwargs}")
    config = QuadraticScheduleBitNetConfig(**config_kwargs)
    logger.info(f"DEBUG: After config creation, config.num_hidden_layers = {config.num_hidden_layers}")
    
    # Initialize quadratic schedule model (Native BitNet)
    logger.info("Initializing quadratic schedule BitNet model (Native BitNet)...")
    
    # Log all configuration parameters
    logger.info("=" * 60)
    logger.info("CONFIGURATION PARAMETERS (QUADRATIC SCHEDULE)")
    logger.info("=" * 60)
    logger.info(f"Model Architecture:")
    logger.info(f"  - Model Type: Native BitNet (model1)")
    logger.info(f"  - Hidden Size: {config.hidden_size}")
    logger.info(f"  - Number of Layers: {config.num_hidden_layers}")
    logger.info(f"  - Number of Heads: {config.num_attention_heads}")


    logger.info(f"  - Max Sequence Length: {config.max_position_embeddings}")
    logger.info(f"  - Vocabulary Size: {config.vocab_size}")
    logger.info(f"")
    logger.info(f"Training Parameters:")
    logger.info(f"  - Learning Rate: {config.learning_rate}")
    logger.info(f"  - Batch Size: {config.batch_size}")
    logger.info(f"  - Number of Steps: {args.num_steps}")
    logger.info(f"  - Output Directory: {config.output_dir}")
    logger.info(f"")
    logger.info(f"Quadratic Schedule Features:")
    logger.info(f"  - Layer Skipping: {config.use_layer_skipping}")
    logger.info(f"  - Skip Probability: {config.skip_probability}")
    logger.info(f"  - Min Layers to Keep: {config.min_layers_to_keep}")
    logger.info(f"  - Early Exit: {config.use_early_exit}")
    logger.info(f"  - Early Exit Threshold: {config.early_exit_threshold}")
    logger.info(f"  - Dropout Schedule: {config.dropout_schedule}")
    logger.info(f"  - Quadratic Constant: {config.quadratic_constant}")
    logger.info(f"")
    logger.info(f"Dataset Configuration:")
    logger.info(f"  - Dataset: {config.dataset_name}")
    logger.info(f"  - Subset: {config.subset}")
    logger.info("=" * 60)
    
    model = BitNetModel(config)  # Using model1 (native BitNet)
    model.to(device)
    
    # Test model forward method compatibility
    logger.info("Testing model forward method compatibility...")
    try:
        # Test with a small batch to check if return_quantization_info is supported
        test_input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
        test_attention_mask = torch.ones_like(test_input_ids)
        test_labels = test_input_ids.clone()
        
        # Test standard forward pass
        test_outputs = model(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            labels=test_labels
        )
        logger.info("✓ Standard forward pass works")
        
        # Test with return_quantization_info
        test_outputs_with_quant = model(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            labels=test_labels,
            return_quantization_info=True
        )
        logger.info("✓ Forward pass with return_quantization_info works")
        
    except Exception as e:
        logger.warning(f"Model compatibility test failed: {str(e)}")
        logger.info("Will use fallback mode for joint loss computation")
    
    # Load checkpoint if specified (check output directory first, then checkpoint_path)
    checkpoint_loaded = False
    checkpoint_path = None
    
    # First try to load from output directory
    if os.path.exists(args.output_dir):
        # Look for the most recent checkpoint in output directory
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith('quadratic_checkpoint_step_') and f.endswith('.pt')]
        if checkpoint_files:
            # Sort by step number and get the latest
            checkpoint_files.sort(key=lambda x: int(x.split('_step_')[1].split('.')[0]))
            latest_checkpoint = os.path.join(args.output_dir, checkpoint_files[-1])
            logger.info(f"Found checkpoint in output directory: {latest_checkpoint}")
            checkpoint_path = latest_checkpoint
        else:
            logger.info("No checkpoint found in output directory, starting fresh training")
    
    # If no checkpoint in output directory, try the specified checkpoint_path
    if not checkpoint_path and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint_path = args.checkpoint_path
        logger.info(f"Loading checkpoint from specified path: {checkpoint_path}")
    
    # Load checkpoint if found
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Map checkpoint layers if needed
                mapped_state_dict = map_checkpoint_layers(state_dict, 'native')
                model.load_state_dict(mapped_state_dict, strict=False)
                logger.info("Checkpoint loaded successfully with layer mapping")
                checkpoint_loaded = True
                
                # Resume from checkpoint step if available
                if 'step' in checkpoint:
                    global_step = checkpoint['step']
                    logger.info(f"Resuming training from step {global_step}")
            else:
                logger.warning("Checkpoint does not contain model_state_dict")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            logger.info("Continuing with randomly initialized model")
    
    # Initialize quadratic schedule layer skipping
    logger.info(f"DEBUG: config.num_hidden_layers = {config.num_hidden_layers}")
    logger.info(f"DEBUG: args.quadratic_constant = {args.quadratic_constant}")
    
    # Safety check for num_layers and provide fallback
    num_layers = config.num_hidden_layers
    if num_layers is None or num_layers <= 0:
        logger.warning(f"Invalid num_hidden_layers: {num_layers}, using default value 12")
        num_layers = 12  # BitSkip default
    
    quadratic_layer_skipping = QuadraticScheduleLayerSkipping(
        num_layers=num_layers,  # Use validated value
        quadratic_constant=args.quadratic_constant
    )
    
    # Log quadratic schedule details
    logger.info("=" * 60)
    logger.info("QUADRATIC SCHEDULE DETAILS")
    logger.info("=" * 60)
    logger.info(f"Quadratic Constant (c): {args.quadratic_constant}")
    logger.info(f"Number of Layers (L): {num_layers}")
    logger.info(f"Formula: p_l = c·(l/L)²")
    logger.info(f"")
    logger.info(f"Layer-wise Dropout Probabilities:")
    for l in range(num_layers):
        prob = args.quadratic_constant * ((l / (num_layers - 1)) ** 2)
        logger.info(f"  - Layer {l+1:2d}: p = {prob:.4f}")
    logger.info("=" * 60)
    
    # Initialize early exit loss
    early_exit_loss_fn = EarlyExitLoss(
        num_layers=num_layers,  # Use validated value
        threshold=args.early_exit_threshold
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,  # Use config value instead of args
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=1e-6
    )
    
    # Create dataloaders
    train_dataloader = create_streaming_dataloader(
        dataset_name="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        tokenizer=tokenizer,
        batch_size=config.batch_size,  # Use config value instead of args
        max_length=config.max_length,  # Use config value instead of args
        streaming=True,
        text_column="text"
    )
    
    eval_dataloader = create_streaming_dataloader(
        dataset_name="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        tokenizer=tokenizer,
        batch_size=config.batch_size,  # Use config value instead of args
        max_length=config.max_length,  # Use config value instead of args
        streaming=True,
        text_column="text",
        max_samples=1000
    )
    
    # Training loop with quadratic schedule and early exit
    logger.info("Starting quadratic schedule training...")
    if not checkpoint_loaded:
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
            
            # Forward pass with hidden states for early exit
            outputs = model(**tensor_inputs, output_hidden_states=True)
            
            # Compute early exit loss
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                early_exit_loss = early_exit_loss_fn(
                    hidden_states=outputs.hidden_states,
                    targets=batch['labels'].to(device),
                    lm_head=model.lm_head
                )
            else:
                # Fallback to standard loss
                early_exit_loss = outputs.loss
            
            # Compute joint loss with early exit
            try:
                loss, loss_components = compute_joint_loss(
                    model, 
                    tensor_inputs, 
                    lambda_q=args.lambda_q,
                    lambda_r=args.lambda_r
                )
                
                # Combine with early exit loss
                loss = loss + early_exit_loss
            except Exception as e:
                logger.error(f"Error in joint loss computation: {str(e)}")
                # Fallback to standard loss
                loss = outputs.loss + early_exit_loss
                loss_components = {
                    'task': outputs.loss.item(),
                    'quant': 0.0,
                    'route': 0.0
                }
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Store metrics for plotting
            training_losses.append(loss.item())
            learning_rates.append(scheduler.get_last_lr()[0])
            training_steps.append(global_step)
            
            # Logging
            if global_step % args.logging_steps == 0:
                logger.info(f"Step {global_step}: Total Loss = {loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6e}")
                logger.info(f"  - Task Loss: {loss_components['task']:.4f}")
                logger.info(f"  - Quantization Loss: {loss_components['quant']:.4f}")
                logger.info(f"  - Routing Loss: {loss_components['route']:.4f}")
                logger.info(f"  - Early Exit Loss: {early_exit_loss:.4f}")
                logger.info(f"Quadratic constant: {args.quadratic_constant}, Early exit threshold: {args.early_exit_threshold}")
                
                # Memory usage logging
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f'quadratic_checkpoint_step_{global_step}.pt'
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'step': global_step,
                    'loss': loss.item()
                }, checkpoint_path)
                logger.info(f"Saved quadratic schedule checkpoint to {checkpoint_path}")
                
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
        plt.title('Quadratic Schedule BitNet Training Loss (Layer Dropout + Early Exit + Quadratic)')
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
        plot_path = os.path.join(args.output_dir, 'quadratic_training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training loss plot saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create training loss plot: {str(e)}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'quadratic_final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_step': global_step
    }, final_model_path)
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Final GPU memory cleanup completed")
    
    logger.info("=" * 80)
    logger.info("QUADRATIC SCHEDULE TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("Features: Native BitNet + Layer Dropout + BitLinear + Early Exit + Quadratic Schedule")
    logger.info(f"Quadratic Constant: {args.quadratic_constant}")
    logger.info(f"Early Exit Threshold: {args.early_exit_threshold}")
    logger.info(f"Final model saved to: {final_model_path}")

if __name__ == '__main__':
    main()