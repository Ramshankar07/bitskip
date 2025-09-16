"""
Training pipeline for BitNet with layer skipping and mixed precision support.
"""

from typing import Dict, Optional, Any
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class MemoryEfficientTrainer:
    """
    Memory efficient trainer for BitNet with support for:
    - Mixed precision training
    - Gradient checkpointing
    - Activation offloading
    - Optimizer state offloading
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        config: Dict[str, Any]
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize mixed precision training
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')  # Updated to new format
        
        # Enable gradient checkpointing if specified
        if config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
        
        # Move model to device
        self.model.to(self.device)
        
        # Ensure model parameters require gradients
        self.model.train()
        trainable_params = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                logger.warning(f"Parameter {name} does not require gradients!")
            else:
                trainable_params += param.numel()
        
        logger.info(f"Trainer initialized with {trainable_params:,} trainable parameters")
        
        # Move optimizer states to CPU if specified
        if config.get('optimizer_offloading', False):
            for param in self.model.parameters():
                if param.requires_grad:
                    param.register_hook(lambda grad: grad.to('cpu'))
    
    def train_step(self, batch, model_config, checkpoint_config=None, gradient_accumulation_steps=1):
        """Perform a single training step."""
        # Ensure model is in training mode
        self.model.train()
        
        # Move tensors to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with model configuration
        with torch.amp.autocast('cuda') if self.use_amp else torch.enable_grad():
            outputs = self.model(
                **batch,
                **model_config
            )
        
        # Check if loss is computed
        if not hasattr(outputs, 'loss') or outputs.loss is None:
            raise ValueError("Model did not return a loss. Ensure labels are provided.")
        
        loss = outputs.loss / gradient_accumulation_steps
        
        # Check if loss requires gradient
        if not loss.requires_grad:
            logger.error("Loss does not require gradient!")
            logger.error(f"Loss value: {loss.item()}")
            logger.error(f"Loss requires_grad: {loss.requires_grad}")
            logger.error(f"Model training mode: {self.model.training}")
            
            # Check individual parameters
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    logger.error(f"Parameter {name} does not require grad")
            
            raise RuntimeError("Loss does not require gradient. Check model initialization.")
        
        # Backward pass with gradient scaling
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if hasattr(self, 'current_step') and self.current_step % gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.get('max_grad_norm', 0) > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
            
            # Update weights
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        if not hasattr(self, 'current_step'):
            self.current_step = 0
        self.current_step += 1
        
        return {
            'loss': loss.item() * gradient_accumulation_steps,
            'step': self.current_step
        }
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.use_amp and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Dict
) -> float:
    """
    Training loop for BitNet model.
    
    Args:
        model: BitNet model
        train_dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    # Set up layer skipping strategy
    if config.get('dynamic_layer_skipping', False) and config.get('layer_skip_warmup', 0) > 0:
        # Start with no skipping, gradually introduce it
        model.layer_skipping.strategy = 'none'
    
    # Training loop
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        # Process batch
        input_ids = batch["input_ids"].to(config['device'])
        attention_mask = batch["attention_mask"].to(config['device'])
        labels = batch["labels"].to(config['device'])
        
        # Forward pass with gradient checkpointing
        with torch.amp.autocast('cuda') if config.get('use_amp', False) else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Scale loss for mixed precision training
        if config.get('use_amp', False):
            config['scaler'].scale(loss).backward()
            if config.get('max_grad_norm', 0) > 0:
                config['scaler'].unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            config['scaler'].step(optimizer)
            config['scaler'].update()
        else:
            # Standard training
            loss.backward()
            if config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        # Enable dynamic layer skipping after warmup
        if (config.get('dynamic_layer_skipping', False) and 
            step == config.get('layer_skip_warmup', 0)):
            model.layer_skipping.strategy = 'dynamic'
            print("Enabled dynamic layer skipping")
        
        # Logging
        if step % config.get('logging_steps', 100) == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_dataloader)


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    config: Dict
) -> float:
    """
    Evaluation loop for BitNet model.
    
    Args:
        model: BitNet model
        eval_dataloader: Evaluation data loader
        config: Evaluation configuration
        
    Returns:
        Average evaluation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Process batch
            input_ids = batch["input_ids"].to(config['device'])
            attention_mask = batch["attention_mask"].to(config['device'])
            labels = batch["labels"].to(config['device'])
            
            # Forward pass
            with torch.amp.autocast('cuda') if config.get('use_amp', False) else torch.enable_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(eval_dataloader)