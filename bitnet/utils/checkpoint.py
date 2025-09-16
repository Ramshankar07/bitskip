"""
Checkpoint utilities for saving and loading model states.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    output_dir: Union[str, Path],
    **kwargs
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Optional learning rate scheduler
        epoch: Current epoch number
        output_dir: Directory to save checkpoint
        **kwargs: Additional items to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    checkpoint_path = output_dir / f"checkpoint-{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    checkpoint_path: Union[str, Path] = None,
    device: str = "cpu"
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
    
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint 