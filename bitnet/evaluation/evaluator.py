"""
Evaluation utilities for BitNet model.
"""

from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def calculate_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Calculate accuracy between predictions and labels.
    
    Args:
        predictions: List of predicted token IDs
        labels: List of ground truth token IDs
        
    Returns:
        Accuracy score
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Mask out padding tokens (usually -100)
    mask = labels != -100
    return np.mean(predictions[mask] == labels[mask])


def evaluate(
    model: torch.nn.Module,
    eval_dataloader,
    config: Dict
) -> Dict[str, float]:
    """
    Evaluate model on evaluation dataset.
    
    Args:
        model: BitNet model
        eval_dataloader: Evaluation data loader
        config: Evaluation configuration
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0  # Count batches processed
    
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
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            batch_count += 1  # Increment batch count
    
    if batch_count == 0:
        raise ValueError("No batches processed in evaluation.")
    
    accuracy = calculate_accuracy(all_preds, all_labels)
    perplexity = torch.exp(torch.tensor(total_loss / batch_count))
    
    return {
        "loss": total_loss / batch_count,
        "perplexity": perplexity.item(),
        "accuracy": accuracy
    } 