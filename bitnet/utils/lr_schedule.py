import torch
import math
from typing import Optional, Dict, List
from collections import deque
import numpy as np


class WSDSchedulerWithSMA:
    """
    Warmup-Stable-Decay (WSD) Learning Rate Scheduler with Simple Moving Average (SMA)
    
    This scheduler is specifically designed for BitNet + LayerSkip training which requires:
    - Higher learning rates due to quantization and layer dropout
    - Stable training phase to handle gradient noise from STE and stochastic layers
    - SMA to smooth out fluctuations from quantization noise
    
    Three phases:
    1. Warmup: Linear increase from 0 to peak_lr
    2. Stable: Maintain peak_lr (critical for BitNet convergence)
    3. Decay: Cosine decay to min_lr
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 0.0,
        sma_window: int = 100,
        gradient_accumulation_steps: int = 1,
        bitnet_scale_factor: float = 5.0,  # BitNet requires 5-10x higher LR
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            peak_lr: Maximum learning rate (will be scaled by bitnet_scale_factor)
            warmup_steps: Number of steps for warmup phase
            stable_steps: Number of steps to maintain peak LR
            decay_steps: Number of steps for decay phase
            min_lr: Minimum learning rate after decay
            sma_window: Window size for moving average of LR adjustments
            gradient_accumulation_steps: For proper step counting
            bitnet_scale_factor: Multiplier for BitNet's higher LR requirement
        """
        self.optimizer = optimizer
        self.base_peak_lr = peak_lr
        self.peak_lr = peak_lr * bitnet_scale_factor
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr * bitnet_scale_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # SMA components
        self.sma_window = sma_window
        self.lr_history = deque(maxlen=sma_window)
        self.gradient_norm_history = deque(maxlen=sma_window)
        
        # State tracking
        self.current_step = 0
        self.total_steps = warmup_steps + stable_steps + decay_steps
        
        # For adaptive adjustments based on training dynamics
        self.loss_history = deque(maxlen=sma_window)
        self.quantization_error_history = deque(maxlen=sma_window)
        
        # Initialize optimizer LR
        self._update_lr(0.0)
    
    def get_phase(self) -> str:
        """Determine current training phase"""
        if self.current_step < self.warmup_steps:
            return "warmup"
        elif self.current_step < self.warmup_steps + self.stable_steps:
            return "stable"
        else:
            return "decay"
    
    def compute_lr(self) -> float:
        """Compute learning rate based on current phase"""
        phase = self.get_phase()
        
        if phase == "warmup":
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            lr = self.peak_lr * progress
            
        elif phase == "stable":
            # Maintain peak LR with small oscillations for exploration
            # This helps BitNet escape local minima caused by quantization
            progress = (self.current_step - self.warmup_steps) / self.stable_steps
            oscillation = 0.05 * math.sin(2 * math.pi * progress * 4)  # 4 cycles
            lr = self.peak_lr * (1.0 + oscillation)
            
        else:  # decay phase
            # Cosine decay
            progress = (self.current_step - self.warmup_steps - self.stable_steps) / self.decay_steps
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        return lr
    
    def apply_sma(self, lr: float) -> float:
        """Apply Simple Moving Average to smooth learning rate transitions"""
        self.lr_history.append(lr)
        
        if len(self.lr_history) < 10:  # Don't apply SMA in very early training
            return lr
        
        # Weighted SMA - recent values have more weight
        weights = np.linspace(0.5, 1.0, len(self.lr_history))
        weights = weights / weights.sum()
        smoothed_lr = np.average(list(self.lr_history), weights=weights)
        
        return smoothed_lr
    
    def adaptive_adjustment(self, metrics: Optional[Dict] = None) -> float:
        """
        Adaptive LR adjustment based on training metrics
        Particularly important for BitNet + LayerSkip
        """
        adjustment_factor = 1.0
        
        if metrics:
            # Check gradient norm - if too high, reduce LR
            if 'gradient_norm' in metrics:
                grad_norm = metrics['gradient_norm']
                self.gradient_norm_history.append(grad_norm)
                
                if len(self.gradient_norm_history) > 50:
                    avg_norm = np.mean(list(self.gradient_norm_history))
                    if grad_norm > 2 * avg_norm:
                        adjustment_factor *= 0.9  # Reduce LR
                    elif grad_norm < 0.5 * avg_norm:
                        adjustment_factor *= 1.1  # Increase LR
            
            # Check loss plateau
            if 'loss' in metrics:
                self.loss_history.append(metrics['loss'])
                
                if len(self.loss_history) == self.sma_window:
                    recent_loss = np.mean(list(self.loss_history)[-20:])
                    older_loss = np.mean(list(self.loss_history)[:20])
                    
                    # If loss not improving, adjust LR
                    if recent_loss > 0.99 * older_loss:
                        adjustment_factor *= 0.95
            
            # BitNet-specific: monitor quantization error
            if 'quantization_error' in metrics:
                quant_error = metrics['quantization_error']
                self.quantization_error_history.append(quant_error)
                
                if len(self.quantization_error_history) > 20:
                    avg_error = np.mean(list(self.quantization_error_history))
                    # High quantization error suggests need for higher LR
                    if quant_error > 1.5 * avg_error:
                        adjustment_factor *= 1.05
        
        return max(0.8, min(1.2, adjustment_factor))  # Limit adjustment range
    
    def _update_lr(self, lr: float):
        """Update optimizer's learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self, metrics: Optional[Dict] = None):
        """
        Update learning rate for current step
        
        Args:
            metrics: Optional dict with training metrics for adaptive adjustment
                    Keys can include: 'loss', 'gradient_norm', 'quantization_error'
        """
        # Only count actual optimizer steps (not gradient accumulation steps)
        if self.current_step % self.gradient_accumulation_steps == 0:
            # Compute base LR for current phase
            base_lr = self.compute_lr()
            
            # Apply SMA smoothing
            smoothed_lr = self.apply_sma(base_lr)
            
            # Apply adaptive adjustments
            adjustment = self.adaptive_adjustment(metrics)
            final_lr = smoothed_lr * adjustment
            
            # Update optimizer
            self._update_lr(final_lr)
        
        self.current_step += 1
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates (compatible with PyTorch schedulers)"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict:
        """Save scheduler state"""
        return {
            'current_step': self.current_step,
            'lr_history': list(self.lr_history),
            'gradient_norm_history': list(self.gradient_norm_history),
            'loss_history': list(self.loss_history),
            'quantization_error_history': list(self.quantization_error_history),
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state"""
        self.current_step = state_dict['current_step']
        self.lr_history = deque(state_dict['lr_history'], maxlen=self.sma_window)
        self.gradient_norm_history = deque(state_dict['gradient_norm_history'], maxlen=self.sma_window)
        self.loss_history = deque(state_dict['loss_history'], maxlen=self.sma_window)
        self.quantization_error_history = deque(state_dict['quantization_error_history'], maxlen=self.sma_window)


# Example usage for your BitNet + LayerSkip setup
def create_scheduler_for_bitnet_layerskip(
    optimizer: torch.optim.Optimizer,
    total_training_steps: int,
    base_learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    stable_ratio: float = 0.4,
    decay_ratio: float = 0.5,
):
    """
    Create WSD scheduler with recommended settings for BitNet + LayerSkip
    
    Args:
        optimizer: Your optimizer (AdamW recommended)
        total_training_steps: Total number of training steps
        base_learning_rate: Base LR (will be scaled up for BitNet)
        warmup_ratio: Fraction of steps for warmup
        stable_ratio: Fraction of steps for stable phase
        decay_ratio: Fraction of steps for decay
    """
    warmup_steps = int(total_training_steps * warmup_ratio)
    stable_steps = int(total_training_steps * stable_ratio)
    decay_steps = int(total_training_steps * decay_ratio)
    
    scheduler = WSDSchedulerWithSMA(
        optimizer=optimizer,
        peak_lr=base_learning_rate,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        min_lr=base_learning_rate * 0.1,  # 10% of peak
        sma_window=100,
        bitnet_scale_factor=8.0,  # 8x for BitNet as per paper
    )
    
    return scheduler


# Training loop integration example
# def train_step_with_scheduler(
#     model,
#     optimizer,
#     scheduler,
#     data_batch,
#     compute_metrics_fn,
# ):
#     """Example of how to integrate the scheduler in your training loop"""
    
#     # Forward pass
#     outputs = model(data_batch)
#     loss = outputs.loss
    
#     # Backward pass
#     loss.backward()
    
#     # Compute metrics for adaptive LR
#     with torch.no_grad():
#         # Gradient norm
#         total_norm = 0.0
#         for p in model.parameters():
#             if p.grad is not None:
#                 param_norm = p.grad.data.norm(2)
#                 total_norm += param_norm.item() ** 2
#         gradient_norm = total_norm ** 0.5
        
#         # Quantization error (if available from your BitNet implementation)
#         quantization_error = compute_metrics_fn(model) if compute_metrics_fn else None
    
#     # Update optimizer
#     optimizer.step()
#     optimizer.zero_grad()
    
#     # Update scheduler with metrics
#     metrics = {
#         'loss': loss.item(),
#         'gradient_norm': gradient_norm,
#     }
#     if quantization_error is not None:
#         metrics['quantization_error'] = quantization_error
    
#     scheduler.step(metrics)
    
#     return loss.item(), scheduler.get_last_lr()[0]
