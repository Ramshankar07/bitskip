"""
Default configuration settings for BitNet model.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import torch

@dataclass
class DefaultConfig:
    """Default configuration for BitNet model with streaming datasets."""
    
    # Dataset configuration
    dataset_name: str = "allenai/dolmino-mix-1124"  
    subset: str = "default"  
    text_column: str = "text"  
    split: str = "train"  
    
    
    max_length: int = 1024  # Reduced sequence length for memory efficiency (was 2048) (matches max_seq_len)
    max_train_samples: Optional[int] = 1000  # Max training samples (for testing)
    max_eval_samples: Optional[int] = 100  # Max evaluation samples (for testing)
    
    # Model architecture (BitSkip specifications)
    # All dimensions are powers of 2 to ensure compatibility with H-BitLinear layers
    # Target: ~2.1B parameters with 12 layers, 1024 hidden dim, 16×64 attention heads
    vocab_size: int = 128256
    hidden_size: int = 1024  # dim (2^10 - power of 2 for H-BitLinear)
    num_hidden_layers: int = 12  # num_layers (BitSkip specification)
    num_attention_heads: int = 16  # num_heads (BitSkip specification)
    num_kv_heads: int = 4  # num_kv_heads (2^2 - must divide hidden_size)
    head_dim: int = 64  # head_dim (1024/16 = 64, BitSkip specification)
    mlp_ratio: float = 2.0  # 4096/1024 = 4.0 (BitSkip FFN intermediate specification)
    max_position_embeddings: int = 1024  # max_seq_len (matches max_length for memory efficiency)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    
    # BitNet specific (BitSkip specifications)
    activation_bits: int = 8  # 8-bit activation quantization (BitSkip specification)
    weight_bits: int = 2  # Ternary quantization (BitSkip specification)
    
    # Joint loss parameters for quantization and routing optimization
    lambda_q: float = 0.1  # Weight for quantization loss in joint optimization
    lambda_r: float = 0.05  # Weight for routing loss in joint optimization
    
    # Layer skipping (BitSkip specifications)
    use_layer_skipping: bool = True
    skip_probability: float = 0.1
    min_layers_to_keep: int = 4  # Appropriate for 12-layer model
    use_early_exit: bool = True
    early_exit_threshold: float = 0.95
    
    # Training
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8  # Increased for memory efficiency
    batch_size: int = 4  # Reduced for memory efficiency (was 16)
    eval_batch_size: int = 2  # Reduced for memory efficiency (was 8)
    num_epochs: int = 1
    
    # Memory optimization
    use_amp: bool = True  # Enable automatic mixed precision for memory efficiency
    gradient_checkpointing: bool = True  # Enable gradient checkpointing for memory efficiency
    activation_offloading: bool = False
    optimizer_offloading: bool = False
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging and output
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    output_dir: str = "outputs"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DefaultConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Get config value with default fallback (dict-like interface)."""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str):
        """Enable dictionary-style access to config attributes."""
        return getattr(self, key)
    
    def validate(self):
        """Validate config after initialization."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.hidden_size % self.num_kv_heads == 0, \
            "hidden_size must be divisible by num_kv_heads"
        assert self.hidden_size % self.head_dim == 0, \
            "hidden_size must be divisible by head_dim"
        assert self.mlp_ratio > 0, \
            "mlp_ratio must be greater than 0"
        assert 0 <= self.skip_probability <= 1, \
            "skip_probability must be between 0 and 1"
        assert self.min_layers_to_keep > 0, \
            "min_layers_to_keep must be greater than 0"
        assert self.min_layers_to_keep <= self.num_hidden_layers, \
            "min_layers_to_keep must be less than or equal to num_hidden_layers"
        assert 0 <= self.early_exit_threshold <= 1, \
            "early_exit_threshold must be between 0 and 1"
        assert self.activation_bits > 0, \
            "activation_bits must be greater than 0"
        assert self.max_length > 0, \
            "max_length must be greater than 0"
        assert self.batch_size > 0, \
            "batch_size must be greater than 0"
        assert self.eval_batch_size > 0, \
            "eval_batch_size must be greater than 0"
        assert self.num_epochs > 0, \
            "num_epochs must be greater than 0" 