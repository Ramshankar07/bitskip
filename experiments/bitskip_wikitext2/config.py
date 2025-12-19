
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    # Model Architecture
    vocab_size: int = 50257  # GPT-2 tokenizer
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    
    # Quantization
    weight_bits: int = 8  # 16 (FP16), 8, 4
    activation_bits: int = 8
    use_hadamard: bool = False
    
    # Early Exit / Layer Skipping
    use_early_exit: bool = False
    early_exit_loss_weight: float = 0.0  # lambda
    dropout_probability_max: float = 0.0 # p_max
    dropout_schedule: str = "quadratic" # quadratic, linear, uniform
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 6e-4
    num_steps: int = 50000
    warmup_steps: int = 1000
    seed: int = 42
    
    def __post_init__(self):
        # Ensure hidden_size is divisible by heads
        assert self.hidden_size % self.num_attention_heads == 0
