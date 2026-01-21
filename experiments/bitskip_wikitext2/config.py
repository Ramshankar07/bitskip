
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExperimentConfig:
    # Model Size (preset)
    model_size: str = "125M"  # "125M" or "200M"
    
    # Dataset
    dataset: str = "mix"  # wikitext2, wikitext103, ptb, mix
    
    # Model Architecture (can be overridden or set by model_size)
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
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-4
    num_steps: int = 100
    warmup_steps: int = 10
    seed: int = 42
    
    # Early Stopping
    eval_every_steps: int = 25  # Evaluate validation every N steps
    patience: int = 3  # Stop after N evaluations without improvement
    min_delta: float = 0.01  # Minimum improvement threshold for validation perplexity
    
    def __post_init__(self):
        # Apply model size preset if specified
        self.apply_model_size(self.model_size)
        # Ensure hidden_size is divisible by heads
        assert self.hidden_size % self.num_attention_heads == 0
    
    def apply_model_size(self, size: str):
        """Apply a model size preset (125M or 200M)."""
        from model_configs import MODEL_CONFIGS
        if size in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[size]
            self.hidden_size = cfg["hidden_size"]
            self.num_hidden_layers = cfg["num_hidden_layers"]
            self.num_attention_heads = cfg["num_attention_heads"]
            self.intermediate_size = cfg["intermediate_size"]
            self.model_size = size
