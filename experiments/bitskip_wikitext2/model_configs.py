"""
Model configuration presets for BitSkip experiments.
Provides 125M parameter configuration.
"""

# Model size configurations matching the revision plan
MODEL_CONFIGS = {
    "125M": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "description": "~125M parameters, trainable in ~8 hours on single GPU"
    },
    "85M_H": {
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "description": "~85M params, all dims power-of-2 for H-BitLinear (no FWHT padding)"
    },
}

# Paper terminology mapping for clarity
# W = weight bits, A = activation bits, H = Hadamard transform
PAPER_NAMES = {
    # (weight_bits, activation_bits, use_hadamard) -> paper name
    (16, 16, False): "FP16-Baseline",
    (8, 8, False): "BitSkip-W1.58A8",      # Ternary weights + 8-bit activations
    (4, 4, True): "BitSkip-W1.58A4-H",     # Ternary weights + 4-bit acts + Hadamard
    (8, 8, True): "BitSkip-W1.58A8-H",     # Ternary weights + 8-bit acts + Hadamard
}

def get_paper_name(weight_bits: int, activation_bits: int, use_hadamard: bool) -> str:
    """Get the paper terminology for a given configuration."""
    key = (weight_bits, activation_bits, use_hadamard)
    return PAPER_NAMES.get(key, f"Custom-W{weight_bits}A{activation_bits}{'H' if use_hadamard else ''}")

def get_model_config(size: str = "125M") -> dict:
    """Get model configuration for a given size."""
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[size].copy()

# Ablation study configurations
LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7]
P_MAX_VALUES = [0.0, 0.2, 0.5, 0.7]
SCHEDULE_OPTIONS = ["quadratic", "linear", "uniform"]
HADAMARD_PLACEMENTS = ["none", "pre_attention", "pre_ffn", "full"]

# Single seed for experiments
SEEDS = [42]

# Dataset options
DATASETS = ["wikitext2", "wikitext103", "ptb", "mix"]
