
import torch
import torch.nn as nn
from bitnet.modeling.model import BitNetModel
from bitnet.modeling.model2 import BitNetModel2
from bitnet.utils.default_config import DefaultConfig

def create_model(exp_config):
    # Convert ExperimentConfig to DefaultConfig suited for BitNet models
    # BitNetModel is likely the standard one, BitNetModel2 has H-BitLinear
    
    model_config = DefaultConfig(
        vocab_size=exp_config.vocab_size,
        hidden_size=exp_config.hidden_size,
        num_hidden_layers=exp_config.num_hidden_layers,
        num_attention_heads=exp_config.num_attention_heads,
        # DefaultConfig uses mlp_ratio, not intermediate_size directly
        mlp_ratio=exp_config.intermediate_size / exp_config.hidden_size,
        max_position_embeddings=exp_config.max_position_embeddings,
        
        weight_bits=exp_config.weight_bits,
        activation_bits=exp_config.activation_bits,
        
        use_early_exit=exp_config.use_early_exit,
        dropout_schedule=exp_config.dropout_schedule,
        # We might need to handle p_max specifically if it's not in DefaultConfig standard fields yet
        # checking previous view, skip_probability was there, likely p_max maps to skip_probability or similar
        skip_probability=exp_config.dropout_probability_max,
        use_layer_skipping=exp_config.dropout_probability_max > 0
    )
    
    # Select model class based on Hadamard usage
    # Assuming BitNetModel2 uses H-BitLinear (Hadamard) and BitNetModel uses standard BitLinear
    if exp_config.use_hadamard:
        print("Using BitNetModel2 (Hadamard)")
        model = BitNetModel2(model_config)
    else:
        print("Using BitNetModel (Standard)")
        model = BitNetModel(model_config)
        
    return model
