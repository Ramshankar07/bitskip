#!/usr/bin/env python3
"""
Load and count parameters using the actual training scripts to avoid custom class errors.
"""

import os
import sys
import torch
import argparse

# Add current directory to path
sys.path.insert(0, os.getcwd())

def count_parameters_in_state_dict(state_dict):
    """Count parameters in a state dict."""
    total_params = 0
    param_details = []
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            param_count = param.numel()
            total_params += param_count
            
            param_details.append({
                'name': name,
                'shape': list(param.shape),
                'params': param_count,
                'dtype': str(param.dtype)
            })
    
    return total_params, param_details

def load_quadratic_model():
    """Load the quadratic model using the training script approach."""
    print("Loading Quadratic BitNet model...")
    
    try:
        # Import the training script modules
        from train_quadratic import QuadraticScheduleBitNetConfig, BitNetModel
        from bitnet.utils.default_config import DefaultConfig
        
        # Create a minimal config (we just need the model structure)
        config = QuadraticScheduleBitNetConfig(
            vocab_size=128256,  # Use a reasonable default
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            max_position_embeddings=2048
        )
        
        # Create model
        model = BitNetModel(config)
        
        # Load checkpoint
        checkpoint = torch.load("quadratic_final_model.pt", map_location="cpu", weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict into model
        model.load_state_dict(state_dict, strict=False)
        
        # Get model state dict
        model_state_dict = model.state_dict()
        
        print(f"✓ Successfully loaded Quadratic BitNet model")
        return model_state_dict, config
        
    except Exception as e:
        print(f"✗ Error loading Quadratic BitNet model: {e}")
        return None, None

def load_quadratic_hbitlinear_model():
    """Load the quadratic H-BitLinear model using the training script approach."""
    print("Loading Quadratic H-BitLinear model...")
    
    try:
        # Import the training script modules
        from train_quadratic_hbitlinear import QuadraticScheduleHBitLinearConfig, BitNetModel2
        from bitnet.utils.default_config import DefaultConfig
        
        # Create a minimal config
        config = QuadraticScheduleHBitLinearConfig(
            vocab_size=128256,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            max_position_embeddings=2048
        )
        
        # Create model
        model = BitNetModel2(config)
        
        # Load checkpoint
        checkpoint = torch.load("quadratic_hbitlinear_final_model.pt", map_location="cpu", weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict into model
        model.load_state_dict(state_dict, strict=False)
        
        # Get model state dict
        model_state_dict = model.state_dict()
        
        print(f"✓ Successfully loaded Quadratic H-BitLinear model")
        return model_state_dict, config
        
    except Exception as e:
        print(f"✗ Error loading Quadratic H-BitLinear model: {e}")
        return None, None

def analyze_model(model_name, state_dict, config):
    """Analyze a model's parameters."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_name.upper()}")
    print(f"{'='*60}")
    
    if state_dict is None:
        print("No model to analyze")
        return None
    
    # Count parameters
    total_params, param_details = count_parameters_in_state_dict(state_dict)
    
    print(f"\n📊 PARAMETER SUMMARY:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Total Size: {total_params * 4 / (1024**3):.2f} GB (assuming float32)")
    
    if config:
        print(f"\n🏗️  MODEL ARCHITECTURE:")
        print(f"  Hidden Size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  Number of Layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"  Number of Heads: {getattr(config, 'num_attention_heads', 'N/A')}")
        print(f"  Vocabulary Size: {getattr(config, 'vocab_size', 'N/A')}")
    
    # Show parameter breakdown by layer type
    print(f"\n📋 PARAMETER BREAKDOWN BY LAYER TYPE:")
    
    layer_stats = {}
    for param in param_details:
        name = param['name']
        params = param['params']
        
        # Categorize by layer type
        if 'embed_tokens' in name:
            layer_type = 'Token Embeddings'
        elif 'embed_positions' in name:
            layer_type = 'Position Embeddings'
        elif 'self_attn' in name:
            layer_type = 'Self Attention'
        elif 'feed_forward' in name:
            layer_type = 'Feed Forward'
        elif 'layer_norm' in name or 'norm' in name:
            layer_type = 'Layer Normalization'
        elif 'lm_head' in name:
            layer_type = 'Language Model Head'
        else:
            layer_type = 'Other'
        
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {'count': 0, 'params': 0}
        layer_stats[layer_type]['count'] += 1
        layer_stats[layer_type]['params'] += params
    
    for layer_type, stats in sorted(layer_stats.items(), key=lambda x: x[1]['params'], reverse=True):
        percentage = (stats['params'] / total_params) * 100
        print(f"  {layer_type:20s}: {stats['params']:8,} params ({percentage:5.1f}%) - {stats['count']} layers")
    
    # Show largest parameters
    print(f"\n🔍 LARGEST PARAMETERS:")
    largest_params = sorted(param_details, key=lambda x: x['params'], reverse=True)[:10]
    for i, param in enumerate(largest_params, 1):
        print(f"  {i:2d}. {param['name']:40s}: {param['params']:8,} params {param['shape']}")
    
    return {
        'total_params': total_params,
        'param_details': param_details,
        'layer_stats': layer_stats
    }

def main():
    print("Loading models using training script approach...")
    
    # Load both models
    quadratic_state_dict, quadratic_config = load_quadratic_model()
    hbitlinear_state_dict, hbitlinear_config = load_quadratic_hbitlinear_model()
    
    # Analyze each model
    quadratic_stats = analyze_model("Quadratic BitNet", quadratic_state_dict, quadratic_config)
    hbitlinear_stats = analyze_model("Quadratic H-BitLinear", hbitlinear_state_dict, hbitlinear_config)
    
    # Compare if both loaded successfully
    if quadratic_stats and hbitlinear_stats:
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON")
        print(f"{'='*80}")
        
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"  {'Quadratic BitNet':20s}: {quadratic_stats['total_params']:8,} parameters")
        print(f"  {'Quadratic H-BitLinear':20s}: {hbitlinear_stats['total_params']:8,} parameters")
        
        diff = hbitlinear_stats['total_params'] - quadratic_stats['total_params']
        diff_pct = (diff / quadratic_stats['total_params']) * 100 if quadratic_stats['total_params'] > 0 else 0
        
        print(f"  Difference: {diff:+,} parameters ({diff_pct:+.1f}%)")
        
        if diff > 0:
            print(f"  H-BitLinear model has {diff:,} more parameters than BitNet model")
        elif diff < 0:
            print(f"  BitNet model has {abs(diff):,} more parameters than H-BitLinear model")
        else:
            print(f"  Both models have the same number of parameters")

if __name__ == "__main__":
    main()
