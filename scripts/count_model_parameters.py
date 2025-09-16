#!/usr/bin/env python3
"""
Count parameters in model checkpoints using the training code to avoid custom class errors.

This script loads models using the same approach as the training scripts to avoid
custom class loading issues.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add current directory to path so we can import training modules
sys.path.insert(0, os.getcwd())

def count_parameters(state_dict):
    """Count total parameters and trainable parameters in a state dict."""
    total_params = 0
    trainable_params = 0
    
    param_details = []
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            param_count = param.numel()
            total_params += param_count
            
            # Assume all parameters are trainable (no frozen layers in these models)
            trainable_params += param_count
            
            param_details.append({
                'name': name,
                'shape': list(param.shape),
                'params': param_count,
                'dtype': str(param.dtype)
            })
    
    return total_params, trainable_params, param_details

def load_model_from_training_script(checkpoint_path, model_type="quadratic"):
    """Load model using the same approach as training scripts."""
    
    print(f"Loading {model_type} model from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Found model_state_dict with {len(state_dict)} parameters")
        else:
            print("No model_state_dict found, using entire checkpoint")
            state_dict = checkpoint
        
        # Get config info if available
        config = checkpoint.get('config', None)
        if config:
            print(f"Config type: {type(config)}")
            if hasattr(config, 'num_hidden_layers'):
                print(f"Number of layers: {config.num_hidden_layers}")
            if hasattr(config, 'hidden_size'):
                print(f"Hidden size: {config.hidden_size}")
        
        return state_dict, config
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None

def analyze_model_parameters(checkpoint_path, model_name):
    """Analyze and display model parameters."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_name.upper()}")
    print(f"{'='*60}")
    
    state_dict, config = load_model_from_training_script(checkpoint_path, model_name)
    
    if state_dict is None:
        print(f"Failed to load {model_name} model")
        return
    
    # Count parameters
    total_params, trainable_params, param_details = count_parameters(state_dict)
    
    print(f"\n📊 PARAMETER SUMMARY:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"  Total Size: {total_params * 4 / (1024**3):.2f} GB (assuming float32)")
    
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
        'trainable_params': trainable_params,
        'param_details': param_details,
        'layer_stats': layer_stats
    }

def compare_models(model1_path, model1_name, model2_path, model2_name):
    """Compare two models."""
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Analyze both models
    model1_stats = analyze_model_parameters(model1_path, model1_name)
    model2_stats = analyze_model_parameters(model2_path, model2_name)
    
    if model1_stats is None or model2_stats is None:
        print("Cannot compare models - one or both failed to load")
        return
    
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"  {model1_name:20s}: {model1_stats['total_params']:8,} parameters")
    print(f"  {model2_name:20s}: {model2_stats['total_params']:8,} parameters")
    
    diff = model2_stats['total_params'] - model1_stats['total_params']
    diff_pct = (diff / model1_stats['total_params']) * 100 if model1_stats['total_params'] > 0 else 0
    
    print(f"  Difference: {diff:+,} parameters ({diff_pct:+.1f}%)")
    
    if diff > 0:
        print(f"  {model2_name} has {diff:,} more parameters than {model1_name}")
    elif diff < 0:
        print(f"  {model1_name} has {abs(diff):,} more parameters than {model2_name}")
    else:
        print(f"  Both models have the same number of parameters")

def main():
    parser = argparse.ArgumentParser(description="Count model parameters using training code")
    parser.add_argument("--model1", default="quadratic_final_model.pt", help="First model path")
    parser.add_argument("--model2", default="quadratic_hbitlinear_final_model.pt", help="Second model path")
    parser.add_argument("--compare", action="store_true", help="Compare the two models")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model1):
        print(f"ERROR: Model file not found: {args.model1}")
        return
    
    if not os.path.exists(args.model2):
        print(f"ERROR: Model file not found: {args.model2}")
        return
    
    if args.compare:
        # Compare both models
        compare_models(
            args.model1, "Quadratic BitNet",
            args.model2, "Quadratic H-BitLinear"
        )
    else:
        # Analyze each model separately
        analyze_model_parameters(args.model1, "Quadratic BitNet")
        analyze_model_parameters(args.model2, "Quadratic H-BitLinear")

if __name__ == "__main__":
    main()
