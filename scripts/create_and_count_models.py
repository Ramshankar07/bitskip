#!/usr/bin/env python3
"""
Create fresh models using training scripts and count their parameters.
This avoids the custom class loading issues by creating new models.
"""

import os
import sys
import torch
import argparse

# Add current directory to path
sys.path.insert(0, os.getcwd())

def count_parameters_in_model(model):
    """Count parameters in a model."""
    total_params = 0
    trainable_params = 0
    param_details = []
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        param_details.append({
            'name': name,
            'shape': list(param.shape),
            'params': param_count,
            'dtype': str(param.dtype),
            'requires_grad': param.requires_grad
        })
    
    return total_params, trainable_params, param_details

def create_quadratic_model():
    """Create a fresh quadratic model using the training script."""
    print("Creating fresh Quadratic BitNet model...")
    
    try:
        # Import the training script modules
        from train_quadratic import QuadraticScheduleBitNetConfig, BitNetModel
        
        # Create config with same parameters as training
        config = QuadraticScheduleBitNetConfig(
            vocab_size=128256,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            max_position_embeddings=2048,
            use_layer_skipping=True,
            skip_probability=0.1,
            min_layers_to_keep=2,
            use_early_exit=True,
            early_exit_threshold=0.95,
            dropout_schedule='quadratic',
            quadratic_constant=0.3
        )
        
        # Create model
        model = BitNetModel(config)
        
        print("✓ Successfully created Quadratic BitNet model")
        return model, config
        
    except Exception as e:
        print(f"✗ Error creating Quadratic BitNet model: {e}")
        return None, None

def create_quadratic_hbitlinear_model():
    """Create a fresh quadratic H-BitLinear model using the training script."""
    print("Creating fresh Quadratic H-BitLinear model...")
    
    try:
        # Import the training script modules
        from train_quadratic_hbitlinear import QuadraticScheduleHBitLinearConfig, BitNetModel2
        
        # Create config with same parameters as training
        config = QuadraticScheduleHBitLinearConfig(
            vocab_size=128256,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            max_position_embeddings=2048,
            use_layer_skipping=True,
            skip_probability=0.1,
            min_layers_to_keep=2,
            use_early_exit=True,
            early_exit_threshold=0.95,
            dropout_schedule='quadratic',
            quadratic_constant=0.3
        )
        
        # Create model
        model = BitNetModel2(config)
        
        print("✓ Successfully created Quadratic H-BitLinear model")
        return model, config
        
    except Exception as e:
        print(f"✗ Error creating Quadratic H-BitLinear model: {e}")
        return None, None

def analyze_model(model, config, model_name):
    """Analyze a model's parameters."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_name.upper()}")
    print(f"{'='*60}")
    
    if model is None:
        print("No model to analyze")
        return None
    
    # Count parameters
    total_params, trainable_params, param_details = count_parameters_in_model(model)
    
    print(f"\n📊 PARAMETER SUMMARY:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"  Total Size: {total_params * 4 / (1024**3):.2f} GB (assuming float32)")
    
    if config:
        print(f"\n🏗️  MODEL ARCHITECTURE:")
        print(f"  Hidden Size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  Number of Layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"  Number of Heads: {getattr(config, 'num_attention_heads', 'N/A')}")
        print(f"  Vocabulary Size: {getattr(config, 'vocab_size', 'N/A')}")
        print(f"  Model Type: {type(model).__name__}")
    
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

def main():
    print("Creating fresh models to count parameters...")
    
    # Create both models
    quadratic_model, quadratic_config = create_quadratic_model()
    hbitlinear_model, hbitlinear_config = create_quadratic_hbitlinear_model()
    
    # Analyze each model
    quadratic_stats = analyze_model(quadratic_model, quadratic_config, "Quadratic BitNet")
    hbitlinear_stats = analyze_model(hbitlinear_model, hbitlinear_config, "Quadratic H-BitLinear")
    
    # Compare if both created successfully
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
        
        # Show detailed comparison
        print(f"\n📈 DETAILED COMPARISON:")
        print(f"  {'Metric':20s} {'BitNet':>12s} {'H-BitLinear':>12s} {'Difference':>12s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'Total Params':20s} {quadratic_stats['total_params']:>12,} {hbitlinear_stats['total_params']:>12,} {diff:>+12,}")
        print(f"  {'Trainable Params':20s} {quadratic_stats['trainable_params']:>12,} {hbitlinear_stats['trainable_params']:>12,} {hbitlinear_stats['trainable_params'] - quadratic_stats['trainable_params']:>+12,}")
        
        # Compare layer types
        print(f"\n🔍 LAYER TYPE COMPARISON:")
        all_layer_types = set(quadratic_stats['layer_stats'].keys()) | set(hbitlinear_stats['layer_stats'].keys())
        
        for layer_type in sorted(all_layer_types):
            bitnet_params = quadratic_stats['layer_stats'].get(layer_type, {}).get('params', 0)
            hbitlinear_params = hbitlinear_stats['layer_stats'].get(layer_type, {}).get('params', 0)
            diff = hbitlinear_params - bitnet_params
            
            print(f"  {layer_type:20s}: {bitnet_params:>8,} vs {hbitlinear_params:>8,} ({diff:>+8,})")

if __name__ == "__main__":
    main()
