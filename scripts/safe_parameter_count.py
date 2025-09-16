#!/usr/bin/env python3
"""
Count parameters using PyTorch's safe globals approach.
"""

import os
import sys
import torch
import argparse

def count_parameters_with_safe_globals(checkpoint_path, model_name):
    """Count parameters using safe globals."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Add the custom classes to safe globals
        torch.serialization.add_safe_globals([
            'QuadraticScheduleBitNetConfig',
            'QuadraticScheduleHBitLinearConfig',
            'QuadraticScheduleBitNetConfig',
            'QuadraticScheduleHBitLinearConfig'
        ])
        
        # Try to load with weights_only=True
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print("✓ Loaded with weights_only=True using safe globals")
        
    except Exception as e:
        print(f"Safe globals approach failed: {e}")
        print("Trying weights_only=False...")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print("✓ Loaded with weights_only=False")
        except Exception as e2:
            print(f"✗ Failed to load checkpoint: {e2}")
            return None
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Found model_state_dict with {len(state_dict)} parameters")
        else:
            print("No model_state_dict found, using entire checkpoint")
            state_dict = checkpoint
    else:
        print("Checkpoint is not a dictionary")
        return None
    
    # Count parameters
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
    
    print(f"\n📊 PARAMETER SUMMARY:")
    print(f"  Total Parameters: {total_params:,}")
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
        'param_details': param_details,
        'layer_stats': layer_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Count parameters using safe globals")
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
    
    # Analyze both models
    model1_stats = count_parameters_with_safe_globals(args.model1, "Quadratic BitNet")
    model2_stats = count_parameters_with_safe_globals(args.model2, "Quadratic H-BitLinear")
    
    # Compare if both loaded successfully
    if args.compare and model1_stats and model2_stats:
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON")
        print(f"{'='*80}")
        
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"  {'Quadratic BitNet':20s}: {model1_stats['total_params']:8,} parameters")
        print(f"  {'Quadratic H-BitLinear':20s}: {model2_stats['total_params']:8,} parameters")
        
        diff = model2_stats['total_params'] - model1_stats['total_params']
        diff_pct = (diff / model1_stats['total_params']) * 100 if model1_stats['total_params'] > 0 else 0
        
        print(f"  Difference: {diff:+,} parameters ({diff_pct:+.1f}%)")
        
        if diff > 0:
            print(f"  H-BitLinear model has {diff:,} more parameters than BitNet model")
        elif diff < 0:
            print(f"  BitNet model has {abs(diff):,} more parameters than H-BitLinear model")
        else:
            print(f"  Both models have the same number of parameters")

if __name__ == "__main__":
    main()
