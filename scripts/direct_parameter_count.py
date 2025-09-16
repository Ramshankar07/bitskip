#!/usr/bin/env python3
"""
Direct parameter counting by examining checkpoint structure without loading custom classes.
"""

import os
import sys
import torch
import pickle
import argparse

def safe_load_checkpoint(checkpoint_path):
    """Safely load checkpoint by extracting only tensor data."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # Try to load with weights_only=True first
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print("✓ Loaded with weights_only=True")
        return checkpoint
    except Exception as e:
        print(f"weights_only=True failed: {e}")
        
        # Try to extract tensors manually
        try:
            print("Attempting manual tensor extraction...")
            
            # Use a custom unpickler that only extracts tensors
            class TensorExtractor(pickle.Unpickler):
                def __init__(self, file):
                    super().__init__(file)
                    self.tensors = {}
                    self.current_key = None
                
                def find_class(self, module, name):
                    # Replace any custom class with a simple object
                    return object
                
                def load(self):
                    try:
                        obj = super().load()
                        if isinstance(obj, dict):
                            self._extract_tensors_from_dict(obj, "")
                        return self.tensors
                    except Exception as e:
                        print(f"Error during extraction: {e}")
                        return self.tensors
                
                def _extract_tensors_from_dict(self, obj, prefix):
                    """Recursively extract tensors from nested dictionaries."""
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            new_prefix = f"{prefix}.{key}" if prefix else key
                            if isinstance(value, torch.Tensor):
                                self.tensors[new_prefix] = value
                            elif isinstance(value, dict):
                                self._extract_tensors_from_dict(value, new_prefix)
            
            with open(checkpoint_path, 'rb') as f:
                extractor = TensorExtractor(f)
                tensors = extractor.load()
                
            if tensors:
                print(f"✓ Extracted {len(tensors)} tensors manually")
                return tensors
            else:
                print("✗ No tensors extracted")
                return None
                
        except Exception as e2:
            print(f"✗ Manual extraction failed: {e2}")
            return None

def count_parameters_in_tensors(tensors):
    """Count parameters in a dictionary of tensors."""
    
    if not tensors:
        return 0, []
    
    total_params = 0
    param_details = []
    
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            param_count = tensor.numel()
            total_params += param_count
            
            param_details.append({
                'name': name,
                'shape': list(tensor.shape),
                'params': param_count,
                'dtype': str(tensor.dtype)
            })
    
    return total_params, param_details

def analyze_checkpoint(checkpoint_path, model_name):
    """Analyze a checkpoint file."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = safe_load_checkpoint(checkpoint_path)
    
    if checkpoint is None:
        print("✗ Failed to load checkpoint")
        return None
    
    # Count parameters
    total_params, param_details = count_parameters_in_tensors(checkpoint)
    
    if total_params == 0:
        print("✗ No parameters found")
        return None
    
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
    parser = argparse.ArgumentParser(description="Count parameters in model checkpoints")
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
    model1_stats = analyze_checkpoint(args.model1, "Quadratic BitNet")
    model2_stats = analyze_checkpoint(args.model2, "Quadratic H-BitLinear")
    
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
