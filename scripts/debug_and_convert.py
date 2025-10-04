#!/usr/bin/env python3
"""
BitNet Checkpoint Debug and SafeTensors Conversion Script

This script:
1. Loads a BitNet checkpoint and thoroughly validates it
2. Checks for all required weights and components
3. Reports missing weights and issues
4. Converts to SafeTensors in bf16 format if validation passes
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Set, List, Optional
from collections import defaultdict

import torch
from safetensors.torch import save_file


class BitNetCheckpointDebugger:
    """Debug and validate BitNet checkpoints before conversion."""
    
    def __init__(self, checkpoint_path: str, output_path: Optional[str] = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path) if output_path else None
        
        # Expected key patterns for a complete BitNet model
        self.expected_patterns = {
            'embeddings': ['embed_tokens.weight', 'embed_positions.weight'],
            'lm_head': ['lm_head.weight'],
            'layer_norm': ['layer_norm.weight', 'layer_norm.bias'],
            'attention': ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight'],
            'attention_scales': ['q_proj.weight_scale', 'k_proj.weight_scale', 
                                'v_proj.weight_scale', 'o_proj.weight_scale'],
            'feedforward': ['up_proj.weight', 'down_proj.weight'],
            'feedforward_scales': ['up_proj.weight_scale', 'down_proj.weight_scale'],
            'layer_norms': ['self_attn_norm.norm.weight', 'self_attn_norm.norm.bias',
                          'feed_forward_norm.norm.weight', 'feed_forward_norm.norm.bias'],
        }
    
    def load_checkpoint(self) -> Dict:
        """Load checkpoint with multiple fallback strategies."""
        print(f"\n{'='*80}")
        print("STEP 1: LOADING CHECKPOINT")
        print(f"{'='*80}")
        print(f"Loading from: {self.checkpoint_path}")
        
        try:
            # Try weights_only=True first (safer)
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
            print("✓ Loaded with weights_only=True")
        except Exception as e:
            print(f"⚠ weights_only=True failed: {e}")
            print("  Trying with weights_only=False...")
            
            # Add safe globals for numpy
            import torch.serialization
            torch.serialization.add_safe_globals([
                'numpy.core.multiarray._reconstruct',
                'numpy._core.multiarray.scalar',
                'numpy.dtype'
            ])
            
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            print("✓ Loaded with weights_only=False")
        
        return checkpoint
    
    def extract_state_dict(self, checkpoint: Dict) -> Optional[Dict]:
        """Extract state dict from various checkpoint formats."""
        print(f"\n{'='*80}")
        print("STEP 2: EXTRACTING STATE DICT")
        print(f"{'='*80}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint is a dictionary with keys: {list(checkpoint.keys())}")
            
            # Try different extraction strategies
            strategies = [
                ('model_state_dict', lambda c: c.get('model_state_dict')),
                ('state_dict', lambda c: c.get('state_dict')),
                ('model', lambda c: c.get('model')),
                ('direct_tensors', lambda c: c if any(isinstance(v, torch.Tensor) for v in c.values()) else None),
            ]
            
            for name, extractor in strategies:
                state_dict = extractor(checkpoint)
                if state_dict is not None:
                    print(f"✓ Extracted state_dict using strategy: {name}")
                    return state_dict
            
            print("✗ Could not extract state_dict from checkpoint")
            return None
        else:
            print("✓ Checkpoint is already a state_dict")
            return checkpoint
    
    def analyze_state_dict(self, state_dict: Dict) -> Dict:
        """Comprehensive analysis of state dict."""
        print(f"\n{'='*80}")
        print("STEP 3: ANALYZING STATE DICT")
        print(f"{'='*80}")
        
        analysis = {
            'total_keys': len(state_dict),
            'total_parameters': 0,
            'total_size_bytes': 0,
            'dtypes': defaultdict(int),
            'key_groups': defaultdict(list),
            'layers': set(),
            'missing_patterns': [],
            'found_patterns': [],
        }
        
        # Analyze each tensor
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                analysis['total_parameters'] += value.numel()
                analysis['total_size_bytes'] += value.element_size() * value.numel()
                analysis['dtypes'][str(value.dtype)] += 1
                
                # Group by first component
                prefix = key.split('.')[0]
                analysis['key_groups'][prefix].append(key)
                
                # Extract layer numbers
                if 'layers.' in key:
                    layer_num = key.split('layers.')[1].split('.')[0]
                    if layer_num.isdigit():
                        analysis['layers'].add(int(layer_num))
        
        # Check for expected patterns
        all_keys = set(state_dict.keys())
        
        # Check embeddings and core components
        for pattern_type, patterns in self.expected_patterns.items():
            found = any(any(p in key for p in patterns) for key in all_keys)
            if found:
                analysis['found_patterns'].append(pattern_type)
            else:
                analysis['missing_patterns'].append(pattern_type)
        
        # Print summary
        print(f"\nBasic Statistics:")
        print(f"  Total keys: {analysis['total_keys']}")
        print(f"  Total parameters: {analysis['total_parameters']:,}")
        print(f"  Total size: {analysis['total_size_bytes'] / (1024**3):.2f} GB")
        print(f"  Number of layers: {len(analysis['layers'])} (layers: {sorted(analysis['layers'])})")
        
        print(f"\nData Types:")
        for dtype, count in sorted(analysis['dtypes'].items()):
            print(f"  {dtype}: {count} tensors")
        
        print(f"\nKey Groups:")
        for prefix, keys in sorted(analysis['key_groups'].items()):
            print(f"  {prefix}: {len(keys)} keys")
        
        print(f"\n✓ Found Components: {', '.join(analysis['found_patterns'])}")
        if analysis['missing_patterns']:
            print(f"✗ Missing Components: {', '.join(analysis['missing_patterns'])}")
        
        return analysis
    
    def detailed_validation(self, state_dict: Dict, analysis: Dict) -> bool:
        """Perform detailed validation of the checkpoint."""
        print(f"\n{'='*80}")
        print("STEP 4: DETAILED VALIDATION")
        print(f"{'='*80}")
        
        all_checks_passed = True
        
        # Check 1: Embeddings
        print("\n[Check 1] Embeddings:")
        has_embed_tokens = any('embed_tokens.weight' in k for k in state_dict.keys())
        has_embed_pos = any('embed_positions.weight' in k for k in state_dict.keys())
        
        if has_embed_tokens:
            embed_key = [k for k in state_dict.keys() if 'embed_tokens.weight' in k][0]
            vocab_size, hidden_size = state_dict[embed_key].shape
            print(f"  ✓ embed_tokens found: vocab_size={vocab_size}, hidden_size={hidden_size}")
        else:
            print(f"  ✗ embed_tokens.weight NOT FOUND")
            all_checks_passed = False
        
        if has_embed_pos:
            pos_key = [k for k in state_dict.keys() if 'embed_positions.weight' in k][0]
            max_pos, hidden_size = state_dict[pos_key].shape
            print(f"  ✓ embed_positions found: max_positions={max_pos}")
        else:
            print(f"  ✗ embed_positions.weight NOT FOUND")
            all_checks_passed = False
        
        # Check 2: LM Head
        print("\n[Check 2] LM Head:")
        has_lm_head = any('lm_head.weight' in k for k in state_dict.keys())
        if has_lm_head:
            lm_head_key = [k for k in state_dict.keys() if 'lm_head.weight' in k][0]
            vocab_size, hidden_size = state_dict[lm_head_key].shape
            print(f"  ✓ lm_head found: shape={state_dict[lm_head_key].shape}")
        else:
            print(f"  ✗ lm_head.weight NOT FOUND")
            all_checks_passed = False
        
        # Check 3: Layer structure
        print("\n[Check 3] Layer Structure:")
        num_layers = len(analysis['layers'])
        
        if num_layers > 0:
            print(f"  ✓ Found {num_layers} layers")
            
            # Check first layer in detail
            layer_0_keys = [k for k in state_dict.keys() if k.startswith('layers.0.')]
            print(f"  Layer 0 has {len(layer_0_keys)} keys")
            
            # Check for attention weights
            has_q = any('q_proj.weight' in k for k in layer_0_keys)
            has_k = any('k_proj.weight' in k for k in layer_0_keys)
            has_v = any('v_proj.weight' in k for k in layer_0_keys)
            has_o = any('o_proj.weight' in k for k in layer_0_keys)
            
            print(f"    Attention projections: q={has_q}, k={has_k}, v={has_v}, o={has_o}")
            if not all([has_q, has_k, has_v, has_o]):
                print(f"    ✗ Missing attention weights!")
                all_checks_passed = False
            
            # Check for feedforward weights
            has_up = any('up_proj.weight' in k for k in layer_0_keys)
            has_down = any('down_proj.weight' in k for k in layer_0_keys)
            
            print(f"    Feedforward projections: up={has_up}, down={has_down}")
            if not all([has_up, has_down]):
                print(f"    ✗ Missing feedforward weights!")
                all_checks_passed = False
        else:
            print(f"  ✗ No layers found!")
            all_checks_passed = False
        
        # Check 4: Quantization scales
        print("\n[Check 4] Quantization Scales:")
        scale_keys = [k for k in state_dict.keys() if 'weight_scale' in k]
        if scale_keys:
            print(f"  ✓ Found {len(scale_keys)} weight_scale tensors")
            # Show sample
            sample_key = scale_keys[0]
            print(f"    Example: {sample_key} = {state_dict[sample_key].item():.6f}")
        else:
            print(f"  ⚠ No weight_scale tensors found (might be okay for unquantized checkpoints)")
        
        return all_checks_passed
    
    def show_sample_keys(self, state_dict: Dict):
        """Display sample keys from each major component."""
        print(f"\n{'='*80}")
        print("STEP 5: SAMPLE KEYS FROM EACH COMPONENT")
        print(f"{'='*80}")
        
        components = {
            'Embeddings': lambda k: 'embed' in k,
            'Layer 0 Attention': lambda k: k.startswith('layers.0.') and 'attn' in k,
            'Layer 0 FFN': lambda k: k.startswith('layers.0.') and 'feed_forward' in k,
            'Layer Norms': lambda k: 'norm' in k,
            'LM Head': lambda k: 'lm_head' in k,
            'Routing': lambda k: 'routing' in k,
        }
        
        for component_name, filter_fn in components.items():
            matching_keys = [k for k in state_dict.keys() if filter_fn(k)]
            if matching_keys:
                print(f"\n{component_name} ({len(matching_keys)} keys):")
                for key in sorted(matching_keys)[:5]:
                    shape = state_dict[key].shape
                    dtype = state_dict[key].dtype
                    print(f"  {key}: {shape} ({dtype})")
                if len(matching_keys) > 5:
                    print(f"  ... and {len(matching_keys) - 5} more")
    
    def convert_to_bf16_safetensors(self, state_dict: Dict) -> bool:
        """Convert state dict to bf16 and save as SafeTensors."""
        print(f"\n{'='*80}")
        print("STEP 6: CONVERTING TO BF16 SAFETENSORS")
        print(f"{'='*80}")
        
        if self.output_path is None:
            print("✗ No output path specified, skipping conversion")
            return False
        
        print(f"Output path: {self.output_path}")
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert all tensors to bf16
        converted_state_dict = {}
        conversion_stats = defaultdict(int)
        
        print("\nConverting tensors to bf16...")
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                original_dtype = value.dtype
                
                # Convert floating point tensors to bf16
                if original_dtype in [torch.float32, torch.float64, torch.float16]:
                    converted = value.cpu().contiguous().to(torch.bfloat16)
                    conversion_stats[f"{original_dtype} -> bfloat16"] += 1
                elif original_dtype == torch.bfloat16:
                    converted = value.cpu().contiguous()
                    conversion_stats["already bfloat16"] += 1
                else:
                    # Keep non-floating point as-is (int, long, bool)
                    converted = value.cpu().contiguous()
                    conversion_stats[f"kept as {original_dtype}"] += 1
                
                converted_state_dict[key] = converted
        
        print("\nConversion statistics:")
        for conversion_type, count in sorted(conversion_stats.items()):
            print(f"  {conversion_type}: {count} tensors")
        
        # Calculate size
        total_size = sum(t.element_size() * t.numel() for t in converted_state_dict.values())
        print(f"\nTotal size: {total_size / (1024**3):.2f} GB")
        
        # Create metadata
        metadata = {
            'format': 'pt',
            'model_type': 'bitnet',
            'torch_dtype': 'bfloat16',
            'total_parameters': str(sum(t.numel() for t in converted_state_dict.values())),
            'num_layers': str(len(set(k.split('.')[1] for k in converted_state_dict.keys() if k.startswith('layers.')))),
            'conversion_script': 'debug_and_convert.py',
        }
        
        # Save to SafeTensors
        print(f"\nSaving to {self.output_path}...")
        save_file(converted_state_dict, str(self.output_path), metadata=metadata)
        
        file_size = self.output_path.stat().st_size / (1024**3)
        print(f"✓ Saved successfully!")
        print(f"  File size: {file_size:.2f} GB")
        
        # Save config if available in same directory
        config_source = self.checkpoint_path.parent / "config.json"
        if config_source.exists():
            config_dest = self.output_path.parent / "config.json"
            import shutil
            shutil.copy(config_source, config_dest)
            print(f"✓ Copied config.json to {config_dest}")
        
        return True
    
    def run_full_debug(self) -> bool:
        """Run complete debug and conversion pipeline."""
        try:
            # Step 1: Load checkpoint
            checkpoint = self.load_checkpoint()
            
            # Step 2: Extract state dict
            state_dict = self.extract_state_dict(checkpoint)
            if state_dict is None:
                print("\n✗ FAILED: Could not extract state dict")
                return False
            
            # Step 3: Analyze
            analysis = self.analyze_state_dict(state_dict)
            
            # Step 4: Validate
            validation_passed = self.detailed_validation(state_dict, analysis)
            
            # Step 5: Show samples
            self.show_sample_keys(state_dict)
            
            # Final summary
            print(f"\n{'='*80}")
            print("VALIDATION SUMMARY")
            print(f"{'='*80}")
            
            if validation_passed:
                print("✓ ALL VALIDATION CHECKS PASSED")
                print("  This checkpoint contains all required weights for inference.")
                
                # Step 6: Convert if validation passed and output path specified
                if self.output_path:
                    conversion_success = self.convert_to_bf16_safetensors(state_dict)
                    if conversion_success:
                        print("\n✓ CONVERSION COMPLETED SUCCESSFULLY")
                        return True
                else:
                    print("\n  Note: No output path specified, skipping conversion.")
                    print("  To convert, run with: --output path/to/model.safetensors")
                    return True
            else:
                print("✗ VALIDATION FAILED")
                print("  This checkpoint is INCOMPLETE and missing critical weights.")
                print("  Cannot proceed with conversion.")
                print("\n  Possible causes:")
                print("  1. Checkpoint was saved with strict=False and only partial weights")
                print("  2. Training script saved only certain components (e.g., layer norms)")
                print("  3. Checkpoint is from an intermediate/debug save")
                print("\n  Solution:")
                print("  - Check your training script's save logic")
                print("  - Ensure model.state_dict() includes all weights")
                print("  - Retrain and save a complete checkpoint")
                return False
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Debug and convert BitNet checkpoints to SafeTensors (bf16)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug only (no conversion):
  python debug_and_convert.py checkpoint-1000/model.pt

  # Debug and convert:
  python debug_and_convert.py checkpoint-1000/model.pt --output model.safetensors
  
  # Convert with custom output directory:
  python debug_and_convert.py checkpoint-1000/model.pt --output ./converted/model.safetensors
        """
    )
    
    parser.add_argument('checkpoint', type=str,
                       help='Path to BitNet checkpoint (.pt file)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for SafeTensors file (bf16). If not specified, only debug mode.')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run debugger
    debugger = BitNetCheckpointDebugger(args.checkpoint, args.output)
    success = debugger.run_full_debug()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

