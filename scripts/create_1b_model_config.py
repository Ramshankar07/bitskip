#!/usr/bin/env python3
"""
Create a 1B parameter BitNet model configuration.

Current models are ~368M parameters, so we need to scale up by ~2.7x.
We can do this by increasing:
1. Hidden size (most effective)
2. Number of layers
3. Vocabulary size (if needed)
"""

import os
import sys
import torch
import argparse

# Add current directory to path
sys.path.insert(0, os.getcwd())

def calculate_model_size(hidden_size, num_layers, num_heads, vocab_size, max_position_embeddings=2048):
    """Calculate approximate model size based on architecture."""
    
    # Token embeddings: vocab_size * hidden_size
    token_embeddings = vocab_size * hidden_size
    
    # Position embeddings: max_position_embeddings * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    
    # Per layer parameters:
    # - Self attention: 4 * hidden_size^2 (Q, K, V, O projections)
    # - Feed forward: 2 * hidden_size * (4 * hidden_size) = 8 * hidden_size^2
    # - Layer norms: 2 * hidden_size (per layer)
    per_layer_params = 4 * hidden_size * hidden_size + 8 * hidden_size * hidden_size + 2 * hidden_size
    
    # All layers
    layer_params = num_layers * per_layer_params
    
    # Language model head: hidden_size * vocab_size
    lm_head = hidden_size * vocab_size
    
    # Final layer norm: hidden_size
    final_norm = hidden_size
    
    total_params = token_embeddings + position_embeddings + layer_params + lm_head + final_norm
    
    return {
        'total_params': total_params,
        'token_embeddings': token_embeddings,
        'position_embeddings': position_embeddings,
        'layer_params': layer_params,
        'lm_head': lm_head,
        'final_norm': final_norm,
        'per_layer_params': per_layer_params
    }

def find_1b_config(target_params=1_000_000_000, vocab_size=128256):
    """Find configuration that gives approximately 1B parameters."""
    
    print(f"Searching for 1B parameter configuration...")
    print(f"Target: {target_params:,} parameters")
    print(f"Vocabulary size: {vocab_size:,}")
    
    best_config = None
    best_diff = float('inf')
    
    # Try different combinations
    for hidden_size in [1536, 1792, 2048, 2304, 2560]:
        for num_layers in [16, 20, 24, 28, 32]:
            for num_heads in [16, 20, 24, 28, 32]:
                # Ensure num_heads divides hidden_size
                if hidden_size % num_heads != 0:
                    continue
                
                config = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'vocab_size': vocab_size,
                    'max_position_embeddings': 2048
                }
                
                size_info = calculate_model_size(**config)
                total_params = size_info['total_params']
                diff = abs(total_params - target_params)
                
                if diff < best_diff:
                    best_diff = diff
                    best_config = config
                    best_size_info = size_info
                
                print(f"  Hidden: {hidden_size:4d}, Layers: {num_layers:2d}, Heads: {num_heads:2d} -> {total_params:10,} params (diff: {diff:+,})")
    
    return best_config, best_size_info

def create_1b_training_script(config, model_type="bitnet"):
    """Create a training script for the 1B model."""
    
    script_name = f"train_1b_{model_type}.py"
    
    if model_type == "bitnet":
        base_script = "train_quadratic.py"
        model_class = "BitNetModel"
        config_class = "QuadraticScheduleBitNetConfig"
    else:  # hbitlinear
        base_script = "train_quadratic_hbitlinear.py"
        model_class = "BitNetModel2"
        config_class = "QuadraticScheduleHBitLinearConfig"
    
    print(f"Creating training script: {script_name}")
    print(f"Based on: {base_script}")
    print(f"Model class: {model_class}")
    print(f"Config class: {config_class}")
    
    # Read the base script
    with open(base_script, 'r') as f:
        content = f.read()
    
    # Replace configuration values
    content = content.replace(
        'class QuadraticScheduleBitNetConfig(DefaultConfig):',
        f'class QuadraticSchedule1B{model_type.title()}Config(DefaultConfig):'
    )
    
    content = content.replace(
        'QuadraticScheduleBitNetConfig',
        f'QuadraticSchedule1B{model_type.title()}Config'
    )
    
    content = content.replace(
        'BitNetModel(config)',
        f'{model_class}(config)'
    )
    
    # Update the configuration section
    config_section = f'''    # Create 1B parameter configuration
    config_kwargs = {{
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "vocab_size": actual_vocab_size,
        # 1B parameter architecture
        "hidden_size": {config['hidden_size']},
        "num_hidden_layers": {config['num_layers']},
        "num_attention_heads": {config['num_heads']},
        "max_position_embeddings": {config['max_position_embeddings']},
        # All features enabled
        "use_layer_skipping": True,
        "skip_probability": 0.1,
        "min_layers_to_keep": 4,  # Increased for larger model
        "use_early_exit": True,
        "early_exit_threshold": args.early_exit_threshold,
        "dropout_schedule": 'quadratic',
        "quadratic_constant": args.quadratic_constant,
        "output_dir": args.output_dir
    }}'''
    
    # Find and replace the config section
    import re
    pattern = r'config_kwargs = \{[^}]+\}'
    content = re.sub(pattern, config_section, content, flags=re.DOTALL)
    
    # Update script description
    content = content.replace(
        'Quadratic Schedule BitNet Training Script - Early Exit + Quadratic Schedule',
        f'1B Parameter {model_type.title()} Training Script - Early Exit + Quadratic Schedule'
    )
    
    content = content.replace(
        'QUADRATIC SCHEDULE BITNET TRAINING - EARLY EXIT + QUADRATIC SCHEDULE',
        f'1B PARAMETER {model_type.upper()} TRAINING - EARLY EXIT + QUADRATIC SCHEDULE'
    )
    
    # Write the new script
    with open(script_name, 'w') as f:
        f.write(content)
    
    print(f"✓ Created {script_name}")
    return script_name

def main():
    parser = argparse.ArgumentParser(description="Create 1B parameter model configuration")
    parser.add_argument("--target-params", type=int, default=1_000_000_000, help="Target number of parameters")
    parser.add_argument("--vocab-size", type=int, default=128256, help="Vocabulary size")
    parser.add_argument("--create-scripts", action="store_true", help="Create training scripts")
    parser.add_argument("--model-type", choices=["bitnet", "hbitlinear"], default="bitnet", help="Model type")
    
    args = parser.parse_args()
    
    print("="*80)
    print("1B PARAMETER MODEL CONFIGURATION GENERATOR")
    print("="*80)
    
    # Find best configuration
    config, size_info = find_1b_config(args.target_params, args.vocab_size)
    
    print(f"\n🎯 BEST CONFIGURATION FOUND:")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Number of Layers: {config['num_layers']}")
    print(f"  Number of Heads: {config['num_heads']}")
    print(f"  Vocabulary Size: {config['vocab_size']}")
    print(f"  Max Position Embeddings: {config['max_position_embeddings']}")
    
    print(f"\n📊 PARAMETER BREAKDOWN:")
    print(f"  Total Parameters: {size_info['total_params']:,}")
    print(f"  Token Embeddings: {size_info['token_embeddings']:,} ({size_info['token_embeddings']/size_info['total_params']*100:.1f}%)")
    print(f"  Position Embeddings: {size_info['position_embeddings']:,} ({size_info['position_embeddings']/size_info['total_params']*100:.1f}%)")
    print(f"  Layer Parameters: {size_info['layer_params']:,} ({size_info['layer_params']/size_info['total_params']*100:.1f}%)")
    print(f"  LM Head: {size_info['lm_head']:,} ({size_info['lm_head']/size_info['total_params']*100:.1f}%)")
    print(f"  Final Norm: {size_info['final_norm']:,} ({size_info['final_norm']/size_info['total_params']*100:.1f}%)")
    print(f"  Per Layer: {size_info['per_layer_params']:,}")
    
    print(f"\n💾 MEMORY ESTIMATES:")
    print(f"  Model Size (float32): {size_info['total_params'] * 4 / (1024**3):.2f} GB")
    print(f"  Model Size (float16): {size_info['total_params'] * 2 / (1024**3):.2f} GB")
    print(f"  Training Memory (est): {size_info['total_params'] * 8 / (1024**3):.2f} GB (with gradients + optimizer)")
    
    # Create training scripts if requested
    if args.create_scripts:
        print(f"\n📝 CREATING TRAINING SCRIPTS:")
        script_name = create_1b_training_script(config, args.model_type)
        print(f"✓ Training script created: {script_name}")
        print(f"\n🚀 TO TRAIN THE MODEL:")
        print(f"  python {script_name} --num_steps 1000 --batch_size 1 --max_length 512")
        print(f"  # Adjust batch_size and max_length based on your GPU memory")
    
    print(f"\n✅ 1B parameter model configuration ready!")

if __name__ == "__main__":
    main()

