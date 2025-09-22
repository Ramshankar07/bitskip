#!/usr/bin/env python3
"""
Calculate optimal 1B and 2B parameter configurations for H200 GPU scaling study.
H200 GPU has 141GB VRAM, so we need to ensure both models fit comfortably.
"""

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

def find_optimal_config(target_params, vocab_size=128256, max_position_embeddings=2048):
    """Find optimal configuration for target parameter count."""
    
    print(f"Searching for {target_params/1e9:.1f}B parameter configuration...")
    print(f"Target: {target_params:,} parameters")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Max position embeddings: {max_position_embeddings:,}")
    
    best_config = None
    best_diff = float('inf')
    
    # Try different combinations - focus on powers of 2 for BitNet compatibility
    for hidden_size in [1024, 1536, 2048, 2560, 3072, 4096]:
        for num_layers in [12, 16, 20, 24, 28, 32, 36, 40]:
            for num_heads in [16, 20, 24, 28, 32, 40, 48, 64]:
                # Ensure num_heads divides hidden_size
                if hidden_size % num_heads != 0:
                    continue
                
                config = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'vocab_size': vocab_size,
                    'max_position_embeddings': max_position_embeddings
                }
                
                size_info = calculate_model_size(**config)
                total_params = size_info['total_params']
                diff = abs(total_params - target_params)
                
                if diff < best_diff:
                    best_diff = diff
                    best_config = config
                    best_size_info = size_info
                
                # Only print close matches to avoid spam
                if diff < target_params * 0.1:  # Within 10% of target
                    print(f"  Hidden: {hidden_size:4d}, Layers: {num_layers:2d}, Heads: {num_heads:2d} -> {total_params:10,} params (diff: {diff:+,})")
    
    return best_config, best_size_info

def estimate_training_memory(model_params, batch_size=1, seq_length=1024, precision='fp16'):
    """Estimate training memory requirements."""
    
    # Model parameters in bytes
    if precision == 'fp16':
        model_memory = model_params * 2  # 2 bytes per parameter
    else:  # fp32
        model_memory = model_params * 4  # 4 bytes per parameter
    
    # Gradient memory (same size as model)
    gradient_memory = model_memory
    
    # Optimizer memory (AdamW: 2x model size for momentum + variance)
    optimizer_memory = model_memory * 2
    
    # Activation memory (rough estimate)
    # This is a simplified calculation - actual activations depend on architecture
    activation_memory = batch_size * seq_length * 1024 * 2  # Rough estimate
    
    # Total training memory
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'model_memory_gb': model_memory / (1024**3),
        'gradient_memory_gb': gradient_memory / (1024**3),
        'optimizer_memory_gb': optimizer_memory / (1024**3),
        'activation_memory_gb': activation_memory / (1024**3),
        'total_memory_gb': total_memory / (1024**3)
    }

def main():
    print("="*80)
    print("H200 GPU SCALING STUDY CONFIGURATION CALCULATOR")
    print("="*80)
    print("H200 GPU Specifications:")
    print("  - VRAM: 141 GB")
    print("  - Memory Bandwidth: 4.8 TB/s")
    print("  - Compute: 67 TFLOPS (FP64/FP32)")
    print("="*80)
    
    # Calculate 1B parameter configuration
    print("\n🎯 CALCULATING 1B PARAMETER CONFIGURATION:")
    config_1b, size_info_1b = find_optimal_config(1_000_000_000)
    
    print(f"\n✅ BEST 1B CONFIGURATION:")
    print(f"  Hidden Size: {config_1b['hidden_size']}")
    print(f"  Number of Layers: {config_1b['num_layers']}")
    print(f"  Number of Heads: {config_1b['num_heads']}")
    print(f"  Vocabulary Size: {config_1b['vocab_size']}")
    print(f"  Max Position Embeddings: {config_1b['max_position_embeddings']}")
    
    print(f"\n📊 1B PARAMETER BREAKDOWN:")
    print(f"  Total Parameters: {size_info_1b['total_params']:,}")
    print(f"  Token Embeddings: {size_info_1b['token_embeddings']:,} ({size_info_1b['token_embeddings']/size_info_1b['total_params']*100:.1f}%)")
    print(f"  Position Embeddings: {size_info_1b['position_embeddings']:,} ({size_info_1b['position_embeddings']/size_info_1b['total_params']*100:.1f}%)")
    print(f"  Layer Parameters: {size_info_1b['layer_params']:,} ({size_info_1b['layer_params']/size_info_1b['total_params']*100:.1f}%)")
    print(f"  LM Head: {size_info_1b['lm_head']:,} ({size_info_1b['lm_head']/size_info_1b['total_params']*100:.1f}%)")
    print(f"  Per Layer: {size_info_1b['per_layer_params']:,}")
    
    # Calculate 2B parameter configuration
    print("\n🎯 CALCULATING 2B PARAMETER CONFIGURATION:")
    config_2b, size_info_2b = find_optimal_config(2_000_000_000)
    
    print(f"\n✅ BEST 2B CONFIGURATION:")
    print(f"  Hidden Size: {config_2b['hidden_size']}")
    print(f"  Number of Layers: {config_2b['num_layers']}")
    print(f"  Number of Heads: {config_2b['num_heads']}")
    print(f"  Vocabulary Size: {config_2b['vocab_size']}")
    print(f"  Max Position Embeddings: {config_2b['max_position_embeddings']}")
    
    print(f"\n📊 2B PARAMETER BREAKDOWN:")
    print(f"  Total Parameters: {size_info_2b['total_params']:,}")
    print(f"  Token Embeddings: {size_info_2b['token_embeddings']:,} ({size_info_2b['token_embeddings']/size_info_2b['total_params']*100:.1f}%)")
    print(f"  Position Embeddings: {size_info_2b['position_embeddings']:,} ({size_info_2b['position_embeddings']/size_info_2b['total_params']*100:.1f}%)")
    print(f"  Layer Parameters: {size_info_2b['layer_params']:,} ({size_info_2b['layer_params']/size_info_2b['total_params']*100:.1f}%)")
    print(f"  LM Head: {size_info_2b['lm_head']:,} ({size_info_2b['lm_head']/size_info_2b['total_params']*100:.1f}%)")
    print(f"  Per Layer: {size_info_2b['per_layer_params']:,}")
    
    # Memory estimates for both configurations
    print(f"\n💾 MEMORY ESTIMATES (FP16 Training):")
    
    for config_name, config, size_info in [("1B", config_1b, size_info_1b), ("2B", config_2b, size_info_2b)]:
        print(f"\n{config_name} Model Memory Requirements:")
        
        # Try different batch sizes to find optimal
        for batch_size in [1, 2, 4, 8]:
            memory_est = estimate_training_memory(
                size_info['total_params'], 
                batch_size=batch_size, 
                seq_length=config['max_position_embeddings'],
                precision='fp16'
            )
            
            print(f"  Batch Size {batch_size:2d}: {memory_est['total_memory_gb']:5.1f} GB total")
            print(f"    - Model: {memory_est['model_memory_gb']:5.1f} GB")
            print(f"    - Gradients: {memory_est['gradient_memory_gb']:5.1f} GB") 
            print(f"    - Optimizer: {memory_est['optimizer_memory_gb']:5.1f} GB")
            print(f"    - Activations: {memory_est['activation_memory_gb']:5.1f} GB")
            
            if memory_est['total_memory_gb'] < 120:  # Leave some headroom
                print(f"    ✅ Fits comfortably in H200 (141GB)")
            elif memory_est['total_memory_gb'] < 141:
                print(f"    ⚠️  Fits but tight in H200 (141GB)")
            else:
                print(f"    ❌ Too large for H200 (141GB)")
    
    print(f"\n🚀 RECOMMENDED CONFIGURATIONS:")
    print(f"1B Model: {config_1b['hidden_size']}d, {config_1b['num_layers']}L, {config_1b['num_heads']}H")
    print(f"2B Model: {config_2b['hidden_size']}d, {config_2b['num_layers']}L, {config_2b['num_heads']}H")
    print(f"\nBoth models should fit comfortably in H200 GPU with appropriate batch sizes!")

if __name__ == "__main__":
    main()
