#!/usr/bin/env python3
"""
Test script to verify H-BitLinear CUDA FWHT kernel usage.
"""

import torch
import time
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitnet.modeling.kernels import is_available as fwht_cuda_available
from bitnet.modeling.h_bitlinear import HBitLinear

def test_fwht_cuda_availability():
    """Test if FWHT CUDA kernel is available."""
    print("="*60)
    print("Testing FWHT CUDA Kernel Availability")
    print("="*60)
    
    try:
        available = fwht_cuda_available()
        print(f"FWHT CUDA kernel available: {available}")
        
        if available:
            print("✅ CUDA FWHT kernel is available")
        else:
            print("❌ CUDA FWHT kernel is NOT available - will use CPU fallback")
        
        return available
    except Exception as e:
        print(f"❌ Error checking FWHT CUDA availability: {e}")
        return False

def test_hbitlinear_cuda_usage():
    """Test H-BitLinear CUDA usage with timing."""
    print("\n" + "="*60)
    print("Testing H-BitLinear CUDA Usage")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping CUDA test")
        return False
    
    try:
        # Create H-BitLinear layer
        in_features = 1024
        out_features = 1024
        bit_width = 2
        
        layer = HBitLinear(in_features, out_features, bit_width=bit_width)
        layer = layer.cuda()
        
        print(f"H-BitLinear layer created: {in_features} -> {out_features}, bit_width={bit_width}")
        
        # Create test input
        batch_size = 64
        seq_length = 128
        x = torch.randn(batch_size, seq_length, in_features, device='cuda', dtype=torch.float16)
        
        print(f"Test input: {x.shape}")
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = layer(x)
        
        torch.cuda.synchronize()
        
        # Time CPU fallback
        device_type = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
        if device_type != '1':
            print("Note: CUDA_LAUNCH_BLOCKING not set, CUDA kernel timings may be misleading")
        
        # Test with timing
        num_runs = 100
        
        # Time evaluation runs
        times = []
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                output = layer(x)
            
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i % 20 == 0:
                print(f"Run {i}/{num_runs}: {times[-1]*1000:.2f}ms")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nTiming Results:")
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Min time: {min_time*1000:.2f} ms")
        print(f"  Max time: {max_time*1000:.2f} ms")
        print(f"  Throughput: {x.numel()/avg_time/1e9:.2f} GB/s")
        
        # Check output
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during H-BitLinear test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_cuda_usage():
    """Test H-BitLinear usage in a full BitNet model."""
    print("\n" + "="*60)
    print("Testing H-BitLinear in BitNet Model")
    print("="*60)
    
    try:
        from bitnet.modeling.model2 import BitNetModel2
        from bitnet.utils.default_config import DefaultConfig
        
        # Create a small BitNet2 model
        config = DefaultConfig(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1000,
            max_position_embeddings=128
        )
        
        model = BitNetModel2(config).cuda()
        print(f"Created BitNet2 model with H-BitLinear layers")
        print(f"Model contains {sum('h_bitlinear' in name.lower() for name, _ in model.named_modules())} H-BitLinear layers")
        
        # Create test input
        batch_size = 4
        seq_length = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device='cuda')
        
        # Test forward pass
        with torch.no_grad():
            output = model(input_ids, return_dict=True)
        
        print(f"Model forward pass successful")
        print(f"Output logits shape: {output.logits.shape}")
        print(f"Model device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("H-BitLinear CUDA Kernel Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test FWHT CUDA availability
    fwht_available = test_fwht_cuda_availability()
    
    # Test H-BitLinear CUDA usage
    if fwht_available:
        test_hbitlinear_cuda_usage()
        
        # Test in full model
        test_model_cuda_usage()
    else:
        print("\n⚠️  Skipping detailed tests because FWHT CUDA kernel is not available")
        print("This means H-BitLinear will use CPU fallback (slower)")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    main()
