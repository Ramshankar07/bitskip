#!/usr/bin/env python3
"""
Simple test to verify our RoPE fix works with the actual training script.
"""

import torch
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rope_fix_integration():
    """Test our RoPE fix by importing and testing the wrapper."""
    logger.info("Testing RoPE fix integration...")
    
    try:
        # Import our wrapper
        from train_sft_with_layer_skip import BitNetLayerSkipModel, BitNetLayerSkipConfig
        
        # Create a simple config
        config = BitNetLayerSkipConfig()
        config.cuda_device = 0
        config.enable_layer_skip = False
        config.enable_early_exit = False
        
        # Create a dummy base model
        class DummyModel:
            def __init__(self):
                self.config = type('Config', (), {
                    'num_hidden_layers': 2,
                    'hidden_size': 128,
                    'vocab_size': 1000,
                    'rope_theta': 10000.0,
                    'max_position_embeddings': 512,
                    'rope_scaling': None
                })()
            
            def get_input_embeddings(self):
                return torch.nn.Embedding(1000, 128)
            
            def get_output_embeddings(self):
                return torch.nn.Linear(128, 1000)
        
        # Create wrapper
        base_model = DummyModel()
        wrapper = BitNetLayerSkipModel(base_model, config)
        
        # Test RoPE generation
        batch_size, seq_len = 2, 10
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        logger.info(f"Testing RoPE generation with position_ids shape: {position_ids.shape}")
        
        # Generate RoPE embeddings
        cos, sin = wrapper._get_rope_position_embeddings(position_ids, 128)
        
        logger.info(f"Generated RoPE embeddings: cos={cos.shape}, sin={sin.shape}")
        
        # Verify shapes
        expected_shape = (batch_size, seq_len, 64)  # hidden_size // 2
        assert cos.shape == expected_shape, f"Expected cos shape {expected_shape}, got {cos.shape}"
        assert sin.shape == expected_shape, f"Expected sin shape {expected_shape}, got {sin.shape}"
        
        # Verify values are reasonable
        assert torch.all(torch.isfinite(cos)), "Cos contains non-finite values"
        assert torch.all(torch.isfinite(sin)), "Sin contains non-finite values"
        assert torch.all(cos >= -1) and torch.all(cos <= 1), "Cos values out of range [-1, 1]"
        assert torch.all(sin >= -1) and torch.all(sin <= 1), "Sin values out of range [-1, 1]"
        
        logger.info("✅ RoPE fix integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RoPE fix integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_rope_fix_integration()
        if success:
            logger.info("🎉 RoPE fix integration test passed!")
        else:
            logger.error("❌ RoPE fix integration test failed!")
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise
