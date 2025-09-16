#!/usr/bin/env python3
"""
Test script to verify RoPE functionality in the modified wrapper.
"""

import torch
import logging
from train_sft_with_layer_skip import BitNetLayerSkipModel, BitNetLayerSkipConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rope_functionality():
    """Test RoPE position embedding generation."""
    logger.info("Testing RoPE functionality...")
    
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
                'hidden_size': 1024,
                'vocab_size': 32000,
                'rope_theta': 10000.0,
                'max_position_embeddings': 2048,
                'rope_scaling': None
            })()
        
        def get_input_embeddings(self):
            return torch.nn.Embedding(32000, 1024)
        
        def get_output_embeddings(self):
            return torch.nn.Linear(1024, 32000)
    
    # Create wrapper
    base_model = DummyModel()
    wrapper = BitNetLayerSkipModel(base_model, config)
    
    # Test RoPE generation
    batch_size, seq_len = 2, 10
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    logger.info(f"Position IDs shape: {position_ids.shape}")
    
    # Generate RoPE embeddings
    cos, sin = wrapper._get_rope_position_embeddings(position_ids, 1024)
    
    logger.info(f"Cos shape: {cos.shape}")
    logger.info(f"Sin shape: {sin.shape}")
    logger.info(f"Cos dtype: {cos.dtype}")
    logger.info(f"Sin dtype: {sin.dtype}")
    
    # Verify shapes
    assert cos.shape == (batch_size, seq_len, 512), f"Expected cos shape (2, 10, 512), got {cos.shape}"
    assert sin.shape == (batch_size, seq_len, 512), f"Expected sin shape (2, 10, 512), got {sin.shape}"
    
    # Verify values are reasonable
    assert torch.all(torch.isfinite(cos)), "Cos contains non-finite values"
    assert torch.all(torch.isfinite(sin)), "Sin contains non-finite values"
    assert torch.all(cos >= -1) and torch.all(cos <= 1), "Cos values out of range [-1, 1]"
    assert torch.all(sin >= -1) and torch.all(sin <= 1), "Sin values out of range [-1, 1]"
    
    logger.info("✅ RoPE functionality test passed!")
    
    # Test caching
    cos2, sin2 = wrapper._get_rope_position_embeddings(position_ids, 1024)
    assert torch.allclose(cos, cos2), "RoPE caching not working"
    assert torch.allclose(sin, sin2), "RoPE caching not working"
    
    logger.info("✅ RoPE caching test passed!")
    
    return True

if __name__ == "__main__":
    try:
        test_rope_functionality()
        logger.info("🎉 All tests passed!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
