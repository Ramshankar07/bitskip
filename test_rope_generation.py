#!/usr/bin/env python3
"""
Test script to verify RoPE generation works correctly.
"""

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_rope_generation():
    """Test RoPE generation with different input shapes."""
    logger.info("Testing RoPE generation...")
    
    # Test parameters
    batch_size, seq_len = 2, 10
    hidden_size = 1024
    rope_theta = 10000.0
    
    # Create position IDs
    position_ids = torch.arange(seq_len, device='cpu').unsqueeze(0).expand(batch_size, -1)
    logger.info(f"Position IDs shape: {position_ids.shape}")
    
    try:
        # Generate inv_freq for RoPE
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size))
        logger.info(f"inv_freq shape: {inv_freq.shape}")
        
        # Get position values - flatten to 1D for torch.outer
        t = position_ids.float().flatten()  # Shape: (batch_size * seq_len,)
        logger.info(f"t shape after flatten: {t.shape}")
        
        # Compute frequencies using outer product
        freqs = torch.outer(t, inv_freq)  # Shape: (batch_size * seq_len, hidden_size // 2)
        logger.info(f"freqs shape after outer: {freqs.shape}")
        
        # Reshape back to original batch and sequence dimensions
        freqs = freqs.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, hidden_size // 2)
        logger.info(f"freqs shape after reshape: {freqs.shape}")
        
        # Generate cos and sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        logger.info(f"Final cos shape: {cos.shape}")
        logger.info(f"Final sin shape: {sin.shape}")
        
        # Verify shapes
        expected_shape = (batch_size, seq_len, hidden_size // 2)
        assert cos.shape == expected_shape, f"Expected cos shape {expected_shape}, got {cos.shape}"
        assert sin.shape == expected_shape, f"Expected sin shape {expected_shape}, got {sin.shape}"
        
        # Verify values are reasonable
        assert torch.all(torch.isfinite(cos)), "Cos contains non-finite values"
        assert torch.all(torch.isfinite(sin)), "Sin contains non-finite values"
        assert torch.all(cos >= -1) and torch.all(cos <= 1), "Cos values out of range [-1, 1]"
        assert torch.all(sin >= -1) and torch.all(sin <= 1), "Sin values out of range [-1, 1]"
        
        logger.info("✅ RoPE generation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RoPE generation test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = test_rope_generation()
        if success:
            logger.info("🎉 All tests passed!")
        else:
            logger.error("❌ Tests failed!")
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        raise
