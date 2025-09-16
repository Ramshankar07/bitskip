#!/usr/bin/env python3
"""
Complete test script to verify LLaMA RoPE fix works end-to-end.
"""

import torch
import logging
from transformers import LlamaForCausalLM, LlamaConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llama_complete_fix():
    """Test that LLaMA model works with our complete RoPE fix."""
    logger.info("Testing complete LLaMA RoPE fix...")
    
    # Create a small LLaMA model for testing
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    
    model = LlamaForCausalLM(config)
    model.eval()
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Model config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,  # Let model generate position_ids
            )
        
        logger.info(f"Output logits shape: {outputs.logits.shape}")
        logger.info("✅ LLaMA forward pass successful!")
        
        # Test with explicit position_ids
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        with torch.no_grad():
            outputs2 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        logger.info(f"Output2 logits shape: {outputs2.logits.shape}")
        logger.info("✅ LLaMA with explicit position_ids successful!")
        
        # Test individual layer forward pass
        logger.info("Testing individual layer forward pass...")
        layer = model.model.layers[0]
        
        # Get embeddings
        inputs_embeds = model.model.embed_tokens(input_ids)
        
        # Test layer forward
        with torch.no_grad():
            layer_outputs = layer(
                inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
        
        logger.info(f"Layer output shape: {layer_outputs[0].shape}")
        logger.info("✅ Individual layer forward pass successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLaMA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_llama_complete_fix()
        if success:
            logger.info("🎉 Complete LLaMA RoPE fix test passed!")
        else:
            logger.error("❌ Complete LLaMA RoPE fix test failed!")
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise
