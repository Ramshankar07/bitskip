"""
Boolean tensor fix wrapper for BitNet models
"""

import torch

# Store original model class
_original_model = None

def patch_bitnet_model():
    """Monkey patch BitNet model to fix boolean tensor issues."""
    global _original_model
    
    try:
        from bitnet.modeling import model as model_module
        
        if _original_model is None:
            _original_model = model_module.BitNetModel
        
        class PatchedBitNetModel(_original_model):
            def forward(self, *args, **kwargs):
                # Force config booleans
                if hasattr(self.config, 'use_layer_skipping'):
                    self.config.use_layer_skipping = False
                if hasattr(self.config, 'use_early_exit'):
                    self.config.use_early_exit = False
                
                # Call original forward
                return super().forward(*args, **kwargs)
        
        # Replace the model class
        model_module.BitNetModel = PatchedBitNetModel
        print("Applied BitNet model patch")
        
    except Exception as e:
        print(f"Warning: Could not patch BitNet model: {e}")

# Auto-patch on import
patch_bitnet_model()
