#!/usr/bin/env python3
"""
Manual Boolean Tensor Debugging Checklist
"""

# =============================================================================
# BOOLEAN TENSOR DEBUGGING CHECKLIST
# =============================================================================

"""
PROBLEMATIC PATTERNS TO FIND AND FIX:

1. DIRECT TENSOR BOOLEAN CHECKS:
   ❌ if tensor.any():           → ✅ if tensor.any().item():
   ❌ if tensor.all():           → ✅ if tensor.all().item():
   ❌ if not tensor[idx]:        → ✅ if not tensor[idx].item():
   ❌ if tensor:                 → ✅ if tensor.numel() > 0:

2. TRAINING MODE CHECKS:
   ❌ if self.training:          → ✅ if bool(self.training):
   ❌ if module.training:        → ✅ if bool(module.training):

3. CONFIG BOOLEAN CHECKS:
   ❌ if self.config.use_xxx:    → ✅ if bool(self.config.use_xxx):
   ❌ if config.flag:            → ✅ if bool(config.flag):

4. TENSOR COMPARISON CHECKS:
   ❌ if (tensor == value).any(): → ✅ if (tensor == value).any().item():
   ❌ if torch.isnan(tensor):    → ✅ if torch.isnan(tensor).any():
   ❌ if torch.isinf(tensor):    → ✅ if torch.isinf(tensor).any():

5. LIST/TUPLE CHECKS:
   ❌ if tensor_list:            → ✅ if len(tensor_list) > 0:
   ❌ if accepted_tokens:       → ✅ if len(accepted_tokens) > 0:

FILES TO CHECK MANUALLY:
=======================

1. bitnet/modeling/model.py
   - Line 54: curriculum_mask[layer_idx].item() ✅
   - Line 59: active_mask.any().item() ✅
   - Line 67: torch.isnan(logits).any() ✅
   - Line 77: torch.isnan(loss).any() ✅
   - Line 280: getattr(self, 'training', False) is True ✅
   - Line 415: getattr(self, 'training', False) is True ✅
   - Line 457: getattr(self, 'training', False) is True ✅
   - Line 697: accept.any().item() ✅
   - Line 713: (generated == eos_token_id).any().item() ✅

2. bitnet/modeling/transformer.py
   - Line 122: torch.isnan(attn_output).any() ✅
   - Line 131: torch.isnan(hidden_states).any() ✅
   - Line 140: torch.isnan(ff_output).any() ✅
   - Line 149: torch.isnan(hidden_states).any() ✅

3. bitnet/modeling/bitlinear.py
   - Line 131: bool(self.training) ✅
   - Line 143: torch.isnan(output).any() ✅

4. bitnet/modeling/h_bitlinear.py
   - Line 169: bool(self.training) ✅

5. bitnet/modeling/layer_skipping.py
   - Line 162: bool(self.training) ✅
   - Line 169: skip_mask.any().item() ✅

MANUAL DEBUGGING STEPS:
======================

STEP 1: Add Debug Prints
------------------------
Add these debug prints to identify the exact location:

```python
# In model.py forward method, add at the beginning:
print(f"DEBUG: Model training mode: {self.training}")
print(f"DEBUG: Model training type: {type(self.training)}")

# In each layer loop:
print(f"DEBUG: Processing layer {layer_idx}")
print(f"DEBUG: Layer training mode: {self.layers[layer_idx].training}")
print(f"DEBUG: Layer training type: {type(self.layers[layer_idx].training)}")

# In transformer block:
print(f"DEBUG: Transformer training mode: {self.training}")
print(f"DEBUG: Attention training mode: {self.self_attn.training}")

# In BitLinear forward:
print(f"DEBUG: BitLinear training mode: {self.training}")
print(f"DEBUG: BitLinear training type: {type(self.training)}")
```

STEP 2: Test Each Component Separately
-------------------------------------
Create individual tests for each component:

1. Test BitLinear alone
2. Test GQA attention alone  
3. Test transformer block alone
4. Test full model

STEP 3: Check Training Mode Propagation
---------------------------------------
Ensure training mode is properly propagated:

```python
# Check if training mode is being set correctly
model.train()
print(f"Model training: {model.training}")
for i, layer in enumerate(model.layers):
    print(f"Layer {i} training: {layer.training}")
    print(f"Layer {i} attention training: {layer.self_attn.training}")
```

STEP 4: Fix Remaining Issues
-----------------------------
Look for these specific patterns that might still be problematic:

1. Any remaining `if tensor:` without `.item()`
2. Any remaining `if self.training:` without `bool()`
3. Any remaining `if config.flag:` without `bool()`
4. Any tensor comparisons in if statements

STEP 5: Test Incrementally
--------------------------
Test after each fix:
1. Test eval mode first
2. Test training mode
3. Test backward pass
4. Test full training loop

COMMON ISSUES TO LOOK FOR:
=========================

1. Module training state not being set properly
2. Config values being tensors instead of booleans
3. Tensor operations returning tensors instead of scalars
4. Missing .item() calls on tensor operations
5. Boolean operations on multi-element tensors

QUICK FIXES TO APPLY:
====================

1. Replace all `if tensor:` with `if tensor.numel() > 0:`
2. Replace all `if self.training:` with `if bool(self.training):`
3. Replace all `if config.flag:` with `if bool(config.flag):`
4. Replace all `if tensor.any():` with `if tensor.any().item():`
5. Replace all `if torch.isnan(tensor):` with `if torch.isnan(tensor).any():`

TESTING COMMAND:
===============
python -c "
import torch
from bitnet.modeling.model import BitNetModel
from bitnet.utils.default_config import DefaultConfig

config = DefaultConfig(
    vocab_size=1000, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=8, num_kv_heads=2, use_layer_skipping=False,
    use_early_exit=False, gradient_checkpointing=False
)
model = BitNetModel(config)
model.train()

input_ids = torch.randint(0, 1000, (2, 10))
labels = input_ids.clone()

try:
    outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
    print('SUCCESS: Forward pass worked')
    outputs.loss.backward()
    print('SUCCESS: Backward pass worked')
except Exception as e:
    print(f'ERROR: {e}')
    print(f'Type: {type(e)}')
"
"""

print("Boolean Tensor Debugging Checklist Created!")
print("Follow the steps above to manually debug the boolean tensor issues.")
print("The checklist shows all files and patterns to check.")
