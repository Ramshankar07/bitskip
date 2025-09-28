#!/usr/bin/env python3
"""
Emergency NaN recovery script for BitNet model.
Run this when NaN appears in embeddings or early layers.
"""

import torch
import torch.nn as nn
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def diagnose_model_corruption(model):
    """
    Diagnose where NaN/Inf values are in the model.
    """
    print("\n" + "="*80)
    print("MODEL CORRUPTION DIAGNOSIS")
    print("="*80)
    
    corruption_report = {
        'embeddings': [],
        'parameters': [],
        'buffers': [],
        'critical': []
    }
    
    # Check embeddings specifically
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            weight = module.weight
            if torch.isnan(weight).any().item() or torch.isinf(weight).any().item():
                nan_count = torch.isnan(weight).sum().item()
                inf_count = torch.isinf(weight).sum().item()
                total_params = weight.numel()
                
                print(f"\n❌ CRITICAL: Embedding layer '{name}' is corrupted!")
                print(f"   - NaN values: {nan_count}/{total_params} ({100*nan_count/total_params:.1f}%)")
                print(f"   - Inf values: {inf_count}/{total_params} ({100*inf_count/total_params:.1f}%)")
                print(f"   - Shape: {weight.shape}")
                
                # Find which embedding indices are corrupted
                corrupted_indices = torch.where(torch.isnan(weight).any(dim=1) | torch.isinf(weight).any(dim=1))[0]
                if len(corrupted_indices) > 0:
                    print(f"   - Corrupted token indices: {corrupted_indices[:10].tolist()}...")
                
                corruption_report['critical'].append(name)
                corruption_report['embeddings'].append({
                    'name': name,
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'corrupted_indices': corrupted_indices.tolist()
                })
    
    # Check all parameters
    total_params_checked = 0
    total_nan_params = 0
    total_inf_params = 0
    
    for name, param in model.named_parameters():
        total_params_checked += param.numel()
        if torch.isnan(param).any().item():
            nan_count = torch.isnan(param).sum().item()
            total_nan_params += nan_count
            corruption_report['parameters'].append({
                'name': name,
                'nan_count': nan_count,
                'shape': list(param.shape)
            })
            if 'embed' not in name.lower():  # Non-embedding parameters
                print(f"⚠️  Parameter '{name}' has {nan_count} NaN values")
        
        if torch.isinf(param).any().item():
            inf_count = torch.isinf(param).sum().item()
            total_inf_params += inf_count
            if 'embed' not in name.lower():
                print(f"⚠️  Parameter '{name}' has {inf_count} Inf values")
    
    # Check buffers
    for name, buffer in model.named_buffers():
        if buffer.dtype in [torch.float16, torch.float32, torch.float64]:
            if torch.isnan(buffer).any().item() or torch.isinf(buffer).any().item():
                corruption_report['buffers'].append(name)
                print(f"⚠️  Buffer '{name}' is corrupted")
    
    print(f"\n📊 Summary:")
    print(f"   Total parameters: {total_params_checked:,}")
    print(f"   NaN parameters: {total_nan_params:,} ({100*total_nan_params/total_params_checked:.2f}%)")
    print(f"   Inf parameters: {total_inf_params:,} ({100*total_inf_params/total_params_checked:.2f}%)")
    print(f"   Corrupted embeddings: {len(corruption_report['embeddings'])}")
    print(f"   Corrupted buffers: {len(corruption_report['buffers'])}")
    
    return corruption_report


def emergency_fix_embeddings(model, vocab_size=None):
    """
    Emergency fix for corrupted embedding layers.
    """
    print("\n🚨 EMERGENCY EMBEDDING FIX")
    print("="*60)
    
    fixed_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            weight = module.weight
            
            # Check if embedding is corrupted
            has_nan = torch.isnan(weight).any().item()
            has_inf = torch.isinf(weight).any().item()
            
            if has_nan or has_inf:
                print(f"\n🔧 Fixing embedding layer '{name}'...")
                
                with torch.no_grad():
                    # Option 1: Try to preserve non-corrupted values
                    mask = torch.isnan(weight) | torch.isinf(weight)
                    corrupted_count = mask.sum().item()
                    
                    if corrupted_count < weight.numel() * 0.5:  # Less than 50% corrupted
                        # Replace only corrupted values
                        print(f"   Replacing {corrupted_count} corrupted values...")
                        
                        # Use mean of non-corrupted values
                        valid_values = weight[~mask]
                        if valid_values.numel() > 0:
                            mean_val = valid_values.mean().item()
                            std_val = valid_values.std().item()
                            
                            # Replace with random values from similar distribution
                            weight[mask] = torch.randn_like(weight[mask]) * std_val + mean_val
                        else:
                            # All values corrupted, reinitialize
                            nn.init.normal_(weight, mean=0.0, std=0.02)
                    else:
                        # Too many corrupted values, reinitialize entire embedding
                        print(f"   Reinitializing entire embedding (too corrupted)...")
                        nn.init.normal_(weight, mean=0.0, std=0.02)
                    
                    # Special handling for padding token (usually index 0)
                    if module.padding_idx is not None:
                        weight[module.padding_idx].zero_()
                    
                    # Ensure no extreme values
                    weight.clamp_(-1.0, 1.0)
                    
                fixed_count += 1
                print(f"   ✅ Fixed!")
    
    if fixed_count > 0:
        print(f"\n✨ Fixed {fixed_count} embedding layers")
    else:
        print("\n✓ No embedding fixes needed")
    
    return fixed_count


def emergency_fix_model(model, config=None):
    """
    Complete emergency fix for a corrupted model.
    """
    print("\n🚑 EMERGENCY MODEL RECOVERY")
    print("="*80)
    
    fixes_applied = []
    
    with torch.no_grad():
        # Step 1: Fix embeddings first (most critical)
        embed_fixes = emergency_fix_embeddings(model)
        if embed_fixes > 0:
            fixes_applied.append(f"Fixed {embed_fixes} embeddings")
        
        # Step 2: Fix position embeddings
        for name, module in model.named_modules():
            if 'position' in name.lower() and isinstance(module, nn.Embedding):
                if torch.isnan(module.weight).any().item() or torch.isinf(module.weight).any().item():
                    print(f"🔧 Fixing position embeddings '{name}'...")
                    # Reinitialize with sinusoidal pattern or random
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    fixes_applied.append(f"Fixed position embeddings {name}")
        
        # Step 3: Fix linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight
                    if torch.isnan(weight).any().item() or torch.isinf(weight).any().item():
                        print(f"🔧 Fixing linear layer '{name}'...")
                        nn.init.xavier_uniform_(weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                        fixes_applied.append(f"Fixed linear layer {name}")
        
        # Step 4: Fix LayerNorm parameters
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                if torch.isnan(module.weight).any().item() or torch.isinf(module.weight).any().item():
                    print(f"🔧 Fixing LayerNorm '{name}'...")
                    module.weight.fill_(1.0)
                    module.bias.zero_()
                    fixes_applied.append(f"Fixed LayerNorm {name}")
        
        # Step 5: Fix buffers (especially weight_scale in BitLinear)
        for name, buffer in model.named_buffers():
            if buffer.dtype in [torch.float16, torch.float32, torch.float64]:
                if torch.isnan(buffer).any().item() or torch.isinf(buffer).any().item():
                    print(f"🔧 Fixing buffer '{name}'...")
                    if 'weight_scale' in name:
                        buffer.fill_(1.0)
                    elif 'running_mean' in name:
                        buffer.zero_()
                    elif 'running_var' in name:
                        buffer.fill_(1.0)
                    else:
                        buffer.zero_()
                    fixes_applied.append(f"Fixed buffer {name}")
    
    print(f"\n✅ Applied {len(fixes_applied)} fixes")
    for fix in fixes_applied[:10]:  # Show first 10 fixes
        print(f"   - {fix}")
    
    return fixes_applied


def verify_model_health(model):
    """
    Verify the model is healthy after fixes.
    """
    print("\n🔍 VERIFYING MODEL HEALTH")
    print("="*60)
    
    is_healthy = True
    
    # Check for NaN/Inf
    for name, param in model.named_parameters():
        if torch.isnan(param).any().item() or torch.isinf(param).any().item():
            print(f"❌ Still has NaN/Inf in {name}")
            is_healthy = False
    
    # Test forward pass with dummy input
    print("\n🧪 Testing forward pass...")
    model.eval()
    
    try:
        with torch.no_grad():
            # Create small dummy input
            dummy_input = torch.randint(0, 100, (1, 10)).to(next(model.parameters()).device)
            dummy_attention = torch.ones_like(dummy_input)
            
            # Try forward pass
            output = model(
                input_ids=dummy_input,
                attention_mask=dummy_attention
            )
            
            if hasattr(output, 'logits'):
                logits = output.logits
                if torch.isnan(logits).any().item() or torch.isinf(logits).any().item():
                    print("❌ Forward pass produces NaN/Inf in logits")
                    is_healthy = False
                else:
                    print("✅ Forward pass successful")
            else:
                print("✅ Forward pass completed")
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        is_healthy = False
    
    model.train()
    
    if is_healthy:
        print("\n✅ Model is healthy!")
    else:
        print("\n⚠️ Model still has issues, may need full reinitialization")
    
    return is_healthy


def reset_optimizer_and_scaler(model, learning_rate=5e-5):
    """
    Create fresh optimizer and gradient scaler after fixing model.
    """
    print("\n🔄 Creating fresh optimizer and scaler...")
    
    # Create new optimizer with lower learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate * 0.1,  # Start with 10% of original LR
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    # Conservative gradient scaler
    grad_scaler = torch.amp.GradScaler(
        'cuda',
        init_scale=2**8,  # Much lower than default
        growth_factor=1.2,
        backoff_factor=0.25,
        growth_interval=200,
        enabled=True
    )
    
    print(f"✅ New optimizer with LR={learning_rate * 0.1:.2e}")
    print(f"✅ New gradient scaler with init_scale=256")
    
    return optimizer, grad_scaler


def save_emergency_checkpoint(model, path="emergency_checkpoint.pt"):
    """
    Save emergency checkpoint after fixing.
    """
    print(f"\n💾 Saving emergency checkpoint to {path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'fixed': True,
        'timestamp': torch.tensor([0])  # Placeholder
    }, path)
    print("✅ Checkpoint saved")


# MAIN RECOVERY FUNCTION
def recover_from_nan(model, optimizer=None, grad_scaler=None, config=None):
    """
    Complete recovery procedure from NaN corruption.
    
    Usage:
        model, optimizer, grad_scaler = recover_from_nan(model, optimizer, grad_scaler, config)
    """
    print("\n" + "="*80)
    print("🚨 EMERGENCY NaN RECOVERY PROCEDURE 🚨")
    print("="*80)
    
    # Step 1: Diagnose
    corruption = diagnose_model_corruption(model)
    
    # Step 2: Fix model
    fixes = emergency_fix_model(model, config)
    
    # Step 3: Verify
    is_healthy = verify_model_health(model)
    
    if not is_healthy:
        print("\n⚠️ Model couldn't be fully recovered. Attempting deeper fix...")
        
        # Try more aggressive fixes
        for name, module in model.named_modules():
            if hasattr(module, 'reset_parameters'):
                try:
                    module.reset_parameters()
                    print(f"Reset {name}")
                except:
                    pass
        
        # Verify again
        is_healthy = verify_model_health(model)
    
    # Step 4: Reset optimizer and scaler
    if optimizer is not None and grad_scaler is not None:
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer, grad_scaler = reset_optimizer_and_scaler(model, learning_rate)
    
    # Step 5: Save checkpoint
    save_emergency_checkpoint(model, "emergency_recovery.pt")
    
    print("\n" + "="*80)
    if is_healthy:
        print("✅ RECOVERY SUCCESSFUL - You can resume training")
        print("⚠️ Recommend using lower learning rate and watching for instability")
    else:
        print("❌ RECOVERY FAILED - Model needs reinitialization")
        print("💡 Try loading a previous checkpoint or starting fresh")
    print("="*80)
    
    return model, optimizer, grad_scaler


# INTEGRATION WITH TRAINING LOOP
def integrate_with_training_loop():
    """
    Show how to integrate with training loop.
    """
    example = """
# In your training loop, add this check:

if torch.isnan(loss).item() or loss.item() > 100:
    logger.error("NaN or extreme loss detected, attempting recovery...")
    
    # Run emergency recovery
    model, optimizer, grad_scaler = recover_from_nan(
        model, optimizer, grad_scaler, config
    )
    
    # Skip this iteration and continue
    continue

# Or check periodically:
if step % 100 == 0:
    # Quick health check
    for name, param in model.named_parameters():
        if torch.isnan(param).any().item():
            logger.error(f"NaN detected in {name}, running recovery...")
            model, optimizer, grad_scaler = recover_from_nan(
                model, optimizer, grad_scaler, config
            )
            break
"""
    print(example)


if __name__ == "__main__":
    print("Emergency NaN Recovery Script")
    print("This script should be imported and used in your training loop")
    print("\nExample usage:")
    print("-"*60)
    integrate_with_training_loop()
