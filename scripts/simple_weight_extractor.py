#!/usr/bin/env python3
"""
Simple script to extract model weights from checkpoint using the same environment.
This script should be run from the project root directory.
"""

import os
import sys
import torch
from safetensors.torch import save_file

# Add current directory to path so we can import the training modules
sys.path.insert(0, os.getcwd())

def main():
    checkpoint_path = "quadratic_hbitlinear_final_model.pt"
    output_path = "quadratic_hbitlinear_final_model.safetensors"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load the checkpoint - this should work since we're in the same environment
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Found model state dict with {len(state_dict)} parameters")
        else:
            print("No model_state_dict found, using entire checkpoint")
            state_dict = checkpoint
        
        # Convert to SafeTensors
        print(f"Converting to SafeTensors: {output_path}")
        save_file(state_dict, output_path)
        
        # Verify
        file_size = os.path.getsize(output_path)
        print(f"Success! File size: {file_size / (1024*1024):.2f} MB")
        print(f"SafeTensors file saved: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
