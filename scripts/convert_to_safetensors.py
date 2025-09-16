#!/usr/bin/env python3
"""
Convert PyTorch model checkpoint to SafeTensors format for Hugging Face Hub.

Usage:
    python scripts/convert_to_safetensors.py \
        --input "D:\\BitSkip\\quadratic_hbitlinear_final_model.pt" \
        --output "D:\\BitSkip\\quadratic_hbitlinear_final_model.safetensors"
"""

import os
import sys
import argparse
import torch
import pickle
from safetensors.torch import save_file

# Add the project root to Python path to import custom classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom classes that might be in the checkpoint
try:
    from bitnet.utils.default_config import DefaultConfig
    from bitnet.modeling.model import BitNetModel
    from bitnet.modeling.model2 import BitNetModel2
    print("Successfully imported custom classes")
except ImportError as e:
    print(f"Warning: Could not import custom classes: {e}")
    print("Will attempt to load checkpoint anyway...")


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that can handle missing classes by replacing them with a generic object."""
    
    def find_class(self, module, name):
        # Handle missing custom config classes
        if name in ['QuadraticScheduleHBitLinearConfig', 'QuadraticScheduleBitNetConfig']:
            print(f"Warning: Replacing missing class {name} with generic object")
            return object
        
        # Try to import the class normally
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load class {module}.{name}: {e}")
            print(f"Replacing with generic object")
            return object


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to SafeTensors")
    parser.add_argument("--input", required=True, help="Input .pt file path")
    parser.add_argument("--output", help="Output .safetensors file path (default: same as input with .safetensors extension)")
    parser.add_argument("--model-key", default="model_state_dict", help="Key in checkpoint containing model weights")
    return parser.parse_args()


def convert_checkpoint_to_safetensors(input_path: str, output_path: str, model_key: str = "model_state_dict"):
    """Convert PyTorch checkpoint to SafeTensors format."""
    
    print(f"Loading checkpoint from: {input_path}")
    
    # Load the checkpoint
    try:
        # First try with weights_only=True to avoid class loading issues
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"Error loading with weights_only=True: {e}")
        print("Trying with safe globals...")
        try:
            # Add the missing class to safe globals
            torch.serialization.add_safe_globals(['QuadraticScheduleHBitLinearConfig', 'QuadraticScheduleBitNetConfig'])
            checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        except Exception as e2:
            print(f"Error loading with safe globals: {e2}")
            print("Trying with weights_only=False...")
            try:
                checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
            except Exception as e3:
                print(f"Error loading checkpoint: {e3}")
                return False
    
    # Extract model state dict
    if model_key in checkpoint:
        state_dict = checkpoint[model_key]
        print(f"Found model state dict with {len(state_dict)} parameters")
    else:
        print(f"Warning: '{model_key}' not found in checkpoint. Using entire checkpoint as state dict.")
        state_dict = checkpoint
    
    # Convert to SafeTensors format
    print(f"Converting to SafeTensors format...")
    try:
        save_file(state_dict, output_path)
        print(f"Successfully saved to: {output_path}")
        
        # Verify the conversion
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        return True
    except Exception as e:
        print(f"Error saving SafeTensors file: {e}")
        return False


def main():
    args = parse_args()
    
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        # Replace .pt with .safetensors
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}.safetensors"
    
    print(f"Converting: {input_path} -> {output_path}")
    
    success = convert_checkpoint_to_safetensors(
        input_path, 
        output_path, 
        args.model_key
    )
    
    if success:
        print("Conversion completed successfully!")
        print(f"SafeTensors file ready for upload: {output_path}")
    else:
        print("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
