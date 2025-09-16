#!/usr/bin/env python3
"""
Extract model weights from a PyTorch checkpoint and save as a simple state dict.

This script handles the custom class loading issues by using a more permissive approach.
"""

import os
import sys
import argparse
import torch
import pickle
from safetensors.torch import save_file

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_weights_from_checkpoint(input_path: str, output_path: str, model_key: str = "model_state_dict"):
    """Extract model weights from checkpoint and save as SafeTensors."""
    
    print(f"Extracting weights from: {input_path}")
    
    # Try to load the checkpoint with a more permissive approach
    try:
        # First, try to load with weights_only=False but ignore the config
        print("Attempting to load checkpoint...")
        
        # Load the raw file and extract just the tensors
        with open(input_path, 'rb') as f:
            # Use a custom unpickler that skips problematic objects
            class WeightExtractor(pickle.Unpickler):
                def __init__(self, file):
                    super().__init__(file)
                    self.weights = {}
                
                def find_class(self, module, name):
                    # For any custom class, return a dummy class that can store data
                    if module == '__main__' and name in ['QuadraticScheduleHBitLinearConfig', 'QuadraticScheduleBitNetConfig']:
                        class DummyConfig:
                            def __init__(self, *args, **kwargs):
                                pass
                        return DummyConfig
                    return super().find_class(module, name)
                
                def load(self):
                    # Override load to extract only tensor data
                    try:
                        obj = super().load()
                        if isinstance(obj, dict):
                            # Extract only tensor values
                            for key, value in obj.items():
                                if isinstance(value, torch.Tensor):
                                    self.weights[key] = value
                                elif isinstance(value, dict):
                                    # Recursively extract tensors from nested dicts
                                    for subkey, subvalue in value.items():
                                        if isinstance(subvalue, torch.Tensor):
                                            self.weights[f"{key}.{subkey}"] = subvalue
                        return obj
                    except Exception as e:
                        print(f"Error during load: {e}")
                        return self.weights
            
            unpickler = WeightExtractor(f)
            checkpoint = unpickler.load()
            
            # If we got weights from the custom unpickler, use those
            if hasattr(unpickler, 'weights') and unpickler.weights:
                state_dict = unpickler.weights
                print(f"Extracted {len(state_dict)} tensors using custom extractor")
            else:
                # Fallback to normal loading
                checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
                if model_key in checkpoint:
                    state_dict = checkpoint[model_key]
                else:
                    state_dict = checkpoint
                print(f"Loaded {len(state_dict)} parameters from checkpoint")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
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
    parser = argparse.ArgumentParser(description="Extract model weights and convert to SafeTensors")
    parser.add_argument("--input", required=True, help="Input .pt file path")
    parser.add_argument("--output", help="Output .safetensors file path")
    parser.add_argument("--model-key", default="model_state_dict", help="Key containing model weights")
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}.safetensors"
    
    print(f"Extracting weights: {input_path} -> {output_path}")
    
    success = extract_weights_from_checkpoint(input_path, output_path, args.model_key)
    
    if success:
        print("Weight extraction completed successfully!")
        print(f"SafeTensors file ready for upload: {output_path}")
    else:
        print("Weight extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
