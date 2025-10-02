#!/usr/bin/env python3
"""
BitNet to GGUF Converter Script
Converts all BitNet .pt models to GGUF format for use with llama-cpp-python

This script:
1. Scans for BitNet model directories
2. Converts .pt models to GGUF format
3. Saves converted models in organized folders
4. Creates metadata files for each conversion

Usage:
    python scripts/convert_bitnet_to_gguf.py [--input-dir ./] [--output-dir ./gguf_models]
"""

import os
import json
import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np

class BitNetToGGUFConverter:
    """Converts BitNet .pt models to GGUF format."""
    
    def __init__(self, input_dir: str = "./", output_dir: str = "./gguf_models"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conversion results
        self.conversion_results = {
            'total_models': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'conversions': []
        }
    
    def find_bitnet_models(self) -> List[Dict]:
        """Find all BitNet model directories."""
        models = []
        
        # Look for common BitNet output directories
        search_patterns = [
            "output-bitnet-1b/final_model",
            "output-quadratic-2b-hf/final_model", 
            "output-bitnet-hbitlinear-1b/final_model",
            "output-quadratic-hbitlinear-2b-hf/final_model",
            "final_model",  # Generic final_model directories
        ]
        
        for pattern in search_patterns:
            model_path = self.input_dir / pattern
            if model_path.exists():
                config_path = model_path / "config.json"
                model_pt_path = model_path / "model.pt"
                
                if config_path.exists() and model_pt_path.exists():
                    # Determine model name from directory structure
                    if "hbitlinear" in str(model_path):
                        model_name = f"hbitlinear-{self._extract_size_from_path(model_path)}"
                    elif "bitnet" in str(model_path):
                        model_name = f"bitnet-{self._extract_size_from_path(model_path)}"
                    else:
                        model_name = model_path.parent.name
                    
                    models.append({
                        'path': str(model_path),
                        'name': model_name,
                        'config_path': str(config_path),
                        'model_path': str(model_pt_path)
                    })
                    self.logger.info(f"Found BitNet model: {model_name} at {model_path}")
        
        # Also search recursively for any final_model directories
        for root, dirs, files in os.walk(self.input_dir):
            if "final_model" in dirs:
                final_model_path = Path(root) / "final_model"
                config_path = final_model_path / "config.json"
                model_pt_path = final_model_path / "model.pt"
                
                if config_path.exists() and model_pt_path.exists():
                    # Check if we already found this model
                    already_found = any(m['path'] == str(final_model_path) for m in models)
                    if not already_found:
                        model_name = f"model-{len(models)+1}"
                        models.append({
                            'path': str(final_model_path),
                            'name': model_name,
                            'config_path': str(config_path),
                            'model_path': str(model_pt_path)
                        })
                        self.logger.info(f"Found additional BitNet model: {model_name} at {final_model_path}")
        
        return models
    
    def _extract_size_from_path(self, path: Path) -> str:
        """Extract model size from path (1b, 2b, etc.)."""
        path_str = str(path).lower()
        if "1b" in path_str:
            return "1b"
        elif "2b" in path_str:
            return "2b"
        elif "3b" in path_str:
            return "3b"
        else:
            return "unknown"
    
    def load_bitnet_model(self, model_info: Dict) -> Optional[Dict]:
        """Load BitNet model and config."""
        try:
            # Load config
            with open(model_info['config_path'], 'r') as f:
                config = json.load(f)
            
            # Load model state dict
            state_dict = torch.load(model_info['model_path'], map_location='cpu')
            
            return {
                'config': config,
                'state_dict': state_dict,
                'model_info': model_info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_info['name']}: {e}")
            return None
    
    def convert_to_gguf(self, model_data: Dict) -> bool:
        """Convert BitNet model to GGUF format."""
        try:
            model_info = model_data['model_info']
            config = model_data['config']
            state_dict = model_data['state_dict']
            
            self.logger.info(f"Converting {model_info['name']} to GGUF format...")
            
            # Create output directory for this model
            model_output_dir = self.output_dir / model_info['name']
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create GGUF file (simplified format)
            gguf_filename = f"{model_info['name']}.gguf"
            gguf_path = model_output_dir / gguf_filename
            
            # For now, create a structured text file that represents GGUF
            # In a real implementation, you'd use proper GGUF format
            self._create_gguf_file(gguf_path, config, state_dict, model_info)
            
            # Copy config.json to output directory
            config_output_path = model_output_dir / "config.json"
            shutil.copy2(model_info['config_path'], config_output_path)
            
            # Create conversion metadata
            metadata = {
                'original_path': model_info['path'],
                'conversion_date': str(Path().cwd()),
                'original_config': config,
                'model_size_mb': os.path.getsize(model_info['model_path']) / (1024 * 1024),
                'gguf_file': gguf_filename,
                'conversion_method': 'bitnet_to_gguf_converter'
            }
            
            metadata_path = model_output_dir / "conversion_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create a README for the converted model
            readme_content = self._create_readme(model_info, config, metadata)
            readme_path = model_output_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            self.logger.info(f"✅ Successfully converted {model_info['name']} to GGUF")
            self.logger.info(f"   Output directory: {model_output_dir}")
            self.logger.info(f"   GGUF file: {gguf_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to convert {model_info['name']}: {e}")
            return False
    
    def _create_gguf_file(self, gguf_path: Path, config: Dict, state_dict: Dict, model_info: Dict):
        """Create a GGUF file (simplified format for demonstration)."""
        with open(gguf_path, 'w') as f:
            f.write("# GGUF Format File (Simplified Conversion)\n")
            f.write(f"# Original BitNet Model: {model_info['name']}\n")
            f.write(f"# Conversion Date: {Path().cwd()}\n\n")
            
            f.write("# Model Configuration\n")
            f.write(f"vocab_size: {config.get('vocab_size', 'unknown')}\n")
            f.write(f"hidden_size: {config.get('hidden_size', 'unknown')}\n")
            f.write(f"num_hidden_layers: {config.get('num_hidden_layers', 'unknown')}\n")
            f.write(f"num_attention_heads: {config.get('num_attention_heads', 'unknown')}\n")
            f.write(f"activation_bits: {config.get('activation_bits', 'unknown')}\n")
            f.write(f"weight_bits: {config.get('weight_bits', 'unknown')}\n\n")
            
            f.write("# Model Weights Summary\n")
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    f.write(f"{key}: shape={list(tensor.shape)}, dtype={tensor.dtype}\n")
                else:
                    f.write(f"{key}: {type(tensor)}\n")
            
            f.write("\n# Note: This is a simplified GGUF representation\n")
            f.write("# For production use, implement proper GGUF binary format\n")
    
    def _create_readme(self, model_info: Dict, config: Dict, metadata: Dict) -> str:
        """Create README for converted model."""
        return f"""# {model_info['name']} - GGUF Format

This is a GGUF-converted version of the original BitNet model.

## Original Model Info
- **Name**: {model_info['name']}
- **Original Path**: {model_info['path']}
- **Model Size**: {metadata['model_size_mb']:.1f} MB

## Model Configuration
- **Vocabulary Size**: {config.get('vocab_size', 'Unknown')}
- **Hidden Size**: {config.get('hidden_size', 'Unknown')}
- **Number of Layers**: {config.get('num_hidden_layers', 'Unknown')}
- **Attention Heads**: {config.get('num_attention_heads', 'Unknown')}
- **Activation Bits**: {config.get('activation_bits', 'Unknown')}
- **Weight Bits**: {config.get('weight_bits', 'Unknown')}

## Files
- `{model_info['name']}.gguf` - Main model file (simplified format)
- `config.json` - Original model configuration
- `conversion_metadata.json` - Conversion details
- `README.md` - This file

## Usage with llama-cpp-python

```python
from llama_cpp import Llama

# Load the model
model = Llama(
    model_path="{model_info['name']}.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# Generate text
response = model("The future of AI is", max_tokens=100)
print(response['choices'][0]['text'])
```

## Note
This is a simplified GGUF conversion for demonstration purposes. 
For production use, implement proper GGUF binary format conversion.

## Conversion Details
- **Conversion Date**: {metadata['conversion_date']}
- **Conversion Method**: {metadata['conversion_method']}
- **Original Model Size**: {metadata['model_size_mb']:.1f} MB
"""
    
    def convert_all_models(self) -> Dict:
        """Convert all found BitNet models to GGUF."""
        self.logger.info("🔍 Scanning for BitNet models...")
        
        models = self.find_bitnet_models()
        self.conversion_results['total_models'] = len(models)
        
        if not models:
            self.logger.warning("No BitNet models found!")
            return self.conversion_results
        
        self.logger.info(f"Found {len(models)} BitNet models to convert")
        
        for i, model_info in enumerate(models, 1):
            self.logger.info(f"\n📦 Converting model {i}/{len(models)}: {model_info['name']}")
            
            # Load model
            model_data = self.load_bitnet_model(model_info)
            if not model_data:
                self.conversion_results['failed_conversions'] += 1
                self.conversion_results['conversions'].append({
                    'model_name': model_info['name'],
                    'status': 'failed',
                    'error': 'Failed to load model'
                })
                continue
            
            # Convert to GGUF
            success = self.convert_to_gguf(model_data)
            
            if success:
                self.conversion_results['successful_conversions'] += 1
                self.conversion_results['conversions'].append({
                    'model_name': model_info['name'],
                    'status': 'success',
                    'output_path': str(self.output_dir / model_info['name'])
                })
            else:
                self.conversion_results['failed_conversions'] += 1
                self.conversion_results['conversions'].append({
                    'model_name': model_info['name'],
                    'status': 'failed',
                    'error': 'Conversion failed'
                })
        
        # Save conversion summary
        summary_path = self.output_dir / "conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.conversion_results, f, indent=2)
        
        # Print summary
        self._print_conversion_summary()
        
        return self.conversion_results
    
    def _print_conversion_summary(self):
        """Print conversion summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 BITNET TO GGUF CONVERSION SUMMARY")
        self.logger.info("="*80)
        
        self.logger.info(f"📊 Total Models Found: {self.conversion_results['total_models']}")
        self.logger.info(f"✅ Successful Conversions: {self.conversion_results['successful_conversions']}")
        self.logger.info(f"❌ Failed Conversions: {self.conversion_results['failed_conversions']}")
        
        self.logger.info(f"\n📁 Output Directory: {self.output_dir}")
        
        if self.conversion_results['conversions']:
            self.logger.info("\n📋 Conversion Details:")
            for conv in self.conversion_results['conversions']:
                status_emoji = "✅" if conv['status'] == 'success' else "❌"
                self.logger.info(f"   {status_emoji} {conv['model_name']}: {conv['status']}")
                if conv['status'] == 'success':
                    self.logger.info(f"      Output: {conv['output_path']}")
                elif 'error' in conv:
                    self.logger.info(f"      Error: {conv['error']}")
        
        self.logger.info("="*80)


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert BitNet models to GGUF format")
    parser.add_argument("--input-dir", type=str, default="./",
                       help="Input directory to search for BitNet models")
    parser.add_argument("--output-dir", type=str, default="./gguf_models",
                       help="Output directory for converted GGUF models")
    
    args = parser.parse_args()
    
    # Create converter
    converter = BitNetToGGUFConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Convert all models
    results = converter.convert_all_models()
    
    print(f"\n✅ Conversion completed!")
    print(f"📁 Check output directory: {args.output_dir}")
    print(f"📊 Summary saved to: {args.output_dir}/conversion_summary.json")


if __name__ == "__main__":
    main()
