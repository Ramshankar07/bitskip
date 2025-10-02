#!/usr/bin/env python3
"""
BitNet to SafeTensors Converter Script
Converts all BitNet .pt models to SafeTensors format for use with HuggingFace transformers

This script:
1. Scans for BitNet model directories
2. Converts .pt models to SafeTensors format
3. Saves converted models in organized folders
4. Creates metadata files for each conversion

Usage:
    python scripts/convert_bitnet_to_safetensors.py [--input-dir ./] [--output-dir ./safetensors_models]
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

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class BitNetToSafeTensorsConverter:
    """Converts BitNet .pt models to SafeTensors format."""
    
    def __init__(self, input_dir: str = "./", output_dir: str = "./safetensors_models"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Check SafeTensors availability
        if not SAFETENSORS_AVAILABLE:
            self.logger.error("SafeTensors not available. Install with: pip install safetensors")
            raise ImportError("SafeTensors library not found")
        
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
        
        # Check if input_dir is a specific model directory
        if (self.input_dir / "config.json").exists() and (self.input_dir / "model.pt").exists():
            # This is a specific model directory
            model_name = self.input_dir.name
            if "final_model" in str(self.input_dir):
                # Extract parent directory name for model name
                parent_name = self.input_dir.parent.name
                if "hbitlinear" in parent_name:
                    model_name = f"hbitlinear-{self._extract_size_from_path(self.input_dir.parent)}"
                elif "bitnet" in parent_name:
                    model_name = f"bitnet-{self._extract_size_from_path(self.input_dir.parent)}"
                else:
                    model_name = parent_name
            
            models.append({
                'path': str(self.input_dir),
                'name': model_name,
                'config_path': str(self.input_dir / "config.json"),
                'model_path': str(self.input_dir / "model.pt")
            })
            self.logger.info(f"Found specific BitNet model: {model_name} at {self.input_dir}")
            return models
        
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
    
    def convert_to_safetensors(self, model_data: Dict) -> bool:
        """Convert BitNet model to SafeTensors format."""
        try:
            model_info = model_data['model_info']
            config = model_data['config']
            state_dict = model_data['state_dict']
            
            self.logger.info(f"Converting {model_info['name']} to SafeTensors format...")
            
            # Create output directory for this model
            model_output_dir = self.output_dir / model_info['name']
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create SafeTensors file
            safetensors_filename = f"{model_info['name']}.safetensors"
            safetensors_path = model_output_dir / safetensors_filename
            
            # Convert state dict to SafeTensors format
            self.logger.info(f"Saving model weights to SafeTensors format...")
            save_file(state_dict, str(safetensors_path))
            
            # Copy config.json to output directory
            config_output_path = model_output_dir / "config.json"
            shutil.copy2(model_info['config_path'], config_output_path)
            
            # Create conversion metadata
            metadata = {
                'original_path': model_info['path'],
                'conversion_date': str(Path().cwd()),
                'original_config': config,
                'model_size_mb': os.path.getsize(model_info['model_path']) / (1024 * 1024),
                'safetensors_file': safetensors_filename,
                'conversion_method': 'bitnet_to_safetensors_converter',
                'format': 'safetensors'
            }
            
            metadata_path = model_output_dir / "conversion_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create a README for the converted model
            readme_content = self._create_readme(model_info, config, metadata)
            readme_path = model_output_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            self.logger.info(f"✅ Successfully converted {model_info['name']} to SafeTensors")
            self.logger.info(f"   Output directory: {model_output_dir}")
            self.logger.info(f"   SafeTensors file: {safetensors_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to convert {model_info['name']}: {e}")
            return False
    
    
    def _create_readme(self, model_info: Dict, config: Dict, metadata: Dict) -> str:
        """Create README for converted model."""
        return f"""# {model_info['name']} - SafeTensors Format

This is a SafeTensors-converted version of the original BitNet model.

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
- `{model_info['name']}.safetensors` - Main model file (SafeTensors format)
- `config.json` - Original model configuration
- `conversion_metadata.json` - Conversion details
- `README.md` - This file

## Usage with HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(".")

# Load model weights from SafeTensors
state_dict = load_file("{model_info['name']}.safetensors")

# Create model instance and load weights
model = AutoModelForCausalLM.from_pretrained(".", state_dict=state_dict)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Usage with SafeTensors directly

```python
from safetensors.torch import load_file

# Load model weights
state_dict = load_file("{model_info['name']}.safetensors")

# Access individual tensors
for key, tensor in state_dict.items():
    print(f"{{key}}: {{tensor.shape}} - {{tensor.dtype}}")
```

## Benefits of SafeTensors
- **Memory Safety**: Prevents arbitrary code execution
- **Cross-platform**: Works across different architectures
- **Fast Loading**: Optimized for quick model loading
- **HuggingFace Compatible**: Native support in transformers library

## Conversion Details
- **Conversion Date**: {metadata['conversion_date']}
- **Conversion Method**: {metadata['conversion_method']}
- **Original Model Size**: {metadata['model_size_mb']:.1f} MB
- **Format**: SafeTensors
"""
    
    def convert_all_models(self) -> Dict:
        """Convert all found BitNet models to SafeTensors."""
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
            
            # Convert to SafeTensors
            success = self.convert_to_safetensors(model_data)
            
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
        self.logger.info("🎯 BITNET TO SAFETENSORS CONVERSION SUMMARY")
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
    parser = argparse.ArgumentParser(description="Convert BitNet models to SafeTensors format")
    parser.add_argument("--input-dir", type=str, default="./",
                       help="Input directory to search for BitNet models")
    parser.add_argument("--output-dir", type=str, default="./safetensors_models",
                       help="Output directory for converted SafeTensors models")
    
    args = parser.parse_args()
    
    # Create converter
    converter = BitNetToSafeTensorsConverter(
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
