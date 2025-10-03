#!/usr/bin/env python3
"""
Fixed BitNet to SafeTensors Converter Script
Properly handles BitNet model state dicts and preserves quantization information.
"""

import os
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from datetime import datetime

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class BitNetToSafeTensorsConverter:
    """Properly converts BitNet .pt models to SafeTensors format."""
    
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
        """Find all BitNet model files."""
        models = []
        
        search_patterns = [
            "*/final_model/model.pt",
        ]
        
        found_files = set()
        for pattern in search_patterns:
            for model_path in self.input_dir.glob(pattern):
                if model_path.is_file() and str(model_path) not in found_files:
                    found_files.add(str(model_path))
                    
                    # Try to find associated config
                    config_path = model_path.parent / "config.json"
                    
                    # Determine model name
                    model_name = self._determine_model_name(model_path)
                    
                    models.append({
                        'path': str(model_path.parent),
                        'name': model_name,
                        'config_path': str(config_path) if config_path.exists() else None,
                        'model_path': str(model_path)
                    })
                    self.logger.info(f"Found model: {model_name} at {model_path}")
        
        return models
    
    def _determine_model_name(self, model_path: Path) -> str:
        """Determine a meaningful name for the model."""
        path_str = str(model_path).lower()
        
        # Extract info from path
        if "checkpoint-" in path_str:
            # Extract checkpoint number
            import re
            match = re.search(r'checkpoint-(\d+)', path_str)
            if match:
                checkpoint_num = match.group(1)
                if "hbitlinear" in path_str:
                    return f"bitnet-hbitlinear-checkpoint-{checkpoint_num}"
                else:
                    return f"bitnet-checkpoint-{checkpoint_num}"
        
        if "final_model" in path_str:
            parent_dir = model_path.parent.parent.name
            if "hbitlinear" in parent_dir:
                size = self._extract_size_from_path(model_path)
                return f"bitnet-hbitlinear-{size}-final"
            elif "bitnet" in parent_dir:
                size = self._extract_size_from_path(model_path)
                return f"bitnet-{size}-final"
            else:
                return f"{parent_dir}-final"
        
        # Default: use parent directory name
        return model_path.parent.name
    
    def _extract_size_from_path(self, path: Path) -> str:
        """Extract model size from path."""
        path_str = str(path).lower()
        for size in ["1b", "2b", "3b", "7b", "13b"]:
            if size in path_str:
                return size
        return "unknown"
    
    def load_bitnet_model(self, model_info: Dict) -> Optional[Dict]:
        """Properly load BitNet model state dict."""
        try:
            # Load model checkpoint
            self.logger.info(f"Loading model from {model_info['model_path']}...")
            checkpoint = torch.load(model_info['model_path'], map_location='cpu')
            
            # Extract state dict based on checkpoint format
            state_dict = None
            metadata = {}
            
            if isinstance(checkpoint, dict):
                # Common checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    metadata = {k: v for k, v in checkpoint.items() 
                               if k != 'model_state_dict' and not k.endswith('state_dict')}
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    metadata = {k: v for k, v in checkpoint.items() 
                               if k != 'state_dict' and not k.endswith('state_dict')}
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    metadata = {k: v for k, v in checkpoint.items() if k != 'model'}
                else:
                    # Assume the dict itself is the state dict
                    # Check if it contains tensor keys
                    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                        state_dict = checkpoint
                    else:
                        self.logger.error(f"Cannot find state dict in checkpoint keys: {checkpoint.keys()}")
                        return None
            else:
                # Direct state dict (older format)
                state_dict = checkpoint
            
            if state_dict is None:
                self.logger.error("Could not extract state dict from checkpoint")
                return None
            
            # Load config if available
            config = {}
            if model_info['config_path']:
                try:
                    with open(model_info['config_path'], 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load config: {e}")
            
            # Analyze model architecture from state dict
            model_analysis = self._analyze_model_architecture(state_dict)
            
            return {
                'state_dict': state_dict,
                'config': config,
                'metadata': metadata,
                'model_info': model_info,
                'model_analysis': model_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_model_architecture(self, state_dict: Dict) -> Dict:
        """Analyze model architecture from state dict."""
        analysis = {
            'total_parameters': 0,
            'layers': {},
            'has_bitlinear': False,
            'has_hbitlinear': False,
            'quantization_info': {}
        }
        
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                analysis['total_parameters'] += tensor.numel()
                
                # Detect layer types
                if 'BitLinear' in key or 'bitlinear' in key.lower():
                    analysis['has_bitlinear'] = True
                if 'HBitLinear' in key or 'hbitlinear' in key.lower():
                    analysis['has_hbitlinear'] = True
                
                # Extract layer info
                layer_name = key.split('.')[0] if '.' in key else key
                if layer_name not in analysis['layers']:
                    analysis['layers'][layer_name] = []
                analysis['layers'][layer_name].append(key)
                
                # Check for quantization scales
                if 'weight_scale' in key or 'activation_scale' in key:
                    analysis['quantization_info'][key] = {
                        'shape': list(tensor.shape),
                        'values': tensor.cpu().numpy().tolist() if tensor.numel() < 10 else 'too_large'
                    }
        
        return analysis
    
    def _process_state_dict_for_safetensors(self, state_dict: Dict) -> Dict:
        """Process state dict to ensure compatibility with SafeTensors."""
        processed = {}
        
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Ensure tensor is contiguous and on CPU
                tensor = value.cpu().contiguous()
                
                # Handle different dtypes
                if tensor.dtype == torch.bfloat16:
                    # Convert bfloat16 to float16 for better compatibility
                    self.logger.warning(f"Converting {key} from bfloat16 to float16")
                    tensor = tensor.to(torch.float16)
                
                # Clone to ensure no memory sharing
                processed[key] = tensor.clone()
            else:
                # Skip non-tensor values (SafeTensors only stores tensors)
                self.logger.debug(f"Skipping non-tensor value: {key}")
        
        return processed
    
    def convert_to_safetensors(self, model_data: Dict) -> bool:
        """Convert BitNet model to SafeTensors format."""
        try:
            model_info = model_data['model_info']
            state_dict = model_data['state_dict']
            config = model_data.get('config', {})
            metadata = model_data.get('metadata', {})
            model_analysis = model_data.get('model_analysis', {})
            
            self.logger.info(f"Converting {model_info['name']} to SafeTensors...")
            self.logger.info(f"  Total parameters: {model_analysis['total_parameters']:,}")
            self.logger.info(f"  Has BitLinear: {model_analysis['has_bitlinear']}")
            self.logger.info(f"  Has HBitLinear: {model_analysis['has_hbitlinear']}")
            
            # Create output directory
            model_output_dir = self.output_dir / model_info['name']
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process state dict for SafeTensors
            processed_state_dict = self._process_state_dict_for_safetensors(state_dict)
            
            # Create metadata for SafeTensors file
            safetensors_metadata = {
                'format': 'pt',
                'model_type': 'bitnet',
                'total_parameters': str(model_analysis['total_parameters']),
                'has_bitlinear': str(model_analysis['has_bitlinear']),
                'has_hbitlinear': str(model_analysis['has_hbitlinear']),
                'conversion_date': datetime.now().isoformat(),
                'converter_version': '1.0.0'
            }
            
            
            # Save to SafeTensors
            safetensors_path = model_output_dir / "model.safetensors"
            save_file(processed_state_dict, str(safetensors_path), metadata=safetensors_metadata)
            
            self.logger.info(f"  Saved SafeTensors: {safetensors_path}")
            self.logger.info(f"  File size: {safetensors_path.stat().st_size / (1024**2):.2f} MB")
            
            # Save config if available
            if config:
                config_path = model_output_dir / "config.json"
                
                # Add BitNet-specific fields to config
                config['model_type'] = 'bitnet'
                config['quantization_config'] = {
                    'quant_method': 'bitnet',
                    'weight_bits': 2,
                    'activation_bits': 8
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.logger.info(f"  Saved config: {config_path}")
            
            # Create model card
            self._create_model_card(model_output_dir, model_info, model_analysis, safetensors_metadata)
            
            # Save conversion info
            conversion_info = {
                'original_path': model_info['model_path'],
                'conversion_date': datetime.now().isoformat(),
                'model_analysis': model_analysis,
                'safetensors_metadata': safetensors_metadata,
                'file_sizes': {
                    'original_mb': Path(model_info['model_path']).stat().st_size / (1024**2),
                    'safetensors_mb': safetensors_path.stat().st_size / (1024**2)
                }
            }
            
            info_path = model_output_dir / "conversion_info.json"
            with open(info_path, 'w') as f:
                json.dump(conversion_info, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to convert {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_model_card(self, output_dir: Path, model_info: Dict, model_analysis: Dict, metadata: Dict):
        """Create a model card for the converted model."""
        readme_content = f"""---
tags:
- bitnet
- quantization
- safetensors
---

# {model_info['name']}

Converted BitNet model in SafeTensors format.

## Model Details

- **Parameters**: {model_analysis['total_parameters']:,}
- **Architecture**: BitNet with {"H-BitLinear" if model_analysis['has_hbitlinear'] else "BitLinear"} layers
- **Quantization**: Ternary weights (2-bit), 8-bit activations
- **Format**: SafeTensors

## Usage

```python
from safetensors.torch import load_file
import torch

# Load the model
state_dict = load_file("model.safetensors")

# Initialize your BitNet model architecture
model = YourBitNetModel(config)  # Replace with your model initialization

# Load the weights
model.load_state_dict(state_dict)

# Use the model
model.eval()
with torch.no_grad():
    output = model(input_ids)
```

## Files

- `model.safetensors` - Model weights in SafeTensors format
- `config.json` - Model configuration (if available)
- `conversion_info.json` - Details about the conversion

## Conversion Details

- **Converted on**: {metadata.get('conversion_date', 'Unknown')}
- **Original format**: PyTorch (.pt)
- **Converter version**: {metadata.get('converter_version', 'Unknown')}
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def convert_all_models(self) -> Dict:
        """Convert all found BitNet models."""
        self.logger.info("Scanning for BitNet models...")
        
        models = self.find_bitnet_models()
        self.conversion_results['total_models'] = len(models)
        
        if not models:
            self.logger.warning("No BitNet models found!")
            return self.conversion_results
        
        self.logger.info(f"Found {len(models)} model(s) to convert")
        
        for i, model_info in enumerate(models, 1):
            self.logger.info(f"\n[{i}/{len(models)}] Processing: {model_info['name']}")
            
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
        
        # Save summary
        self._save_conversion_summary()
        
        return self.conversion_results
    
    def _save_conversion_summary(self):
        """Save and display conversion summary."""
        summary_path = self.output_dir / "conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.conversion_results, f, indent=2)
        
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(f"Total models found: {self.conversion_results['total_models']}")
        print(f"Successful: {self.conversion_results['successful_conversions']}")
        print(f"Failed: {self.conversion_results['failed_conversions']}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Summary saved to: {summary_path}")
        print("="*60)


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert BitNet models to SafeTensors format")
    parser.add_argument("--input-dir", type=str, default="./",
                       help="Input directory to search for BitNet models")
    parser.add_argument("--output-dir", type=str, default="./safetensors_models",
                       help="Output directory for converted models")
    
    args = parser.parse_args()
    
    converter = BitNetToSafeTensorsConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    converter.convert_all_models()


if __name__ == "__main__":
    main()