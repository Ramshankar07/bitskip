#!/usr/bin/env python3
"""
Upload Simple BitNet Training Outputs to Hugging Face
Uploads all simple training outputs to Hugging Face Hub
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def find_output_directories() -> List[str]:
    """Find all output directories from simple training scripts."""
    output_dirs = []
    
    # Look for directories matching the simple training output patterns
    patterns = [
        # "output-simple",
        "output-bitnet-2b-simple", 
        "output-hbitlinear-1b-simple",
        "output-hbitlinear-2b-simple"
    ]
    
    for pattern in patterns:
        if os.path.exists(pattern):
            output_dirs.append(pattern)
            logging.info(f"Found output directory: {pattern}")
    
    return output_dirs


def create_model_card(model_name: str, model_type: str, config: Dict) -> str:
    """Create a model card for the uploaded model."""
    
    # Determine model description based on type
    if "hbitlinear" in model_name.lower():
        description = f"H-BitLinear {model_type} - Hardware-optimized BitNet model with 2-bit quantized weights"
        architecture = "H-BitLinear (Hardware-optimized BitLinear layers)"
    else:
        description = f"BitNet {model_type} - 2-bit quantized neural network model"
        architecture = "BitNet (2-bit quantized weights)"
    
    model_card = f"""---
license: apache-2.0
base_model: meta-llama/Llama-3.2-1B
tags:
- bitnet
- quantization
- 2-bit
- efficient
- transformer
- causal-lm
model-index:
- name: {model_name}
  results: []
---

# {model_name}

{description}

## Model Details

- **Architecture**: {architecture}
- **Parameters**: {model_type}
- **Weight Precision**: 2-bit quantized
- **Activation Precision**: 8-bit
- **Hidden Size**: {config.get('hidden_size', 'N/A')}
- **Layers**: {config.get('num_hidden_layers', 'N/A')}
- **Attention Heads**: {config.get('num_attention_heads', 'N/A')}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load model (requires custom BitNet implementation)
model = AutoModelForCausalLM.from_pretrained("{model_name}", trust_remote_code=True)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

This model was trained using the simplified BitNet training script with:
- Random data for testing purposes
- Mixed precision training (FP16)
- Gradient accumulation
- Basic checkpointing

## Model Performance

This is a simplified training run for demonstration purposes. For production use, train on real datasets with proper hyperparameter tuning.

## Citation

```bibtex
@article{{bitnet2024,
  title={{BitNet: Scaling 1-bit Transformers for Large Language Models}},
  author={{Liu, Zihang and Oguz, Barlas and Zhao, Cheng and Chang, Ernie and Stock, Pierre and Mehdad, Yashar and Shi, Yangyang and Krishnamoorthi, Raghuraman and Chandra, Vikas}},
  journal={{arXiv preprint arXiv:2402.17764}},
  year={{2024}}
}}
```
"""
    
    return model_card


def upload_model_to_hf(output_dir: str, repo_name: str, private: bool = False) -> bool:
    """Upload a model directory to Hugging Face."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Check if repo exists, create if not
        try:
            api.repo_info(repo_name)
            logger.info(f"Repository {repo_name} already exists")
        except:
            logger.info(f"Creating repository {repo_name}")
            create_repo(
                repo_id=repo_name,
                private=private,
                repo_type="model"
            )
        
        # Check if final_model directory exists
        final_model_dir = os.path.join(output_dir, "final_model")
        if not os.path.exists(final_model_dir):
            logger.warning(f"Final model directory not found: {final_model_dir}")
            return False
        
        # Load config to create model card
        config_path = os.path.join(final_model_dir, "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create model card
        model_card = create_model_card(repo_name, "1B" if "1b" in repo_name.lower() else "2B", config)
        
        # Write model card
        readme_path = os.path.join(final_model_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        # Upload the model
        logger.info(f"Uploading {final_model_dir} to {repo_name}")
        upload_folder(
            folder_path=final_model_dir,
            repo_id=repo_name,
            repo_type="model",
            commit_message=f"Upload {repo_name} model"
        )
        
        logger.info(f"Successfully uploaded {repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload {repo_name}: {e}")
        return False


def upload_all_simple_models(private: bool = False, username: str = None) -> Dict[str, bool]:
    """Upload all simple training outputs to Hugging Face."""
    logger = logging.getLogger(__name__)
    
    if not username:
        username = os.getenv("HF_USERNAME")
        if not username:
            logger.error("Please provide username or set HF_USERNAME environment variable")
            return {}
    
    # Find all output directories
    output_dirs = find_output_directories()
    
    if not output_dirs:
        logger.warning("No output directories found. Make sure you've run the simple training scripts first.")
        return {}
    
    # Map output directories to repo names
    repo_mapping = {
        # "output-simple": f"{username}/bitnet-1b-simple",
        "output-bitnet-2b-simple": f"{username}/bitnet-2b-simple", 
        "output-hbitlinear-1b-simple": f"{username}/hbitlinear-1b-simple",
        "output-hbitlinear-2b-simple": f"{username}/hbitlinear-2b-simple"
    }
    
    results = {}
    
    for output_dir in output_dirs:
        if output_dir in repo_mapping:
            repo_name = repo_mapping[output_dir]
            logger.info(f"Uploading {output_dir} to {repo_name}")
            success = upload_model_to_hf(output_dir, repo_name, private)
            results[repo_name] = success
        else:
            logger.warning(f"No repo mapping found for {output_dir}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upload Simple BitNet Models to Hugging Face')
    
    parser.add_argument('--username', type=str, default=None,
                       help='Hugging Face username (defaults to HF_USERNAME env var)')
    parser.add_argument('--private', action='store_true',
                       help='Make repositories private')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Specific output directory to upload (optional)')
    parser.add_argument('--repo-name', type=str, default=None,
                       help='Specific repository name (optional)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting upload of simple BitNet models to Hugging Face")
    
    # Check if HF token is available
    if not os.getenv("HUGGINGFACE_TOKEN"):
        logger.error("HUGGINGFACE_TOKEN environment variable not set")
        logger.error("Please set your Hugging Face token: export HUGGINGFACE_TOKEN=your_token")
        return
    
    if args.output_dir and args.repo_name:
        # Upload specific model
        logger.info(f"Uploading specific model: {args.output_dir} -> {args.repo_name}")
        success = upload_model_to_hf(args.output_dir, args.repo_name, args.private)
        if success:
            logger.info("Upload completed successfully!")
        else:
            logger.error("Upload failed!")
    else:
        # Upload all models
        results = upload_all_simple_models(args.private, args.username)
        
        logger.info("\n" + "="*60)
        logger.info("UPLOAD SUMMARY")
        logger.info("="*60)
        
        for repo_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"{repo_name}: {status}")
        
        successful_uploads = sum(1 for success in results.values() if success)
        total_uploads = len(results)
        
        logger.info(f"\nTotal: {successful_uploads}/{total_uploads} models uploaded successfully")
        
        if successful_uploads > 0:
            logger.info("\n🎉 Upload completed! Your models are now available on Hugging Face Hub.")
            logger.info("You can find them at:")
            for repo_name, success in results.items():
                if success:
                    logger.info(f"  https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    main()
