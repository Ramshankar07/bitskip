#!/usr/bin/env python3
"""
Create model cards (README.md) for all BitNet models and upload them to Hugging Face Hub.

Usage:
    export HUGGINGFACE_TOKEN="your_token"
    python scripts/create_model_cards.py [--private] [--revision main]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from huggingface_hub import HfApi, create_repo, hf_hub_url


def parse_args():
    parser = argparse.ArgumentParser(description="Create and upload model cards for BitNet models")
    parser.add_argument("--private", action="store_true", help="Create repos as private")
    parser.add_argument("--revision", default="main", help="Branch/tag to upload to")
    parser.add_argument("--token", help="Hugging Face token (overrides HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="Generate cards but don't upload")
    return parser.parse_args()


def get_model_info(model_path: str) -> Dict:
    """Extract model information from config.json and model.pt"""
    config_path = os.path.join(model_path, "config.json")
    model_path_pt = os.path.join(model_path, "model.pt")
    
    info = {
        "config": {},
        "model_size_mb": 0,
        "has_model": False
    }
    
    # Load config
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                info["config"] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    # Get model size
    if os.path.exists(model_path_pt):
        info["model_size_mb"] = os.path.getsize(model_path_pt) / (1024 * 1024)
        info["has_model"] = True
    
    return info


def create_model_card(model_name: str, model_info: Dict, repo_id: str) -> str:
    """Generate a comprehensive model card in Markdown format"""
    
    config = model_info["config"]
    model_size_mb = model_info["model_size_mb"]
    has_model = model_info["has_model"]
    
    # Extract key config values
    hidden_size = config.get("hidden_size", "Unknown")
    num_layers = config.get("num_hidden_layers", "Unknown")
    num_heads = config.get("num_attention_heads", "Unknown")
    vocab_size = config.get("vocab_size", "Unknown")
    activation_bits = config.get("activation_bits", "Unknown")
    weight_bits = config.get("weight_bits", "Unknown")
    
    # Determine model architecture
    if "hbitlinear" in model_name.lower():
        architecture = "H-BitLinear BitNet"
        description = "A BitNet model using H-BitLinear layers with Hadamard transformations for efficient quantization"
    else:
        architecture = "BitNet"
        description = "A BitNet model with standard BitLinear layers for efficient quantization"
    
    # Model size category
    if model_size_mb > 2000:
        size_category = "Large (~2B parameters)"
    elif model_size_mb > 1000:
        size_category = "Medium (~1B parameters)"
    else:
        size_category = "Small"
    
    # Generate model card
    card = f"""---
license: apache-2.0
tags:
- bitnet
- quantization
- language-model
- causal-lm
- pytorch
- transformers
- {"h-bitlinear" if "hbitlinear" in model_name.lower() else "bitlinear"}
model-index:
- name: {repo_id}
  results: []
---

# {architecture} - {model_name.replace('-', ' ').title()}

{description} with {size_category}.

## Model Details

- **Architecture**: {architecture}
- **Model Size**: {size_category}
- **Parameters**: ~{int(model_size_mb * 0.25):,}M (estimated)
- **Hidden Size**: {hidden_size}
- **Number of Layers**: {num_layers}
- **Attention Heads**: {num_heads}
- **Vocabulary Size**: {vocab_size:,}
- **Activation Bits**: {activation_bits}
- **Weight Bits**: {weight_bits}
- **Model File Size**: {model_size_mb:.1f} MB

## Model Description

This is a {architecture} model trained using efficient quantization techniques. The model uses:

- **BitLinear layers** for weight quantization
- **Grouped Query Attention (GQA)** for memory efficiency
- **Layer Skipping** for dynamic computation
- **Early Exit** capabilities for faster inference
- **Mixed Precision Training** (FP16)

## Usage

### Using Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Using BitNet Inference Engine

```python
from bitnet.inference.engine import BitNetInferenceEngine

# Initialize engine
engine = BitNetInferenceEngine(
    model_path="{repo_id}",
    device="cuda"  # or "cpu"
)

# Generate text
prompt = "The future of artificial intelligence is"
response = engine.generate(
    prompt=prompt,
    max_new_tokens=100,
    temperature=0.7
)
print(response)
```

## Training Details

- **Training Framework**: PyTorch with custom BitNet implementation
- **Mixed Precision**: FP16 training with gradient scaling
- **Optimizer**: AdamW with custom learning rate scheduling
- **Dataset**: FineWeb-Edu (streaming)
- **Tokenizer**: Llama 3 tokenizer
- **Hardware**: Optimized for H200 GPU

## Performance

The model is optimized for:
- **Memory Efficiency**: Through quantization and GQA
- **Speed**: Via layer skipping and early exit
- **Quality**: Maintains performance despite quantization

## Limitations

- Model performance may vary compared to full-precision models
- Quantization artifacts may be present in generated text
- Early exit may reduce output quality for complex tasks

## Citation

If you use this model, please cite:

```bibtex
@misc{{bitnet_{model_name.replace('-', '_')},
  title={{{architecture} - {model_name.replace('-', ' ').title()}}},
  author={{Ram07}},
  year={{2025}},
  url={{https://huggingface.co/{repo_id}}},
  note={{BitNet model with efficient quantization}}
}}
```

## License

This model is released under the Apache 2.0 License.

## Contact

For questions or issues, please open an issue on the [model repository](https://huggingface.co/{repo_id}).

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return card


def upload_model_card(api: HfApi, repo_id: str, card_content: str, revision: str, private: bool, dry_run: bool):
    """Upload model card to Hugging Face Hub"""
    
    if dry_run:
        print(f"[DRY RUN] Would upload model card to {repo_id}")
        return
    
    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id)
        repo_exists = True
    except Exception:
        repo_exists = False
    
    if not repo_exists:
        print(f"Creating repository {repo_id} (private={private})...")
        create_repo(
            repo_id=repo_id,
            token=api.token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
    
    # Upload model card
    print(f"Uploading model card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=card_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=api.token,
    )
    
    url = hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision)
    print(f"Model card uploaded: {url}")


def main():
    args = parse_args()
    
    # Get token
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: No token provided. Set HUGGINGFACE_TOKEN or pass --token.")
        sys.exit(1)
    
    # Set default HF_USER
    hf_user = os.environ.get("HF_USER", "Ram07")
    
    # Define model directories
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    model_dirs = {
        "bitnet-1b": repo_root / "output-bitnet-1b" / "final_model",
        "bitnet-2b": repo_root / "output-quadratic-2b-hf" / "final_model", 
        "hbitlinear-1b": repo_root / "output-bitnet-hbitlinear-1b" / "final_model",
        "hbitlinear-2b": repo_root / "output-quadratic-hbitlinear-2b-hf" / "final_model",
    }
    
    api = HfApi(token=token)
    
    print(f"Creating model cards for {hf_user}...")
    print(f"Revision: {args.revision}")
    print(f"Private: {args.private}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 60)
    
    for model_name, model_path in model_dirs.items():
        if not model_path.exists():
            print(f"[SKIP] {model_name}: Directory not found - {model_path}")
            continue
        
        print(f"\n[INFO] Processing {model_name}")
        print(f"Path: {model_path}")
        
        # Get model info
        model_info = get_model_info(str(model_path))
        
        # Create repo ID
        repo_id = f"{hf_user}/{model_name}"
        
        # Generate model card
        card_content = create_model_card(model_name, model_info, repo_id)
        
        # Upload model card
        upload_model_card(api, repo_id, card_content, args.revision, args.private, args.dry_run)
        
        print(f"[DONE] {model_name}")
    
    print("\n" + "=" * 60)
    print("Model card creation completed!")


if __name__ == "__main__":
    main()
