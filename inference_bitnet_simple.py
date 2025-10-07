#!/usr/bin/env python3
"""
BitNet Simple Inference Script - Clean Minimal Implementation
Streamlined version for easy understanding and modification

Supports both local model checkpoints and Hugging Face model downloads.

Usage Examples:
    # Test with local model
    python inference_bitnet_simple.py --model_path ./output-simple/final_model/model.pt
    
    # Test with Hugging Face model
    python inference_bitnet_simple.py --model_path Ram07/bitnet-1b-simple
    
    # Test with custom prompt and benchmark
    python inference_bitnet_simple.py --model_path Ram07/bitnet-1b-simple --prompt "Hello world" --benchmark --num_runs 10
    
    # Test with HF token
    python inference_bitnet_simple.py --model_path Ram07/bitnet-1b-simple --hf_token your_token_here
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable Xet storage to avoid download issues
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Import BitNet components
from bitnet.modeling.model import BitNetModel
from bitnet.utils.default_config import DefaultConfig
from bitnet.inference.engine import BitNetInferenceEngine

# Import for Hugging Face model downloading
try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Cannot download models from HF.")


class SimpleBitNet(nn.Module):
    """Simplified BitNet wrapper for inference."""
    
    def __init__(self, config_dict):
        super().__init__()
        self.config = self._convert_config(config_dict)
        self.model = BitNetModel(self.config)
        self.lm_head = self.model.lm_head
    
    def _convert_config(self, config_dict):
        """Convert config dict to DefaultConfig."""
        return DefaultConfig(
            vocab_size=config_dict.get('vocab_size', 128256),
            hidden_size=config_dict.get('hidden_size', 1536),
            num_hidden_layers=config_dict.get('num_hidden_layers', 20),
            num_attention_heads=config_dict.get('num_attention_heads', 16),
            num_kv_heads=config_dict.get('num_key_value_heads', 4),
            max_position_embeddings=config_dict.get('max_position_embeddings', 1024),
            layer_norm_eps=config_dict.get('layer_norm_eps', 1e-5),
            hidden_dropout_prob=config_dict.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config_dict.get('attention_dropout', 0.1),
            initializer_range=config_dict.get('initializer_range', 0.02),
            activation_bits=config_dict.get('activation_bits', 8),
            weight_bits=config_dict.get('weight_bits', 2),
            use_layer_skipping=bool(config_dict.get('use_layer_skipping', False)),
            skip_probability=config_dict.get('skip_probability', 0.0),
            min_layers_to_keep=config_dict.get('min_layers_to_keep', 1),
            use_early_exit=bool(config_dict.get('use_early_exit', False)),
            early_exit_threshold=config_dict.get('early_exit_threshold', 0.0),
            gradient_checkpointing=bool(config_dict.get('gradient_checkpointing', False))
        )
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass for inference."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        return {
            'logits': outputs.logits
        }
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9, 
                 attention_mask=None, early_exit_layer=None):
        """Simple text generation."""
        self.eval()
        generated_tokens = []
        
        with torch.no_grad():
            current_input_ids = input_ids
            current_attention_mask = attention_mask
            
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask
                )
                
                # Get logits for last token
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == self.config.vocab_size - 1:  # Assuming EOS token
                    break
                
                generated_tokens.append(next_token.item())
                
                # Update input for next iteration
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                if current_attention_mask is not None:
                    current_attention_mask = torch.cat([
                        current_attention_mask,
                        torch.ones((current_attention_mask.shape[0], 1),
                                 dtype=current_attention_mask.dtype,
                                 device=current_attention_mask.device)
                    ], dim=1)
        
        return generated_tokens


def load_model(model_path, device):
    """Load model from checkpoint or Hugging Face."""
    logger = logging.getLogger(__name__)
    
    # Check if it's a Hugging Face repo ID
    if "/" in model_path and not os.path.exists(model_path):
        logger.info(f"Detected Hugging Face repo ID: {model_path}")
        return load_model_from_hf(model_path, device)
    
    # Load from local checkpoint
    logger.info(f"Loading model from local path: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        logger.error("No config found in checkpoint")
        return None, None
    
    # Create model
    model = SimpleBitNet(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model config: {config}")
    
    return model, config


def load_model_from_hf(repo_id, device):
    """Load model from Hugging Face Hub."""
    logger = logging.getLogger(__name__)
    
    if not HF_HUB_AVAILABLE:
        logger.error("huggingface_hub not available. Cannot download from HF.")
        return None, None
    
    try:
        logger.info(f"Downloading model from Hugging Face: {repo_id}")
        
        # Download model files
        local_dir = snapshot_download(
            repo_id=repo_id,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            local_dir=f"./models/{repo_id.replace('/', '--')}",
            ignore_patterns=["*.bin", "*.h5", "*.tflite", "*.tar.gz"]
        )
        
        logger.info(f"Model downloaded to: {local_dir}")
        
        # Try to load using BitNetInferenceEngine first
        try:
            logger.info("Attempting to load with BitNetInferenceEngine...")
            engine = BitNetInferenceEngine(local_dir)
            model = engine.model
            config = engine.config
            
            logger.info("Successfully loaded with BitNetInferenceEngine")
            return model, config
            
        except Exception as e:
            logger.warning(f"BitNetInferenceEngine failed: {e}")
            logger.info("Falling back to manual loading...")
            
            # Fallback: try to load config and create model manually
            config_path = os.path.join(local_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Convert to DefaultConfig format
                config = DefaultConfig(
                    vocab_size=config_dict.get('vocab_size', 128256),
                    hidden_size=config_dict.get('hidden_size', 1536),
                    num_hidden_layers=config_dict.get('num_hidden_layers', 20),
                    num_attention_heads=config_dict.get('num_attention_heads', 16),
                    num_kv_heads=config_dict.get('num_key_value_heads', 4),
                    max_position_embeddings=config_dict.get('max_position_embeddings', 1024),
                    layer_norm_eps=config_dict.get('layer_norm_eps', 1e-5),
                    hidden_dropout_prob=config_dict.get('hidden_dropout_prob', 0.1),
                    attention_probs_dropout_prob=config_dict.get('attention_dropout', 0.1),
                    initializer_range=config_dict.get('initializer_range', 0.02),
                    activation_bits=config_dict.get('activation_bits', 8),
                    weight_bits=config_dict.get('weight_bits', 2),
                    use_layer_skipping=bool(config_dict.get('use_layer_skipping', False)),
                    skip_probability=config_dict.get('skip_probability', 0.0),
                    min_layers_to_keep=config_dict.get('min_layers_to_keep', 1),
                    use_early_exit=bool(config_dict.get('use_early_exit', False)),
                    early_exit_threshold=config_dict.get('early_exit_threshold', 0.0),
                    gradient_checkpointing=bool(config_dict.get('gradient_checkpointing', False))
                )
                
                # Create model
                model = SimpleBitNet(config_dict)
                model.to(device)
                
                logger.info("Successfully loaded model manually")
                return model, config_dict
            else:
                logger.error(f"Config file not found: {config_path}")
                return None, None
                
    except Exception as e:
        logger.error(f"Failed to load model from Hugging Face: {e}")
        return None, None


def setup_logging():
    """Setup simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple BitNet Inference')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint or Hugging Face repo ID (e.g., Ram07/bitnet-1b-simple)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda/cpu)')
    
    # Generation configuration
    parser.add_argument('--prompt', type=str, default='The future of AI is',
                       help='Input prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    
    # Benchmarking
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of benchmark runs')
    
    # Hugging Face specific
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face token (defaults to HUGGINGFACE_TOKEN env var)')
    
    return parser.parse_args()


def benchmark_inference(model, tokenizer, prompt, max_new_tokens, num_runs, device):
    """Simple inference benchmark."""
    logger = logging.getLogger(__name__)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    times = []
    tokens_per_second = []
    
    logger.info(f"Running {num_runs} inference runs...")
    
    for run in range(num_runs):
        start_time = time.time()
        
        # Generate tokens
        generated_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask
        )
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        num_tokens = len(generated_tokens)
        tps = num_tokens / total_time if total_time > 0 else 0
        
        times.append(total_time)
        tokens_per_second.append(tps)
        
        logger.info(f"Run {run+1}: {num_tokens} tokens in {total_time:.3f}s ({tps:.2f} tok/s)")
    
    # Calculate averages
    avg_time = sum(times) / len(times)
    avg_tps = sum(tokens_per_second) / len(tokens_per_second)
    
    logger.info(f"\nBenchmark Results:")
    logger.info(f"Average time: {avg_time:.3f}s")
    logger.info(f"Average tokens/sec: {avg_tps:.2f}")
    logger.info(f"Average tokens: {max_new_tokens}")
    
    return {
        'avg_time': avg_time,
        'avg_tokens_per_second': avg_tps,
        'avg_tokens': max_new_tokens,
        'times': times,
        'tokens_per_second': tokens_per_second
    }


def main():
    """Main inference function."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting Simple BitNet Inference")
    logger.info(f"Configuration: {vars(args)}")
    
    # Set HF token if provided
    if args.hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = args.hf_token
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
            token=os.getenv("HUGGINGFACE_TOKEN"),
            use_fast=True,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
        logger.info(f"Loaded tokenizer with vocabulary size: {len(tokenizer)}")
    except Exception as e:
        logger.error(f"Could not load tokenizer: {e}")
        return
    
    # Load model
    model, config = load_model(args.model_path, device)
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Model size (FP32): {total_params * 4 / (1024**3):.2f} GB")
    
    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    logger.info(f"Input prompt: '{args.prompt}'")
    logger.info(f"Input tokens: {input_ids.shape[1]}")
    
    # Generate text
    logger.info("Generating text...")
    start_time = time.time()
    
    generated_tokens = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        attention_mask=attention_mask
    )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = args.prompt + generated_text
    
    # Calculate metrics
    num_tokens = len(generated_tokens)
    tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
    
    logger.info(f"\nGenerated text:")
    logger.info(f"'{full_text}'")
    logger.info(f"\nGeneration metrics:")
    logger.info(f"Generated tokens: {num_tokens}")
    logger.info(f"Generation time: {generation_time:.3f}s")
    logger.info(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info("\n" + "="*50)
        benchmark_results = benchmark_inference(
            model, tokenizer, args.prompt, args.max_new_tokens, 
            args.num_runs, device
        )
        
        # Save benchmark results
        results_file = "inference_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Benchmark results saved to {results_file}")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
