#!/usr/bin/env python3
"""
Inference Benchmark Script for 1B Improved BitNet Model (FP16 Optimized)
Measures TPOT, TTFT, tokens/sec, ITL with various early exit thresholds

This script benchmarks the Hugging Face compatible 1B BitNet model with:
- Time Per Output Token (TPOT)
- Time To First Token (TTFT) 
- Tokens per second
- Inter-Token Latency (ITL)
- Early exit analysis at different thresholds
- FP16 inference optimization
- KV cache support for efficient generation
- Comprehensive timing measurements
"""

import os
import time
import json
import argparse
import logging
import statistics
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Import for Hugging Face model downloading
try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Note: Using torch.cuda.amp.autocast directly for proper mixed precision

# Import the improved model
from bitnet.inference.engine import BitNetInferenceEngine

# Load environment variables
load_dotenv()

# Disable HuggingFace caching
os.environ["HF_DATASETS_CACHE"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
os.environ["HF_HOME"] = ""

class InferenceBenchmark:
    """Comprehensive inference benchmark for BitNet models."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.results = {}
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('inference_benchmark.log'),
                logging.StreamHandler()
            ]
        )
    
    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    def _is_huggingface_repo(self, model_path: str) -> bool:
        """Check if the model path is a Hugging Face repository ID."""
        return "/" in model_path and not os.path.exists(model_path)
    
    def _download_from_huggingface(self, repo_id: str) -> Optional[str]:
        """Download model from Hugging Face Hub."""
        if not HF_HUB_AVAILABLE:
            self.logger.error("huggingface_hub not available. Install with: pip install huggingface_hub")
            return None
        
        try:
            self.logger.info(f"Downloading model from Hugging Face: {repo_id}")
            
            # Download to local cache
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir="./models",
                local_files_only=False
            )
            
            self.logger.info(f"Model downloaded to: {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to download model from HF: {e}")
            return None
    
    def _validate_model_path(self, model_path: str) -> str:
        """Validate and resolve model path, downloading if necessary."""
        # Check if it's a Hugging Face repo ID
        if self._is_huggingface_repo(model_path):
            self.logger.info(f"Detected Hugging Face repository: {model_path}")
            downloaded_path = self._download_from_huggingface(model_path)
            if downloaded_path:
                return downloaded_path
            else:
                raise FileNotFoundError(f"Failed to download model from Hugging Face: {model_path}")
        
        # Check if local path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # If it's a directory, check for required files
        if os.path.isdir(model_path):
            # Check for SafeTensors files
            safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            if safetensors_files:
                self.logger.info(f"Found SafeTensors model: {safetensors_files[0]}")
                return model_path
            
            # Check for .pt files
            pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
            if pt_files:
                self.logger.info(f"Found PyTorch model: {pt_files[0]}")
                return model_path
            
            # Check for config.json
            if os.path.exists(os.path.join(model_path, "config.json")):
                self.logger.info("Found model directory with config.json")
                return model_path
            
            raise FileNotFoundError(f"No valid model files found in directory: {model_path}")
        
        # If it's a file, check if it's a valid model file
        if os.path.isfile(model_path):
            if model_path.endswith(('.pt', '.safetensors')):
                self.logger.info(f"Found model file: {model_path}")
                return model_path
            else:
                raise ValueError(f"Unsupported model file format: {model_path}")
        
        raise ValueError(f"Invalid model path: {model_path}")

    def load_model(self):
        """Load the BitNet model and tokenizer using the unified engine."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Validate and resolve model path (download if necessary)
            resolved_model_path = self._validate_model_path(self.model_path)
            self.logger.info(f"Resolved model path: {resolved_model_path}")
            
            # Resolve model directory and model file path
            if os.path.isfile(resolved_model_path):
                model_dir = os.path.dirname(resolved_model_path)
                model_file = resolved_model_path
                self.logger.info(f"Model file detected, using directory: {model_dir}")
            else:
                model_dir = resolved_model_path
                model_file = os.path.join(model_dir, "model.pt")
                self.logger.info(f"Model directory detected: {model_dir}")
            
            # Tokenizer path/name
            tokenizer_name = os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
            
            # Initialize unified engine (auto-detects model vs model2)
            self.engine = BitNetInferenceEngine(
                model_path=model_file,
                tokenizer_path=tokenizer_name,
                device=self.device,
                model_type="auto",
            )
            
            # Expose model and tokenizer for benchmarking internals
            self.model = self.engine.model
            self.tokenizer = self.engine.tokenizer
            
            # Convert model weights to FP16 for inference efficiency (only weights, not inputs)
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.half().cuda()
                self.logger.info("Model converted to FP16 for inference")
            
            self.logger.info("Model and tokenizer loaded successfully via engine")
            self.logger.info(f"Model dtype: {self._model_dtype()}")
            self.logger.info(f"Model device: {self._model_device()}")
        
        except Exception as e:
            self.logger.error(f"Failed to load model via engine: {str(e)}")
            raise
    
    def generate_with_timing(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        early_exit_threshold: float = 0.95,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_runs: int = 5
    ) -> Dict:
        """
        Generate text with detailed timing measurements.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            early_exit_threshold: Early exit confidence threshold
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing metrics
        """
        self.logger.info(f"Benchmarking with early_exit_threshold={early_exit_threshold}")
        
        # Tokenize input and ensure dict of tensors on correct device
        tok_out = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in tok_out.items()}
        
        # Ensure batch dimension is present (should be [1, seq_len])
        if inputs["input_ids"].dim() == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if "attention_mask" in inputs and inputs["attention_mask"].dim() == 1:
            inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
        
        # Keep input_ids as integers (token indices) - NEVER convert to FP16
        input_length = inputs["input_ids"].shape[1]
        
        all_metrics = []
        
        for run in range(num_runs):
            self.logger.info(f"Run {run + 1}/{num_runs}")
            
            # Clear cache before each run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Warmup with proper autocast
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    _ = self.model(**inputs)
            
            # Actual timing run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            generated_tokens = []
            token_times = []
            first_token_time = None
            
            # Generate tokens one by one for precise timing
            with torch.no_grad():
                current_input_ids = inputs["input_ids"]
                past_key_values = None
                
                for step in range(max_new_tokens):
                    step_start = time.time()
                    
                    # Use proper autocast syntax
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        if getattr(self.engine, 'supports_kv_cache', False) and past_key_values is not None:
                            # KV cache: only pass new token after first iteration
                            outputs = self.model(
                                input_ids=next_token.unsqueeze(0),  # Only the new token
                                attention_mask=None,  # Can be None with KV cache
                                past_key_values=past_key_values,
                                use_cache=True
                            )
                        else:
                            # First iteration or no KV cache: pass full sequence
                            outputs = self.model(
                                input_ids=current_input_ids,
                                attention_mask=inputs.get("attention_mask"),
                                use_cache=getattr(self.engine, 'supports_kv_cache', False)
                            )
                    
                    # Get logits and apply temperature
                    logits = outputs.logits[:, -1, :] / temperature
                    
                    # Apply top-p filtering
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
                    
                    # Record timing
                    step_end = time.time()
                    step_time = step_end - step_start
                    token_times.append(step_time)
                    
                    # Record first token time
                    if first_token_time is None:
                        first_token_time = step_time
                    
                    # Check for EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        self.logger.info(f"EOS token generated at step {step}")
                        break
                    
                    generated_tokens.append(next_token.item())
                    
                    # Prepare for next iteration
                    if getattr(self.engine, 'supports_kv_cache', False):
                        past_key_values = getattr(outputs, 'past_key_values', None)
                        # For KV cache, we'll use next_token in the next iteration
                    else:
                        # No KV cache: append token to sequence for next iteration
                        current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=1)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.time() - start_time
            
            # Calculate metrics
            num_generated_tokens = len(generated_tokens)
            if num_generated_tokens == 0:
                self.logger.warning("No tokens generated")
                continue
            
            # Time To First Token (TTFT)
            ttft = first_token_time if first_token_time is not None else 0
            
            # Time Per Output Token (TPOT)
            tpot = total_time / num_generated_tokens if num_generated_tokens > 0 else 0
            
            # Tokens per second
            tokens_per_second = num_generated_tokens / total_time if total_time > 0 else 0
            
            # Inter-Token Latency (ITL) - average time between tokens (exclude first token)
            # ITL should only measure time between subsequent tokens, not including first token generation
            if len(token_times) > 1:
                itl = statistics.mean(token_times[1:])  # Exclude first token time
                itl_std = statistics.stdev(token_times[1:])
            else:
                itl = 0
                itl_std = 0
            
            run_metrics = {
                'run': run + 1,
                'num_generated_tokens': num_generated_tokens,
                'total_time': total_time,
                'ttft': ttft,
                'tpot': tpot,
                'tokens_per_second': tokens_per_second,
                'itl_mean': itl,
                'itl_std': itl_std,
                'token_times': token_times,
                'generated_text': self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            }
            
            all_metrics.append(run_metrics)
            
            self.logger.info(f"Run {run + 1}: {num_generated_tokens} tokens, {tokens_per_second:.2f} tok/s, TTFT: {ttft:.4f}s, TPOT: {tpot:.4f}s")
            
            # Clear GPU memory after each run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate average metrics
        if not all_metrics:
            return {}
        
        avg_metrics = {
            'early_exit_threshold': early_exit_threshold,
            'num_runs': num_runs,
            'num_generated_tokens_avg': statistics.mean([m['num_generated_tokens'] for m in all_metrics]),
            'total_time_avg': statistics.mean([m['total_time'] for m in all_metrics]),
            'ttft_avg': statistics.mean([m['ttft'] for m in all_metrics]),
            'ttft_std': statistics.stdev([m['ttft'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'tpot_avg': statistics.mean([m['tpot'] for m in all_metrics]),
            'tpot_std': statistics.stdev([m['tpot'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'tokens_per_second_avg': statistics.mean([m['tokens_per_second'] for m in all_metrics]),
            'tokens_per_second_std': statistics.stdev([m['tokens_per_second'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'itl_mean_avg': statistics.mean([m['itl_mean'] for m in all_metrics]),
            'itl_mean_std': statistics.stdev([m['itl_mean'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'itl_std_avg': statistics.mean([m['itl_std'] for m in all_metrics]),
            'individual_runs': all_metrics
        }
        
        return avg_metrics
    
    def benchmark_early_exit_thresholds(
        self, 
        prompt: str,
        early_exit_thresholds: List[float] = [0.0, 0.25, 0.5, 0.75, 0.95],
        max_new_tokens: int = 100,
        num_runs: int = 5
    ) -> Dict:
        """
        Benchmark model with different early exit thresholds.
        
        Args:
            prompt: Input prompt
            early_exit_thresholds: List of early exit thresholds to test
            max_new_tokens: Maximum number of new tokens to generate
            num_runs: Number of runs per threshold
            
        Returns:
            Dictionary with results for all thresholds
        """
        self.logger.info("Starting comprehensive early exit benchmark")
        self.logger.info(f"Testing thresholds: {early_exit_thresholds}")
        
        all_results = {}
        
        for threshold in early_exit_thresholds:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing Early Exit Threshold: {threshold}")
            self.logger.info(f"{'='*60}")
            
            # Note: Early exit threshold is typically handled during model forward pass
            # For BitNet models, this would need to be implemented in the model's forward method
            # For now, we'll log the threshold but it won't affect generation
            self.logger.info(f"Testing with early exit threshold: {threshold} (note: may not be implemented in model)")
            
            # Run benchmark
            results = self.generate_with_timing(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                early_exit_threshold=threshold,
                num_runs=num_runs
            )
            
            if results:
                all_results[f'threshold_{threshold}'] = results
                
                # Log summary
                self.logger.info(f"\nResults for threshold {threshold}:")
                self.logger.info(f"  Tokens/sec: {results['tokens_per_second_avg']:.2f} ± {results['tokens_per_second_std']:.2f}")
                self.logger.info(f"  TTFT: {results['ttft_avg']:.4f}s ± {results['ttft_std']:.4f}s")
                self.logger.info(f"  TPOT: {results['tpot_avg']:.4f}s ± {results['tpot_std']:.4f}s")
                self.logger.info(f"  ITL: {results['itl_mean_avg']:.4f}s ± {results['itl_mean_std']:.4f}s")
            
            # Clear GPU memory after each threshold test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU memory cleared after threshold test")
        
        return all_results
    
    def run_comprehensive_benchmark(
        self,
        prompts: List[str],
        early_exit_thresholds: List[float] = [0.0, 0.25, 0.5, 0.75, 0.95],
        max_new_tokens: int = 100,
        num_runs: int = 3
    ) -> Dict:
        """
        Run comprehensive benchmark across multiple prompts and thresholds.
        
        Args:
            prompts: List of prompts to test
            early_exit_thresholds: List of early exit thresholds
            max_new_tokens: Maximum tokens to generate
            num_runs: Number of runs per prompt/threshold combination
            
        Returns:
            Complete benchmark results
        """
        self.logger.info("Starting comprehensive inference benchmark")
        self.logger.info(f"Prompts: {len(prompts)}")
        self.logger.info(f"Early exit thresholds: {early_exit_thresholds}")
        self.logger.info(f"Max new tokens: {max_new_tokens}")
        self.logger.info(f"Runs per combination: {num_runs}")
        
        all_results = {
            'benchmark_info': {
                'model_path': self.model_path,
                'device': self.device,
                'prompts': prompts,
                'early_exit_thresholds': early_exit_thresholds,
                'max_new_tokens': max_new_tokens,
                'num_runs': num_runs,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'prompt_results': {}
        }
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Testing Prompt {i+1}/{len(prompts)}")
            self.logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            self.logger.info(f"{'='*80}")
            
            prompt_results = self.benchmark_early_exit_thresholds(
                prompt=prompt,
                early_exit_thresholds=early_exit_thresholds,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs
            )
            
            all_results['prompt_results'][f'prompt_{i+1}'] = {
                'prompt': prompt,
                'results': prompt_results
            }
        
        return all_results
    
    def save_results(self, results: Dict, output_file: str = "inference_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("INFERENCE BENCHMARK SUMMARY (FP16 OPTIMIZED)")
        print("="*80)
        print(f"Model dtype: {self._model_dtype()}")
        print(f"Model device: {self._model_device()}")
        print(f"Model config: {self.model.config.hidden_size} hidden, {self.model.config.num_hidden_layers} layers")
        print("="*80)
        
        for prompt_key, prompt_data in results['prompt_results'].items():
            print(f"\n{prompt_key.upper()}:")
            print(f"Prompt: {prompt_data['prompt'][:100]}{'...' if len(prompt_data['prompt']) > 100 else ''}")
            print("-" * 60)
            
            for threshold_key, threshold_data in prompt_data['results'].items():
                threshold = threshold_data['early_exit_threshold']
                print(f"\nEarly Exit Threshold: {threshold}")
                print(f"  Tokens/sec: {threshold_data['tokens_per_second_avg']:.2f} ± {threshold_data['tokens_per_second_std']:.2f}")
                print(f"  TTFT:       {threshold_data['ttft_avg']:.4f}s ± {threshold_data['ttft_std']:.4f}s")
                print(f"  TPOT:       {threshold_data['tpot_avg']:.4f}s ± {threshold_data['tpot_std']:.4f}s")
                print(f"  ITL:        {threshold_data['itl_mean_avg']:.4f}s ± {threshold_data['itl_mean_std']:.4f}s")
                print(f"  Tokens:     {threshold_data['num_generated_tokens_avg']:.1f}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Inference Benchmark for 1B Improved BitNet Model'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model directory, model file (.pt/.safetensors), or Hugging Face repo ID (e.g., Ram07/bitnet-1b)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=1000,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of runs per prompt/threshold combination')
    parser.add_argument('--output_file', type=str, default='inference_benchmark_results.json',
                       help='Output file for results')
    
    return parser.parse_args()

def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The key to successful machine learning is",
        "When developing large language models,",
        "The most important aspect of neural networks is"
    ]
    
    # Early exit thresholds to test
    early_exit_thresholds = [0.0, 0.25, 0.5, 0.75]
    
    # Initialize benchmark
    benchmark = InferenceBenchmark(
        model_path=args.model_path,
        device=args.device
    )
    
    # Validate model path before starting (this will download if necessary)
    try:
        resolved_path = benchmark._validate_model_path(args.model_path)
        print(f"✅ Model path validated: {resolved_path}")
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return
    
    # Load model
    benchmark.load_model()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        prompts=test_prompts,
        early_exit_thresholds=early_exit_thresholds,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs
    )
    
    # Save results
    benchmark.save_results(results, args.output_file)
    
    # Print summary
    benchmark.print_summary(results)
    
    print(f"\nBenchmark completed! Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
