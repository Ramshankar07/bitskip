"""
LayerSkip with Dynamic Confidence-Based Early Exit

This implementation allows the model to exit at different layers based on 
confidence thresholds, rather than using a fixed exit layer.

Confidence metrics:
1. Max probability (softmax confidence)
2. Entropy-based confidence
3. Margin-based confidence (difference between top-2 predictions)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import time
import statistics
import logging
import numpy as np


class ConfidenceMetrics:
    """Calculate various confidence metrics for early exit decisions."""
    
    @staticmethod
    def max_probability(logits: torch.Tensor) -> float:
        """
        Maximum softmax probability.
        Higher is more confident.
        Range: [0, 1]
        """
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max().item()
        return max_prob
    
    @staticmethod
    def entropy(logits: torch.Tensor) -> float:
        """
        Entropy of the probability distribution.
        Lower entropy = more confident.
        Normalized to [0, 1] where 1 = most confident.
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        # Normalize by max possible entropy
        vocab_size = logits.shape[-1]
        max_entropy = np.log(vocab_size)
        
        # Convert to confidence (invert entropy)
        confidence = 1.0 - (entropy / max_entropy)
        return confidence
    
    @staticmethod
    def margin(logits: torch.Tensor) -> float:
        """
        Margin between top-2 predictions.
        Larger margin = more confident.
        Normalized to [0, 1].
        """
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1)
        margin = (top2.values[0] - top2.values[1]).item()
        
        # Margin is in [0, 1] already
        return margin
    
    @staticmethod
    def calculate_all(logits: torch.Tensor) -> Dict[str, float]:
        """Calculate all confidence metrics."""
        return {
            'max_prob': ConfidenceMetrics.max_probability(logits),
            'entropy_conf': ConfidenceMetrics.entropy(logits),
            'margin': ConfidenceMetrics.margin(logits)
        }


class DynamicEarlyExitBenchmark:
    """
    LayerSkip with dynamic confidence-based early exit.
    
    The model checks confidence at each layer and exits when
    confidence exceeds threshold, rather than using a fixed exit layer.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        device: str = "cuda",
        confidence_metric: str = "max_prob"
    ):
        """
        Args:
            model: LayerSkip model
            tokenizer: Tokenizer
            device: Device to use
            confidence_metric: Metric to use for confidence
                Options: 'max_prob', 'entropy_conf', 'margin'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.confidence_metric = confidence_metric
        self.logger = logging.getLogger(__name__)
        
        # Get number of layers
        self.num_layers = self._get_num_layers()
        self.logger.info(f"Model has {self.num_layers} layers")
        self.logger.info(f"Using confidence metric: {confidence_metric}")
    
    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.model, 'config'):
            return getattr(self.model.config, 'num_hidden_layers', 32)
        return 32
    
    def _project_to_vocab(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits using LM head."""
        # Apply final layer norm (if model has it)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            hidden_states = self.model.model.norm(hidden_states)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
            hidden_states = self.model.transformer.ln_f(hidden_states)
        
        # Project through LM head
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(hidden_states)
        elif hasattr(self.model, 'score'):
            logits = self.model.score(hidden_states)
        else:
            raise AttributeError("Model doesn't have lm_head")
        
        return logits
    
    def _calculate_confidence(
        self, 
        logits: torch.Tensor,
        metric: Optional[str] = None
    ) -> float:
        """
        Calculate confidence score for logits.
        
        Args:
            logits: Model logits
            metric: Confidence metric to use (None = use default)
            
        Returns:
            Confidence score in [0, 1]
        """
        if metric is None:
            metric = self.confidence_metric
        
        # Get confidence
        if metric == "max_prob":
            return ConfidenceMetrics.max_probability(logits)
        elif metric == "entropy_conf":
            return ConfidenceMetrics.entropy(logits)
        elif metric == "margin":
            return ConfidenceMetrics.margin(logits)
        else:
            raise ValueError(f"Unknown confidence metric: {metric}")
    
    def generate_with_dynamic_exit(
        self,
        prompt: str,
        confidence_threshold: float,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_exit_layer: int = 0,
        max_exit_layer: Optional[int] = None
    ) -> Dict:
        """
        Generate tokens with dynamic confidence-based early exit.
        
        Args:
            prompt: Input prompt
            confidence_threshold: Confidence threshold for early exit (0-1)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            min_exit_layer: Minimum layer to consider for exit
            max_exit_layer: Maximum layer to consider (None = last layer)
            
        Returns:
            Generation results with exit statistics
        """
        if max_exit_layer is None:
            max_exit_layer = self.num_layers - 1
        
        # Validate parameters
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if min_exit_layer < 0 or min_exit_layer >= self.num_layers:
            raise ValueError(f"min_exit_layer must be in [0, {self.num_layers-1}]")
        if max_exit_layer < min_exit_layer or max_exit_layer >= self.num_layers:
            raise ValueError(f"max_exit_layer must be in [{min_exit_layer}, {self.num_layers-1}]")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        generated_tokens = []
        token_times = []
        exit_layers = []  # Track which layer each token exited from
        exit_confidences = []  # Track confidence at exit
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_start = time.time()
                
                # Run full forward pass to get all hidden states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Check each layer for confidence threshold
                exited = False
                exit_layer = max_exit_layer  # Default to last layer
                exit_confidence = 0.0
                
                for layer_idx in range(min_exit_layer, max_exit_layer + 1):
                    # Get hidden states at this layer
                    # Note: hidden_states[0] is embedding, hidden_states[i+1] is layer i output
                    hidden_state = outputs.hidden_states[layer_idx + 1]
                    
                    # Project to vocabulary
                    logits = self._project_to_vocab(hidden_state)
                    
                    # Get logits for last position
                    next_token_logits = logits[:, -1, :]
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(next_token_logits)
                    
                    # Check if we can exit early
                    if confidence >= confidence_threshold:
                        exit_layer = layer_idx
                        exit_confidence = confidence
                        exited = True
                        break
                
                # If we didn't exit early, use last layer
                if not exited:
                    hidden_state = outputs.hidden_states[-1]
                    logits = self._project_to_vocab(hidden_state)
                    next_token_logits = logits[:, -1, :]
                    exit_confidence = self._calculate_confidence(next_token_logits)
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Synchronize after generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_end = time.time()
                
                # Record metrics
                token_times.append(step_end - step_start)
                exit_layers.append(exit_layer)
                exit_confidences.append(exit_confidence)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                
                # Append to input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1),
                                                   dtype=attention_mask.dtype,
                                                   device=attention_mask.device)],
                        dim=1
                    )
        
        # Calculate metrics
        num_tokens = len(generated_tokens)
        if num_tokens == 0:
            return {}
        
        total_time = sum(token_times)
        
        # Calculate exit layer distribution
        from collections import Counter
        exit_distribution = Counter(exit_layers)
        exit_dist_pct = {
            layer: count / len(exit_layers)
            for layer, count in exit_distribution.items()
        }
        
        return {
            'confidence_threshold': confidence_threshold,
            'confidence_metric': self.confidence_metric,
            'num_tokens': num_tokens,
            'generated_text': self.tokenizer.decode(generated_tokens, skip_special_tokens=True),
            'token_times': token_times,
            'exit_layers': exit_layers,
            'exit_confidences': exit_confidences,
            'avg_exit_layer': statistics.mean(exit_layers),
            'std_exit_layer': statistics.stdev(exit_layers) if len(exit_layers) > 1 else 0,
            'min_exit_layer': min(exit_layers),
            'max_exit_layer': max(exit_layers),
            'avg_confidence': statistics.mean(exit_confidences),
            'exit_distribution': exit_dist_pct,
            'total_time': total_time,
            'ttft_ms': token_times[0] * 1000 if token_times else 0,
            'avg_tpot_ms': (total_time / num_tokens) * 1000,
            'tokens_per_second': num_tokens / total_time,
            'itl_mean_ms': statistics.mean(token_times[1:]) * 1000 if len(token_times) > 1 else 0
        }
    
    def benchmark_confidence_thresholds(
        self,
        prompt: str,
        confidence_thresholds: List[float],
        max_new_tokens: int = 100,
        num_runs: int = 5
    ) -> Dict:
        """
        Benchmark different confidence thresholds.
        
        Args:
            prompt: Input prompt
            confidence_thresholds: List of thresholds to test (e.g., [0.5, 0.7, 0.9, 0.95])
            max_new_tokens: Maximum tokens per generation
            num_runs: Number of runs per threshold
            
        Returns:
            Results for all thresholds
        """
        self.logger.info(
            f"Benchmarking {len(confidence_thresholds)} confidence thresholds "
            f"with {num_runs} runs each"
        )
        
        all_results = {}
        
        for threshold in confidence_thresholds:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing Confidence Threshold: {threshold:.2f}")
            self.logger.info(f"{'='*60}")
            
            run_results = []
            
            for run in range(num_runs):
                try:
                    result = self.generate_with_dynamic_exit(
                        prompt=prompt,
                        confidence_threshold=threshold,
                        max_new_tokens=max_new_tokens
                    )
                    
                    if result:
                        run_results.append(result)
                        self.logger.info(
                            f"  Run {run+1}: {result['tokens_per_second']:.2f} tok/s, "
                            f"Avg exit layer: {result['avg_exit_layer']:.1f}, "
                            f"Avg confidence: {result['avg_confidence']:.3f}"
                        )
                except Exception as e:
                    self.logger.error(f"  Run {run+1} failed: {e}")
                    continue
            
            if run_results:
                # Calculate averages
                all_results[f'threshold_{threshold:.2f}'] = {
                    'threshold': threshold,
                    'num_runs': len(run_results),
                    'avg_tokens_per_second': statistics.mean([r['tokens_per_second'] for r in run_results]),
                    'std_tokens_per_second': statistics.stdev([r['tokens_per_second'] for r in run_results]) if len(run_results) > 1 else 0,
                    'avg_exit_layer': statistics.mean([r['avg_exit_layer'] for r in run_results]),
                    'std_exit_layer': statistics.stdev([r['avg_exit_layer'] for r in run_results]) if len(run_results) > 1 else 0,
                    'avg_confidence': statistics.mean([r['avg_confidence'] for r in run_results]),
                    'avg_ttft_ms': statistics.mean([r['ttft_ms'] for r in run_results]),
                    'avg_tpot_ms': statistics.mean([r['avg_tpot_ms'] for r in run_results]),
                    'exit_distribution': run_results[0]['exit_distribution'],
                    'sample_output': run_results[0]['generated_text'][:200],
                    'individual_runs': run_results
                }
                
                self.logger.info(
                    f"\nThreshold {threshold:.2f} Summary:\n"
                    f"  Throughput: {all_results[f'threshold_{threshold:.2f}']['avg_tokens_per_second']:.2f} tok/s\n"
                    f"  Avg Exit Layer: {all_results[f'threshold_{threshold:.2f}']['avg_exit_layer']:.1f}\n"
                    f"  Exit Distribution: {all_results[f'threshold_{threshold:.2f}']['exit_distribution']}"
                )
        
        return all_results
    
    def compare_confidence_metrics(
        self,
        prompt: str,
        threshold: float = 0.8,
        max_new_tokens: int = 50,
        num_runs: int = 3
    ) -> Dict:
        """
        Compare different confidence metrics.
        
        Tests max_prob, entropy_conf, and margin metrics.
        """
        metrics = ['max_prob', 'entropy_conf', 'margin']
        results = {}
        
        for metric in metrics:
            self.logger.info(f"\nTesting metric: {metric}")
            
            # Temporarily change metric
            original_metric = self.confidence_metric
            self.confidence_metric = metric
            
            # Run benchmark
            run_results = []
            for run in range(num_runs):
                result = self.generate_with_dynamic_exit(
                    prompt=prompt,
                    confidence_threshold=threshold,
                    max_new_tokens=max_new_tokens
                )
                if result:
                    run_results.append(result)
            
            # Restore original metric
            self.confidence_metric = original_metric
            
            if run_results:
                results[metric] = {
                    'avg_tokens_per_second': statistics.mean([r['tokens_per_second'] for r in run_results]),
                    'avg_exit_layer': statistics.mean([r['avg_exit_layer'] for r in run_results]),
                    'avg_confidence': statistics.mean([r['avg_confidence'] for r in run_results])
                }
        
        # Print comparison
        print("\n" + "="*80)
        print("CONFIDENCE METRIC COMPARISON")
        print("="*80)
        print(f"{'Metric':<20} {'Tok/s':<15} {'Avg Exit Layer':<20} {'Avg Confidence':<15}")
        print("-"*80)
        
        for metric, metrics_data in results.items():
            print(
                f"{metric:<20} "
                f"{metrics_data['avg_tokens_per_second']:>7.2f}        "
                f"{metrics_data['avg_exit_layer']:>7.1f}             "
                f"{metrics_data['avg_confidence']:>7.3f}"
            )
        print("="*80)
        
        return results


# Example usage
def main():
    """Example of dynamic confidence-based early exit."""
    import argparse
    import logging
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Dynamic Confidence-Based Early Exit Benchmark")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="Number of runs per threshold")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model_name = "facebook/layerskip-llama3.2-1B"
    print(f"Loading model: {model_name}")
    print(f"Device: {args.device}")
    print(f"Number of runs: {args.num_runs}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure padding token exists for padding=True calls
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.bos_token
        try:
            tokenizer.padding_side = 'left'
        except Exception:
            pass
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None
    )
    
    if args.device == "cpu":
        model = model.to("cpu")
    
    # Create benchmark
    benchmark = DynamicEarlyExitBenchmark(
        model, 
        tokenizer, 
        device=args.device,
        confidence_metric="max_prob"  # or "entropy_conf" or "margin"
    )
    
    # Test different confidence thresholds
    prompt = "The future of artificial intelligence is"
    confidence_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    
    results = benchmark.benchmark_confidence_thresholds(
        prompt=prompt,
        confidence_thresholds=confidence_thresholds,
        max_new_tokens=100,
        num_runs=args.num_runs
    )
    
    print("\nDynamic early exit benchmark completed!")
    
    # Compare different confidence metrics
    print("\nComparing confidence metrics...")
    metric_comparison = benchmark.compare_confidence_metrics(
        prompt=prompt,
        threshold=0.8,
        max_new_tokens=50,
        num_runs=args.num_runs
    )


if __name__ == "__main__":
    main()