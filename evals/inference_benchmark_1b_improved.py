#!/usr/bin/env python3
"""
Complete Inference Benchmark Script for 1B BitNet Model with Early Exit Support
Measures TPOT, TTFT, tokens/sec, ITL, memory usage with proper timing and early exit
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

# Import the improved model
from bitnet.inference.engine import BitNetInferenceEngine

# Load environment variables
load_dotenv()

# Disable HuggingFace caching and Xet storage
os.environ["HF_DATASETS_CACHE"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
os.environ["HF_HOME"] = ""
os.environ["HF_HUB_DISABLE_XET"] = "1"


class EarlyExitGenerator:
    """
    Early exit generation wrapper that implements layer-wise early exit during inference.
    """
    
    def __init__(self, model, tokenizer, confidence_threshold: float = 0.95):
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.exit_layer_history = []
    
    def should_exit_early(self, hidden_states: torch.Tensor, layer_idx: int) -> Tuple[bool, float, torch.Tensor]:
        """
        Determine if we should exit early at this layer based on prediction confidence.
        
        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_size]
            layer_idx: Current layer index
            
        Returns:
            Tuple of (should_exit, confidence, logits)
        """
        # Never exit before layer 4 (need minimum computation)
        if layer_idx < 4:
            return False, 0.0, None
        
        # Get logits from current hidden state
        with torch.no_grad():
            # Apply layer norm (if model has it)
            if hasattr(self.model, 'layer_norm'):
                hidden_states = self.model.layer_norm(hidden_states)
            
            # Get logits
            logits = self.model.lm_head(hidden_states[:, -1, :])
            
            # Calculate confidence (max probability)
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max().item()
            
            # Decide whether to exit
            should_exit = confidence >= self.confidence_threshold
            
            return should_exit, confidence, logits
    
    def generate_with_early_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Generate tokens with early exit support.
        
        Returns:
            Tuple of (generated_tokens, exit_layers, confidences)
        """
        self.exit_layer_history = []
        generated_tokens = []
        exit_layers = []
        confidences = []
        
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        
        for step in range(max_new_tokens):
            # Try early exit at each layer
            exited_early = False
            exit_layer = None
            confidence = 0.0
            next_token_logits = None
            
            # Get embeddings
            with torch.no_grad():
                inputs_embeds = self.model.embed_tokens(current_input_ids)
                if hasattr(self.model, 'embed_positions'):
                    batch_size, seq_length = current_input_ids.shape
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=current_input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    position_embeddings = self.model.embed_positions(position_ids)
                    hidden_states = inputs_embeds + position_embeddings
                else:
                    hidden_states = inputs_embeds
                
                # Process through layers with early exit checks
                for layer_idx, layer in enumerate(self.model.layers):
                    # Forward through layer
                    layer_output = layer(
                        hidden_states,
                        attention_mask=current_attention_mask,
                        position_ids=None
                    )
                    
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                    else:
                        hidden_states = layer_output
                    
                    # Check if we should exit early
                    should_exit, conf, logits = self.should_exit_early(hidden_states, layer_idx)
                    
                    if should_exit:
                        exited_early = True
                        exit_layer = layer_idx
                        confidence = conf
                        next_token_logits = logits
                        break
                
                # If didn't exit early, use full model output
                if not exited_early:
                    # Final layer norm
                    hidden_states = self.model.layer_norm(hidden_states)
                    # Get logits
                    next_token_logits = self.model.lm_head(hidden_states[:, -1, :])
                    exit_layer = len(self.model.layers) - 1
                    probs = F.softmax(next_token_logits, dim=-1)
                    confidence = probs.max().item()
            
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
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Record results
            generated_tokens.append(next_token.item())
            exit_layers.append(exit_layer)
            confidences.append(confidence)
            
            # Update input for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            if current_attention_mask is not None:
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1),
                             dtype=current_attention_mask.dtype,
                             device=current_attention_mask.device)
                ], dim=1)
        
        return generated_tokens, exit_layers, confidences


class InferenceBenchmark:
    """Comprehensive inference benchmark for BitNet models with early exit support."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.engine = None
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
            
            # Check disk space
            import shutil
            free_space = shutil.disk_usage("./models").free
            self.logger.info(f"Available disk space: {free_space / (1024**3):.2f} GB")
            
            if free_space < 5 * (1024**3):
                self.logger.warning("Low disk space detected. Download may fail.")
            
            # Download with selective files
            self.logger.info("Starting download with selective file patterns...")
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir="./models",
                local_files_only=False,
                allow_patterns=["*.safetensors", "*.bin", "config.json", "tokenizer*", "*.txt"],
                ignore_patterns=["*.md", "*.yaml", "*.json", "*.git*", "*.png", "*.jpg", "*.jpeg"],
                max_workers=1,
                resume_download=True
            )
            
            self.logger.info(f"Model downloaded to: {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to download model from HF: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _validate_model_path(self, model_path: str) -> str:
        """Validate and resolve model path."""
        if self._is_huggingface_repo(model_path):
            self.logger.info(f"Detected Hugging Face repository: {model_path}")
            if "bitnet" in model_path.lower():
                self.logger.info("Attempting direct model loading from Hugging Face...")
                return model_path
            downloaded_path = self._download_from_huggingface(model_path)
            if downloaded_path:
                return downloaded_path
            else:
                self.logger.warning("Download failed, attempting direct loading...")
                return model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        if os.path.isdir(model_path):
            safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            if safetensors_files:
                self.logger.info(f"Found SafeTensors model: {safetensors_files[0]}")
                return model_path
            
            pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
            if pt_files:
                self.logger.info(f"Found PyTorch model: {pt_files[0]}")
                return model_path
            
            if os.path.exists(os.path.join(model_path, "config.json")):
                self.logger.info("Found model directory with config.json")
                return model_path
            
            raise FileNotFoundError(f"No valid model files found in directory: {model_path}")
        
        if os.path.isfile(model_path):
            if model_path.endswith(('.pt', '.safetensors')):
                self.logger.info(f"Found model file: {model_path}")
                return model_path
            else:
                raise ValueError(f"Unsupported model file format: {model_path}")
        
        raise ValueError(f"Invalid model path: {model_path}")

    def load_model(self):
        """Load the BitNet model and tokenizer."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            resolved_model_path = self._validate_model_path(self.model_path)
            self.logger.info(f"Resolved model path: {resolved_model_path}")
            
            if self._is_huggingface_repo(resolved_model_path):
                self.logger.info("Detected Hugging Face repository...")
                local_path = self._download_from_huggingface(resolved_model_path)
                if local_path:
                    self.logger.info(f"Model downloaded to: {local_path}")
                    model_file = local_path
                else:
                    self.logger.warning("Attempting direct loading...")
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to load tokenizer: {e}")
                            self.logger.info("Using fallback tokenizer")
                            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            resolved_model_path,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            device_map="auto" if self.device == "cuda" else None
                        )
                        
                        if self.device not in ["auto", "cpu"]:
                            self.model = self.model.to(self.device)
                        
                        self.logger.info("Model loaded successfully from Hugging Face")
                        return
                        
                    except Exception as e:
                        self.logger.error(f"Direct loading failed: {e}")
                        raise ValueError(f"Cannot load model {resolved_model_path}")
                
                model_file = local_path
            else:
                if os.path.isfile(resolved_model_path):
                    model_file = resolved_model_path
                else:
                    model_file = resolved_model_path
            
            tokenizer_name = os.getenv("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
            
            # Initialize engine
            self.engine = BitNetInferenceEngine(
                model_path=model_file,
                tokenizer_path=tokenizer_name,
                device=self.device,
                model_type="auto",
            )
            
            self.model = self.engine.model
            self.tokenizer = self.engine.tokenizer
            
            # Convert to FP16 for inference
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.half().cuda()
                self.logger.info("Model converted to FP16 for inference")
            
            self.logger.info("Model and tokenizer loaded successfully")
            self.logger.info(f"Model dtype: {self._model_dtype()}")
            self.logger.info(f"Model device: {self._model_device()}")
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def generate_with_timing(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        early_exit_threshold: float = 0.0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_runs: int = 5,
        use_early_exit: bool = False
    ) -> Dict:
        """
        Generate text with detailed timing measurements.
        
        Timing Definitions:
        - TTFT (Time To First Token): Time from start until first new token
        - Prefill Time: Approximate time spent processing the prompt
        - TPOT (Time Per Output Token): Average time per generated token
        - ITL (Inter-Token Latency): Average time between subsequent tokens
        - Decode Latency: Average time per token during generation (excluding prefill)
        """
        self.logger.info(f"Benchmarking with early_exit_threshold={early_exit_threshold}")
        
        # Tokenize input
        tok_out = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in tok_out.items()}
        
        # Ensure proper shapes
        if inputs["input_ids"].dim() == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if "attention_mask" in inputs and inputs["attention_mask"].dim() == 1:
            inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
        
        # Fix attention mask shape
        input_seq_len = inputs["input_ids"].shape[1]
        if "attention_mask" in inputs:
            attention_seq_len = inputs["attention_mask"].shape[1]
            if attention_seq_len != input_seq_len:
                if attention_seq_len < input_seq_len:
                    padding_length = input_seq_len - attention_seq_len
                    padding = torch.ones(
                        inputs["attention_mask"].shape[0], 
                        padding_length, 
                        dtype=inputs["attention_mask"].dtype, 
                        device=inputs["attention_mask"].device
                    )
                    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], padding], dim=1)
                else:
                    inputs["attention_mask"] = inputs["attention_mask"][:, :input_seq_len]
        
        # Create early exit generator if needed
        early_exit_gen = None
        if use_early_exit and early_exit_threshold > 0.0:
            early_exit_gen = EarlyExitGenerator(self.model, self.tokenizer, early_exit_threshold)
        
        all_metrics = []
        
        for run in range(num_runs):
            self.logger.info(f"Run {run + 1}/{num_runs}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Warmup (outside timing)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    _ = self.model(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Initialize tracking
            generated_tokens = []
            token_times = []
            exit_layers = []
            confidences = []
            
            # Start overall timing
            generation_start = time.time()
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Use early exit generator if enabled
            if early_exit_gen is not None:
                with torch.no_grad():
                    for step in range(max_new_tokens):
                        step_start = time.time()
                        
                        # Generate one token with early exit
                        tokens, layers, confs = early_exit_gen.generate_with_early_exit(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            max_new_tokens=1,
                            temperature=temperature,
                            top_p=top_p
                        )
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        step_end = time.time()
                        step_time = step_end - step_start
                        token_times.append(step_time)
                        
                        if not tokens:
                            break
                        
                        generated_tokens.extend(tokens)
                        exit_layers.extend(layers)
                        confidences.extend(confs)
                        
                        # Update inputs for next token
                        inputs["input_ids"] = torch.cat([
                            inputs["input_ids"],
                            torch.tensor([[tokens[0]]], device=self.device)
                        ], dim=1)
                        
                        if "attention_mask" in inputs:
                            inputs["attention_mask"] = torch.cat([
                                inputs["attention_mask"],
                                torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=self.device)
                            ], dim=1)
            else:
                # Standard generation without early exit
                with torch.no_grad():
                    current_input_ids = inputs["input_ids"]
                    current_attention_mask = inputs.get("attention_mask")
                    past_key_values = None
                    use_kv_cache = hasattr(self, 'engine') and getattr(self.engine, 'supports_kv_cache', False)
                    
                    for step in range(max_new_tokens):
                        step_start = time.time()
                        
                        # Prepare inputs
                        if use_kv_cache and past_key_values is not None:
                            step_input_ids = current_input_ids[:, -1:] if step > 0 else current_input_ids
                            step_attention_mask = None
                        else:
                            step_input_ids = current_input_ids
                            step_attention_mask = current_attention_mask
                        
                        # Forward pass
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            outputs = self.model(
                                input_ids=step_input_ids,
                                attention_mask=step_attention_mask,
                                past_key_values=past_key_values,
                                use_cache=use_kv_cache
                            )
                        
                        # Sample next token
                        logits = outputs.logits[:, -1, :] / temperature
                        
                        # Top-p filtering
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = float('-inf')
                        
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        step_end = time.time()
                        step_time = step_end - step_start
                        token_times.append(step_time)
                        
                        if next_token.item() == self.tokenizer.eos_token_id:
                            self.logger.info(f"EOS token generated at step {step}")
                            break
                        
                        generated_tokens.append(next_token.item())
                        exit_layers.append(len(self.model.layers) - 1)  # Full model
                        
                        # Update for next iteration
                        if use_kv_cache:
                            past_key_values = getattr(outputs, 'past_key_values', None)
                            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                        else:
                            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                            if current_attention_mask is not None:
                                current_attention_mask = torch.cat([
                                    current_attention_mask,
                                    torch.ones((current_attention_mask.shape[0], 1), 
                                             dtype=current_attention_mask.dtype,
                                             device=current_attention_mask.device)
                                ], dim=1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            generation_end = time.time()
            
            # Memory stats
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate metrics
            num_generated_tokens = len(generated_tokens)
            if num_generated_tokens == 0:
                self.logger.warning("No tokens generated")
                continue
            
            total_time = generation_end - generation_start
            
            # TTFT: Time to first token
            ttft = token_times[0] if token_times else 0
            
            # TPOT: Average time per output token
            tpot = total_time / num_generated_tokens if num_generated_tokens > 0 else 0
            
            # Tokens per second
            tokens_per_second = num_generated_tokens / total_time if total_time > 0 else 0
            
            # ITL: Inter-token latency (excluding first token)
            if len(token_times) > 1:
                itl_mean = statistics.mean(token_times[1:])
                itl_std = statistics.stdev(token_times[1:]) if len(token_times) > 2 else 0
                itl_min = min(token_times[1:])
                itl_max = max(token_times[1:])
            else:
                itl_mean = itl_std = itl_min = itl_max = 0
            
            # Decode latency (pure generation time, excluding prefill)
            decode_latency = itl_mean
            
            # Estimate prefill time
            avg_generation_time = decode_latency if decode_latency > 0 else tpot
            estimated_prefill_time = max(0, ttft - avg_generation_time)
            
            # Early exit stats
            avg_exit_layer = statistics.mean(exit_layers) if exit_layers else 0
            avg_confidence = statistics.mean(confidences) if confidences else 0
            
            run_metrics = {
                'run': run + 1,
                'num_generated_tokens': num_generated_tokens,
                'total_time': total_time,
                'ttft': ttft,
                'estimated_prefill_time': estimated_prefill_time,
                'decode_latency': decode_latency,
                'tpot': tpot,
                'tokens_per_second': tokens_per_second,
                'itl_mean': itl_mean,
                'itl_std': itl_std,
                'itl_min': itl_min,
                'itl_max': itl_max,
                'memory_allocated_mb': (memory_after - memory_before) / (1024**2),
                'peak_memory_mb': peak_memory / (1024**2),
                'avg_exit_layer': avg_exit_layer,
                'avg_confidence': avg_confidence,
                'token_times': token_times,
                'exit_layers': exit_layers,
                'confidences': confidences,
                'generated_text': self.tokenizer.decode(generated_tokens, skip_special_tokens=True),
                'used_early_exit': early_exit_gen is not None
            }
            
            all_metrics.append(run_metrics)
            
            self.logger.info(
                f"Run {run + 1}: {num_generated_tokens} tokens, "
                f"{tokens_per_second:.2f} tok/s, "
                f"TTFT: {ttft:.4f}s, "
                f"TPOT: {tpot:.4f}s, "
                f"ITL: {itl_mean:.4f}s, "
                f"Avg Exit Layer: {avg_exit_layer:.1f}"
            )
            
            # Clear GPU memory
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
            'total_time_std': statistics.stdev([m['total_time'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'ttft_avg': statistics.mean([m['ttft'] for m in all_metrics]),
            'ttft_std': statistics.stdev([m['ttft'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'estimated_prefill_time_avg': statistics.mean([m['estimated_prefill_time'] for m in all_metrics]),
            'decode_latency_avg': statistics.mean([m['decode_latency'] for m in all_metrics]),
            'decode_latency_std': statistics.stdev([m['decode_latency'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'tpot_avg': statistics.mean([m['tpot'] for m in all_metrics]),
            'tpot_std': statistics.stdev([m['tpot'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'tokens_per_second_avg': statistics.mean([m['tokens_per_second'] for m in all_metrics]),
            'tokens_per_second_std': statistics.stdev([m['tokens_per_second'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'itl_mean_avg': statistics.mean([m['itl_mean'] for m in all_metrics]),
            'itl_mean_std': statistics.stdev([m['itl_mean'] for m in all_metrics]) if len(all_metrics) > 1 else 0,
            'itl_std_avg': statistics.mean([m['itl_std'] for m in all_metrics]),
            'itl_min_avg': statistics.mean([m['itl_min'] for m in all_metrics]),
            'itl_max_avg': statistics.mean([m['itl_max'] for m in all_metrics]),
            'memory_allocated_mb_avg': statistics.mean([m['memory_allocated_mb'] for m in all_metrics]),
            'peak_memory_mb_avg': statistics.mean([m['peak_memory_mb'] for m in all_metrics]),
            'avg_exit_layer_avg': statistics.mean([m['avg_exit_layer'] for m in all_metrics]),
            'avg_confidence_avg': statistics.mean([m['avg_confidence'] for m in all_metrics]),
            'used_early_exit': all_metrics[0]['used_early_exit'],
            'individual_runs': all_metrics
        }
        
        return avg_metrics
    
    def benchmark_early_exit_thresholds(
        self, 
        prompt: str,
        early_exit_thresholds: List[float] = [0.0, 0.5, 0.75, 0.9, 0.95],
        max_new_tokens: int = 100,
        num_runs: int = 5
    ) -> Dict:
        """Benchmark model with different early exit thresholds."""
        self.logger.info("Starting comprehensive early exit benchmark")
        self.logger.info(f"Testing thresholds: {early_exit_thresholds}")
        
        all_results = {}
        
        for threshold in early_exit_thresholds:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing Early Exit Threshold: {threshold}")
            self.logger.info(f"{'='*60}")
            
            use_early_exit = threshold > 0.0
            
            results = self.generate_with_timing(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                early_exit_threshold=threshold,
                num_runs=num_runs,
                use_early_exit=use_early_exit
            )
            
            if results:
                all_results[f'threshold_{threshold}'] = results
                
                self.logger.info(f"\nResults for threshold {threshold}:")
                self.logger.info(f"  Tokens/sec: {results['tokens_per_second_avg']:.2f} ± {results['tokens_per_second_std']:.2f}")
                self.logger.info(f"  TTFT: {results['ttft_avg']:.4f}s ± {results['ttft_std']:.4f}s")
                self.logger.info(f"  TPOT: {results['tpot_avg']:.4f}s ± {results['tpot_std']:.4f}s")
                self.logger.info(f"  Decode Latency: {results['decode_latency_avg']:.4f}s ± {results['decode_latency_std']:.4f}s")
                self.logger.info(f"  Avg Exit Layer: {results['avg_exit_layer_avg']:.1f}")
                self.logger.info(f"  Peak Memory: {results['peak_memory_mb_avg']:.2f} MB")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_results
    
    def run_comprehensive_benchmark(
        self,
        prompts: List[str],
        early_exit_thresholds: List[float] = [0.0, 0.5, 0.75, 0.9, 0.95],
        max_new_tokens: int = 100,
        num_runs: int = 3
    ) -> Dict:
        """Run comprehensive benchmark across multiple prompts and thresholds."""
        self.logger.info("Starting comprehensive inference benchmark")
        self.logger.info(f"Prompts: {len(prompts)}")
        self.logger.info(f"Early exit thresholds: {early_exit_thresholds}")
        self.logger.info(f"Max new tokens: {max_new_tokens}")
        self.logger.info(f"Runs per combination: {num_runs}")
        
        all_results = {
            'benchmark_info': {
                'model_path': self.model_path,
                'device': self.device,
                'model_dtype': str(self._model_dtype()),
                'model_device': str(self._model_device()),
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
        print("INFERENCE BENCHMARK SUMMARY WITH EARLY EXIT")
        print("="*80)
        print(f"Model dtype: {results['benchmark_info']['model_dtype']}")
        print(f"Model device: {results['benchmark_info']['model_device']}")
        print("="*80)
        
        for prompt_key, prompt_data in results['prompt_results'].items():
            print(f"\n{prompt_key.upper()}:")
            print(f"Prompt: {prompt_data['prompt'][:100]}{'...' if len(prompt_data['prompt']) > 100 else ''}")
            print("-" * 80)
            
            for threshold_key, threshold_data in prompt_data['results'].items():
                threshold = threshold_data['early_exit_threshold']
                print(f"\nEarly Exit Threshold: {threshold}")
                print(f"  Tokens/sec:       {threshold_data['tokens_per_second_avg']:.2f} ± {threshold_data['tokens_per_second_std']:.2f}")
                print(f"  TTFT:             {threshold_data['ttft_avg']:.4f}s ± {threshold_data['ttft_std']:.4f}s")
                print(f"  Prefill Time:     {threshold_data['estimated_prefill_time_avg']:.4f}s")
                print(f"  Decode Latency:   {threshold_data['decode_latency_avg']:.4f}s ± {threshold_data['decode_latency_std']:.4f}s")
                print(f"  TPOT:             {threshold_data['tpot_avg']:.4f}s ± {threshold_data['tpot_std']:.4f}s")
                print(f"  ITL:              {threshold_data['itl_mean_avg']:.4f}s ± {threshold_data['itl_mean_std']:.4f}s")
                print(f"  ITL Range:        [{threshold_data['itl_min_avg']:.4f}s, {threshold_data['itl_max_avg']:.4f}s]")
                print(f"  Avg Exit Layer:   {threshold_data['avg_exit_layer_avg']:.1f}")
                print(f"  Avg Confidence:   {threshold_data['avg_confidence_avg']:.3f}")
                print(f"  Peak Memory:      {threshold_data['peak_memory_mb_avg']:.2f} MB")
                print(f"  Tokens Generated: {threshold_data['num_generated_tokens_avg']:.1f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Inference Benchmark for BitNet with Early Exit'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model or HuggingFace repo ID')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of runs per prompt/threshold combination')
    parser.add_argument('--output_file', type=str, default='inference_benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--early_exit_thresholds', type=float, nargs='+',
                       default=[0.00001],
                       help='List of early exit confidence thresholds to test')
    
    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
    ]
    
    # Initialize benchmark
    benchmark = InferenceBenchmark(
        model_path=args.model_path,
        device=args.device
    )
    
    # Validate model path
    try:
        resolved_path = benchmark._validate_model_path(args.model_path)
        print(f"Model path validated: {resolved_path}")
    except Exception as e:
        print(f"Model validation failed: {e}")
        return
    
    # Load model
    benchmark.load_model()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        prompts=test_prompts,
        early_exit_thresholds=args.early_exit_thresholds,
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