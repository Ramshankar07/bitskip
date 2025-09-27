"""
Inference engine for BitNet model with optimized generation.
"""

from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from ..modeling.model import BitNetModel
from ..modeling.model2 import BitNetModel2
from bitnet.utils.default_config import DefaultConfig


class BitNetInferenceEngine:
    """
    Inference engine for BitNet model with optimized generation.
    
    Args:
        model_path: Path to pretrained model
        tokenizer_path: Path to tokenizer
        device: Device to use for inference
        model_type: Type of model to load ('bitnet' or 'bitnet2')
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        model_type: str = "auto"
    ):
        self.device = device
        self.model_type = model_type
        
        # Load model configuration
        self.config = DefaultConfig()
        
        # Configure layer skipping for inference
        self.config.layer_skip_strategy = "fixed"  # Using fixed pattern is faster for inference
        
        # Load checkpoint
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            # If that fails, try with weights_only=False and add safe globals
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Determine model type if auto
        if model_type == "auto":
            # Check if it's a BitNetModel2 checkpoint by looking for specific keys
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if any("h_bitlinear" in key for key in state_dict.keys()):
                self.model_type = "bitnet2"
                print("Detected BitNetModel2 checkpoint")
            else:
                self.model_type = "bitnet"
                print("Detected BitNetModel checkpoint")
        
        # Load config from checkpoint if present, and convert to DefaultConfig if needed
        config_dict = checkpoint.get("config", None)
        if config_dict is not None and isinstance(config_dict, dict):
            # Only keep keys that are valid DefaultConfig fields
            valid_keys = set(DefaultConfig.__dataclass_fields__.keys())
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
            config = DefaultConfig.from_dict(filtered_config)
            print(f"Loaded config from checkpoint: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
        else:
            config = DefaultConfig()
            print("No config found in checkpoint, using default config")
        
        # Always infer model architecture from state dict to ensure compatibility
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if "embed_tokens.weight" in state_dict:
            vocab_size, hidden_size = state_dict["embed_tokens.weight"].shape
            config.vocab_size = vocab_size
            config.hidden_size = hidden_size
            config.max_position_embeddings = state_dict["embed_positions.weight"].shape[0]
            print(f"Updated config from state dict: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
        
        # Set other required config parameters if not present
        if not hasattr(config, 'num_hidden_layers') or config.num_hidden_layers is None:
            # Count layers from state dict
            layer_count = 0
            for key in state_dict.keys():
                if key.startswith('layers.') and '.self_attn.' in key:
                    layer_idx = int(key.split('.')[1])
                    layer_count = max(layer_count, layer_idx + 1)
            config.num_hidden_layers = layer_count
            print(f"Inferred num_hidden_layers: {config.num_hidden_layers}")
        
        if not hasattr(config, 'num_attention_heads') or config.num_attention_heads is None:
            # Infer from hidden size (assuming standard ratio)
            config.num_attention_heads = config.hidden_size // 64  # Common ratio
            print(f"Inferred num_attention_heads: {config.num_attention_heads}")
        
        # Always infer mlp_ratio from state dict to ensure compatibility
        for key in state_dict.keys():
            if 'feed_forward.up_proj.weight' in key:
                intermediate_size = state_dict[key].shape[0]
                config.mlp_ratio = intermediate_size / config.hidden_size
                print(f"Inferred mlp_ratio from state dict: {config.mlp_ratio} (intermediate_size={intermediate_size})")
                break
        else:
            # Fallback to standard ratio for smaller model (adjusted for power of 2)
            config.mlp_ratio = 2.0
            print(f"Using fallback mlp_ratio: {config.mlp_ratio}")
        
        # Debug: Print final config
        print(f"Final config - hidden_size: {config.hidden_size}, mlp_ratio: {config.mlp_ratio}")
        
        # Initialize the appropriate model
        if self.model_type == "bitnet2":
            self.model = BitNetModel2(config)
            print("Initialized BitNetModel2 (with H-BitLinear layers)")
        else:
            self.model = BitNetModel(config)
            print("Initialized BitNetModel")
        
        # Print model and checkpoint shapes for validation
        print("[Validation] Model shapes:")
        print("  embed_tokens.weight:", self.model.embed_tokens.weight.shape)
        print("  embed_positions.weight:", self.model.embed_positions.weight.shape)
        print("  lm_head.weight:", self.model.lm_head.weight.shape)
        print("[Validation] Checkpoint shapes:")
        
        # Get state dict from checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        print("  embed_tokens.weight:", state_dict["embed_tokens.weight"].shape)
        print("  embed_positions.weight:", state_dict["embed_positions.weight"].shape)
        print("  lm_head.weight:", state_dict["lm_head.weight"].shape)
        
        # Check for shape mismatches
        if (self.model.embed_tokens.weight.shape != state_dict["embed_tokens.weight"].shape or
            self.model.embed_positions.weight.shape != state_dict["embed_positions.weight"].shape or
            self.model.lm_head.weight.shape != state_dict["lm_head.weight"].shape):
            raise ValueError("Model and checkpoint shapes do not match! Please ensure you are using the same config as during training.")
        
        # Fix scalar to 1-element tensor for weight_scale keys
        for k, v in state_dict.items():
            if "weight_scale" in k and isinstance(v, torch.Tensor) and v.shape == torch.Size([]):
                state_dict[k] = v.view(1)
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Initialize KV cache
        self.kv_cache = None
    
    def generate(self, prompt, max_length=50, early_exit_threshold=None, temperature=1.0, top_p=1.0, repetition_penalty=1.0, debug=False):
        with torch.autograd.profiler.record_function("Engine.generate"):
            with torch.autograd.profiler.record_function("Engine.tokenize"):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            prompt_len = input_ids.shape[1]
            actual_max_length = min(max_length, self.config.max_position_embeddings - prompt_len - 1)
            if actual_max_length < max_length and debug:
                print(f"Reducing max_length from {max_length} to {actual_max_length} due to position limit")
            if actual_max_length <= 0:
                if debug:
                    print(f"Warning: Cannot generate any tokens. Prompt length {prompt_len} >= max positions {self.config.max_position_embeddings}")
                return prompt
            generated = input_ids
            past_key_values = None
            input_length = input_ids.shape[1]
            seen_tokens = set()
            with torch.autograd.profiler.record_function("Engine.generation_loop"):
                for step in range(actual_max_length):
                    if step == 0:
                        model_inputs = generated
                        current_position_ids = torch.arange(input_length, dtype=torch.long, device=self.device).unsqueeze(0)
                    else:
                        model_inputs = generated[:, -1:]
                        current_position_ids = torch.tensor([[input_length + step - 1]], dtype=torch.long, device=self.device)
                    current_attention_mask = torch.ones_like(model_inputs)
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=model_inputs,
                            attention_mask=current_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True,
                            return_dict=True,
                            position_ids=current_position_ids
                        )
                        past_key_values = outputs.past_key_values
                        logits = None
                        max_prob = None
                        
                        # Handle early exit for both model types
                        if outputs.hidden_states is not None:
                            for layer_idx, hidden in enumerate(outputs.hidden_states):
                                if hidden is None:
                                    continue
                                layer_logits = self.model.lm_head(hidden[:, -1, :])
                                probs = torch.softmax(layer_logits, dim=-1)
                                layer_max_prob, _ = torch.max(probs, dim=-1)
                                if early_exit_threshold is not None and (layer_max_prob > early_exit_threshold).any().item():
                                    logits = layer_logits
                                    max_prob = layer_max_prob
                                    if debug:
                                        print(f"Early exit at layer {layer_idx} with confidence {max_prob.item():.4f}")
                                    break
                        
                        if logits is None:
                            logits = outputs.logits[:, -1, :]
                            probs = torch.softmax(logits, dim=-1)
                            max_prob, _ = torch.max(probs, dim=-1)
                        
                        # Repetition penalty
                        if repetition_penalty != 1.0 and step > 0:
                            for token_id in set(generated[0].tolist()):
                                logits[0, token_id] /= repetition_penalty
                        
                        # Top-p (nucleus) sampling
                        if temperature > 0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            if sorted_indices_to_remove[..., 1:].any().item():
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                            
                            # Use a safer approach to mask logits
                            mask = torch.ones_like(logits, dtype=torch.bool)
                            for i in range(sorted_indices.shape[0]):
                                for j in range(sorted_indices.shape[1]):
                                    if sorted_indices_to_remove[i, j]:
                                        idx = sorted_indices[i, j]
                                        if idx < logits.shape[-1]:
                                            mask[i, idx] = False
                            
                            logits = logits.masked_fill(~mask, float('-inf'))
                            probs = torch.softmax(logits / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        generated = torch.cat([generated, next_token], dim=1)
                        if debug:
                            print(f"Step {step}: next_token={next_token.item()}, eos_token_id={self.tokenizer.eos_token_id}")
                            if early_exit_threshold is not None and max_prob is not None:
                                print(f"Step {step}: max_prob={max_prob.item()}, threshold={early_exit_threshold}")
                            print(f"Step {step}: generated sequence: {self.tokenizer.decode(generated[0], skip_special_tokens=True)}")
                        if input_length + step + 1 >= self.config.max_position_embeddings:
                            if debug:
                                print(f"Warning: Reached maximum position embeddings limit ({self.config.max_position_embeddings})")
                            break
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
            output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return output_text
    
    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache = None 