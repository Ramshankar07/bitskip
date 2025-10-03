"""
Inference engine for BitNet model with optimized generation and proper SafeTensors loading.
"""

from typing import Dict, List, Optional, Union, Set
import os
import json

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file as load_safetensors

from ..modeling.model import BitNetModel
from ..modeling.model2 import BitNetModel2
from bitnet.utils.default_config import DefaultConfig


class BitNetInferenceEngine:
    """
    Inference engine for BitNet model with optimized generation.
    
    Args:
        model_path: Path to pretrained model (directory or .safetensors file)
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
        self.config.layer_skip_strategy = "fixed"
        
        # Resolve model file path
        resolved_path = self._resolve_model_path(model_path)
        print(f"Loading model from: {resolved_path}")
        
        # Load configuration from config.json if available
        self._load_config_json(resolved_path)
        
        # Load checkpoint (only SafeTensors format)
        print(f"Loading SafeTensors checkpoint...")
        state_dict = load_safetensors(resolved_path, device=self.device)
        print(f"Loaded {len(state_dict)} tensors from checkpoint")
        
        # Debug: Print ALL keys grouped by category to understand structure
        print("\n=== CHECKPOINT KEY ANALYSIS ===")
        
        # Group keys by prefix
        key_groups = {}
        for key in sorted(state_dict.keys()):
            # Get first two components for better grouping
            parts = key.split('.')
            if len(parts) >= 2:
                prefix = f"{parts[0]}.{parts[1]}"
            else:
                prefix = parts[0] if parts else key
            
            if prefix not in key_groups:
                key_groups[prefix] = []
            key_groups[prefix].append(key)
        
        print(f"\nKey prefixes found ({len(key_groups)} groups):")
        
        # Show sample keys from each group
        for prefix, keys in sorted(key_groups.items()):
            print(f"\n{prefix}.* ({len(keys)} keys):")
            # Show all keys if <= 5, otherwise show first 5
            display_keys = keys if len(keys) <= 5 else keys[:5]
            for key in display_keys:
                print(f"  {key}: {state_dict[key].shape}")
            if len(keys) > 5:
                print(f"  ... and {len(keys) - 5} more")
        
        # Critical check: Look for actual model weights
        has_embed_tokens = any('embed_tokens' in k for k in state_dict.keys())
        has_q_proj = any('q_proj.weight' in k for k in state_dict.keys())
        has_up_proj = any('up_proj.weight' in k for k in state_dict.keys())
        has_lm_head = any('lm_head.weight' in k for k in state_dict.keys())
        
        print("\n=== CRITICAL WEIGHT CHECK ===")
        print(f"  Has embed_tokens: {has_embed_tokens}")
        print(f"  Has attention weights (q_proj): {has_q_proj}")
        print(f"  Has FFN weights (up_proj): {has_up_proj}")
        print(f"  Has lm_head: {has_lm_head}")
        
        if not (has_embed_tokens and has_q_proj and has_up_proj and has_lm_head):
            print("\n⚠️  WARNING: Checkpoint appears incomplete!")
            print("  This checkpoint is missing critical model weights.")
            print("  It only contains layer norms and auxiliary components.")
            print("  You need a complete checkpoint with BitLinear weights to run inference.")
        
        print("\n=== END CHECKPOINT ANALYSIS ===\n")
        
        # Normalize state dict keys (remove training wrapper prefixes)
        print("\nBefore normalization - sample keys:")
        for key in list(state_dict.keys())[:3]:
            print(f"  {key}")
        
        state_dict = self._normalize_state_dict(state_dict)
        
        print("\nAfter normalization - sample keys:")
        for key in list(state_dict.keys())[:3]:
            print(f"  {key}")
        
        # Auto-detect model type
        if model_type == "auto":
            self.model_type = self._detect_model_type(state_dict)
            print(f"Auto-detected model type: {self.model_type}")
        
        # Infer model architecture from state dict
        self._infer_config_from_state_dict(state_dict)
        
        # Print final configuration
        print(f"\nFinal Configuration:")
        print(f"  vocab_size: {self.config.vocab_size}")
        print(f"  hidden_size: {self.config.hidden_size}")
        print(f"  num_hidden_layers: {self.config.num_hidden_layers}")
        print(f"  num_attention_heads: {self.config.num_attention_heads}")
        print(f"  max_position_embeddings: {self.config.max_position_embeddings}")
        print(f"  mlp_ratio: {self.config.mlp_ratio}")
        
        # Initialize the appropriate model
        self.model, self.supports_kv_cache = self._initialize_model()
        
        # Get expected model keys for debugging
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        print(f"\nKey Analysis:")
        print(f"  Model expects: {len(model_keys)} keys")
        print(f"  Checkpoint has: {len(checkpoint_keys)} keys")
        
        # Remap state dict keys to match model architecture
        remapped_state_dict = self._smart_remap_state_dict(state_dict, model_keys)
        
        # Validate shapes before loading
        self._validate_shapes(remapped_state_dict)
        
        # Fix scalar weight_scale tensors
        remapped_state_dict = self._fix_weight_scales(remapped_state_dict)
        
        # Load state dict into model
        print("\nLoading state dict into model...")
        missing_keys, unexpected_keys = self.model.load_state_dict(remapped_state_dict, strict=False)
        
        # Report any issues
        self._report_loading_issues(missing_keys, unexpected_keys)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        # Load tokenizer
        print(f"\nLoading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Tokenizer loaded: vocab_size={len(self.tokenizer)}")
        
        # Initialize KV cache
        self.kv_cache = None
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve the model path to a .safetensors file."""
        if os.path.isfile(model_path):
            if model_path.endswith('.safetensors'):
                return model_path
            else:
                raise ValueError(f"Only .safetensors format is supported, got: {model_path}")
        
        if os.path.isdir(model_path):
            # Look for model.safetensors first, then any .safetensors file
            preferred_names = ['model.safetensors', 'pytorch_model.safetensors']
            for name in preferred_names:
                candidate = os.path.join(model_path, name)
                if os.path.exists(candidate):
                    return candidate
            
            # Find any .safetensors file
            safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            if safetensors_files:
                return os.path.join(model_path, safetensors_files[0])
            
            raise FileNotFoundError(f"No .safetensors file found in directory: {model_path}")
        
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    def _load_config_json(self, resolved_path: str) -> None:
        """Load configuration from config.json if available."""
        model_dir = os.path.dirname(resolved_path) if os.path.isfile(resolved_path) else resolved_path
        config_json_path = os.path.join(model_dir, "config.json")
        
        if os.path.exists(config_json_path):
            try:
                with open(config_json_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Filter to only valid config fields
                valid_keys = set(DefaultConfig.__dataclass_fields__.keys())
                filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
                
                # Update config
                self.config = DefaultConfig.from_dict(filtered_config)
                print(f"Loaded config from config.json")
            except Exception as e:
                print(f"Warning: Failed to parse config.json: {e}")
    
    def _normalize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize state dict by removing common prefixes from training wrappers."""
        def strip_prefix(name: str) -> str:
            # Remove common training wrapper prefixes
            prefixes = ["model.", "module.", "_orig_mod."]
            for prefix in prefixes:
                if name.startswith(prefix):
                    return name[len(prefix):]
            return name
        
        normalized = {}
        for key, value in state_dict.items():
            new_key = strip_prefix(key)
            normalized[new_key] = value
        
        print(f"Normalized {len(state_dict)} keys (removed training prefixes)")
        return normalized
    
    def _detect_model_type(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect whether this is BitNetModel or BitNetModel2."""
        # BitNetModel2 uses H-BitLinear layers
        for key in state_dict.keys():
            if "h_bitlinear" in key.lower() or "hadamard" in key.lower():
                return "bitnet2"
        return "bitnet"
    
    def _infer_config_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Infer model configuration from the state dict."""
        # Helper to find a key among candidates
        def find_key(candidates: List[str]) -> Optional[str]:
            for key in candidates:
                if key in state_dict:
                    return key
            return None
        
        # Find embedding layers
        embed_tokens_key = find_key([
            "embed_tokens.weight",
            "tok_embeddings.weight",
            "wte.weight",
            "transformer.wte.weight"
        ])
        
        embed_pos_key = find_key([
            "embed_positions.weight",
            "wpe.weight",
            "position_embeddings.weight",
            "transformer.wpe.weight"
        ])
        
        lm_head_key = find_key([
            "lm_head.weight",
            "output.weight",
            "lm_head.linear.weight"
        ])
        
        # Infer vocab_size and hidden_size from embeddings
        if embed_tokens_key:
            vocab_size, hidden_size = state_dict[embed_tokens_key].shape
            self.config.vocab_size = vocab_size
            self.config.hidden_size = hidden_size
            print(f"Inferred from embeddings: vocab_size={vocab_size}, hidden_size={hidden_size}")
        
        # Infer max_position_embeddings
        if embed_pos_key:
            max_pos = state_dict[embed_pos_key].shape[0]
            self.config.max_position_embeddings = max_pos
            print(f"Inferred max_position_embeddings: {max_pos}")
        
        # Count number of layers - look for various patterns
        layer_count = 0
        layer_patterns = ['layers.', 'h.', 'transformer.h.', 'blocks.']
        
        for key in state_dict.keys():
            for pattern in layer_patterns:
                if pattern in key:
                    # Extract layer index
                    parts = key.split('.')
                    for i, part in enumerate(parts):
                        if part == pattern.rstrip('.'):
                            if i + 1 < len(parts) and parts[i + 1].isdigit():
                                layer_idx = int(parts[i + 1])
                                layer_count = max(layer_count, layer_idx + 1)
                                break
        
        if layer_count > 0:
            self.config.num_hidden_layers = layer_count
            print(f"Inferred num_hidden_layers: {layer_count}")
        
        # Infer num_attention_heads from attention weight shapes
        attention_keys = [k for k in state_dict.keys() if 'q_proj.weight' in k or 'attn.c_attn.weight' in k]
        if attention_keys:
            key = attention_keys[0]
            q_proj_shape = state_dict[key].shape
            # q_proj output size should be hidden_size (all heads)
            if q_proj_shape[0] == self.config.hidden_size:
                # Standard: assume head_dim = 64 or 128
                for head_dim in [64, 128, 96]:
                    if self.config.hidden_size % head_dim == 0:
                        self.config.num_attention_heads = self.config.hidden_size // head_dim
                        print(f"Inferred num_attention_heads: {self.config.num_attention_heads} (head_dim={head_dim})")
                        break
        
        # Infer num_kv_heads (if GQA is used)
        kv_keys = [k for k in state_dict.keys() if 'k_proj.weight' in k]
        if kv_keys:
            key = kv_keys[0]
            k_proj_shape = state_dict[key].shape
            kv_hidden_size = k_proj_shape[0]
            
            # Infer head_dim first
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            
            # Calculate num_kv_heads
            num_kv_heads = kv_hidden_size // head_dim
            self.config.num_kv_heads = num_kv_heads
            print(f"Inferred num_kv_heads: {num_kv_heads} (GQA enabled)")
        
        # Infer mlp_ratio from feed-forward layers
        mlp_keys = [k for k in state_dict.keys() if 'up_proj.weight' in k or 'mlp.c_fc.weight' in k]
        if mlp_keys:
            key = mlp_keys[0]
            intermediate_size = state_dict[key].shape[0]
            mlp_ratio = intermediate_size / self.config.hidden_size
            self.config.mlp_ratio = mlp_ratio
            print(f"Inferred mlp_ratio: {mlp_ratio:.2f} (intermediate_size={intermediate_size})")
        else:
            # Fallback: standard ratio for smaller models
            self.config.mlp_ratio = 2.0
            print(f"Using default mlp_ratio: 2.0")
    
    def _initialize_model(self) -> tuple:
        """Initialize the appropriate model based on type."""
        if self.model_type == "bitnet2":
            model = BitNetModel2(self.config)
            print("Initialized BitNetModel2 (with H-BitLinear layers)")
            supports_kv_cache = True
        else:
            model = BitNetModel(self.config)
            print("Initialized BitNetModel")
            supports_kv_cache = False
        
        return model, supports_kv_cache
    
    def _smart_remap_state_dict(
        self, 
        state_dict: Dict[str, torch.Tensor],
        model_keys: Set[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Intelligently remap state dict keys to match model architecture.
        
        This function tries multiple strategies to match checkpoint keys with model keys.
        """
        remapped = {}
        checkpoint_keys = set(state_dict.keys())
        
        # Strategy 1: Direct match (no remapping needed)
        direct_matches = checkpoint_keys & model_keys
        for key in direct_matches:
            remapped[key] = state_dict[key]
        
        print(f"\nKey Mapping Strategy:")
        print(f"  Direct matches: {len(direct_matches)}")
        
        # Strategy 2: Known key mappings
        key_mappings = {
            # Embedding layers
            "tok_embeddings.weight": "embed_tokens.weight",
            "wte.weight": "embed_tokens.weight",
            "transformer.wte.weight": "embed_tokens.weight",
            
            "wpe.weight": "embed_positions.weight",
            "position_embeddings.weight": "embed_positions.weight",
            "transformer.wpe.weight": "embed_positions.weight",
            
            # LM head
            "output.weight": "lm_head.weight",
            "lm_head.linear.weight": "lm_head.weight",
            
            # Layer norm
            "ln_f.weight": "layer_norm.weight",
            "ln_f.bias": "layer_norm.bias",
            "final_layernorm.weight": "layer_norm.weight",
            "final_layernorm.bias": "layer_norm.bias",
        }
        
        mapped_count = 0
        for old_key, new_key in key_mappings.items():
            if old_key in state_dict and new_key not in remapped:
                remapped[new_key] = state_dict[old_key]
                mapped_count += 1
                print(f"  Mapped: {old_key} -> {new_key}")
        
        print(f"  Known mappings: {mapped_count}")
        
        # Strategy 3: Pattern-based remapping for layer keys
        # Handle different layer naming conventions
        remaining_keys = checkpoint_keys - direct_matches - set(key_mappings.keys())
        pattern_mapped = 0
        
        for key in remaining_keys:
            new_key = key
            
            # Pattern: h.X.* -> layers.X.*
            if key.startswith('h.'):
                new_key = 'layers.' + key[2:]
            
            # Pattern: transformer.h.X.* -> layers.X.*
            elif key.startswith('transformer.h.'):
                new_key = 'layers.' + key[14:]
            
            # Pattern: blocks.X.* -> layers.X.*
            elif key.startswith('blocks.'):
                new_key = 'layers.' + key[7:]
            
            # If we found a mapping and it exists in model
            if new_key != key and new_key in model_keys:
                remapped[new_key] = state_dict[key]
                pattern_mapped += 1
                if pattern_mapped <= 5:  # Show first 5 examples
                    print(f"  Pattern match: {key} -> {new_key}")
        
        if pattern_mapped > 5:
            print(f"  ... and {pattern_mapped - 5} more pattern matches")
        print(f"  Pattern matches: {pattern_mapped}")
        
        # Strategy 4: Keep remaining keys as-is (they might be correct)
        for key in remaining_keys:
            if key not in remapped:
                remapped[key] = state_dict[key]
        
        print(f"  Total mapped: {len(remapped)}")
        
        return remapped
    
    def _validate_shapes(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Validate that checkpoint shapes match model shapes."""
        print("\nValidating shapes...")
        
        shape_checks = [
            ("embed_tokens.weight", self.model.embed_tokens.weight.shape),
            ("embed_positions.weight", self.model.embed_positions.weight.shape),
            ("lm_head.weight", self.model.lm_head.weight.shape),
        ]
        
        mismatches = []
        for key, expected_shape in shape_checks:
            if key in state_dict:
                actual_shape = state_dict[key].shape
                if actual_shape != expected_shape:
                    mismatches.append(f"  {key}: checkpoint={actual_shape}, model={expected_shape}")
                else:
                    print(f"  ✓ {key}: {actual_shape}")
            else:
                print(f"  ⚠ {key}: NOT FOUND in checkpoint")
        
        if mismatches:
            print("\nShape mismatches detected:")
            for mismatch in mismatches:
                print(mismatch)
            raise ValueError(
                "Model and checkpoint shapes do not match! "
                "Please ensure you are using the correct config."
            )
    
    def _fix_weight_scales(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fix scalar weight_scale tensors by converting to 1-element tensors."""
        fixed = {}
        for key, value in state_dict.items():
            if "weight_scale" in key and isinstance(value, torch.Tensor):
                if value.shape == torch.Size([]):
                    # Convert scalar to 1-element tensor
                    fixed[key] = value.view(1)
                else:
                    fixed[key] = value
            else:
                fixed[key] = value
        return fixed
    
    def _report_loading_issues(self, missing_keys: List[str], unexpected_keys: List[str]) -> None:
        """Report any issues with loading the state dict."""
        # Filter out expected missing/unexpected keys
        ignorable_missing = [
            "kv_cache",
            "early_exit_curriculum",
            "layer_skipping",
            "rotary_emb.cos_cached",
            "rotary_emb.sin_cached",
        ]
        
        ignorable_unexpected = [
            "optimizer",
            "scheduler",
            "step",
            "epoch",
        ]
        
        actual_missing = [
            k for k in missing_keys 
            if not any(ignore in k for ignore in ignorable_missing)
        ]
        
        actual_unexpected = [
            k for k in unexpected_keys 
            if not any(ignore in k for ignore in ignorable_unexpected)
        ]
        
        if actual_missing:
            print(f"\n⚠ Missing keys ({len(actual_missing)}):")
            for key in actual_missing[:10]:  # Show first 10
                print(f"  - {key}")
            if len(actual_missing) > 10:
                print(f"  ... and {len(actual_missing) - 10} more")
        
        if actual_unexpected:
            print(f"\n⚠ Unexpected keys ({len(actual_unexpected)}):")
            for key in actual_unexpected[:10]:  # Show first 10
                print(f"  - {key}")
            if len(actual_unexpected) > 10:
                print(f"  ... and {len(actual_unexpected) - 10} more")
        
        if not actual_missing and not actual_unexpected:
            print("✓ All keys loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        early_exit_threshold: Optional[float] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        debug: bool = False
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            early_exit_threshold: Confidence threshold for early exit (None to disable)
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            debug: Enable debug logging
            
        Returns:
            Generated text
        """
        with torch.autograd.profiler.record_function("Engine.generate"):
            # Tokenize input
            with torch.autograd.profiler.record_function("Engine.tokenize"):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            prompt_len = input_ids.shape[1]
            actual_max_length = min(max_length, self.config.max_position_embeddings - prompt_len - 1)
            
            if actual_max_length < max_length and debug:
                print(f"Reducing max_length from {max_length} to {actual_max_length} due to position limit")
            
            if actual_max_length <= 0:
                if debug:
                    print(f"Warning: Cannot generate. Prompt length {prompt_len} >= max positions")
                return prompt
            
            generated = input_ids
            past_key_values = None
            input_length = input_ids.shape[1]
            
            with torch.autograd.profiler.record_function("Engine.generation_loop"):
                for step in range(actual_max_length):
                    # Prepare model inputs based on KV cache support
                    if not self.supports_kv_cache:
                        # No KV cache: feed full sequence
                        model_inputs = generated
                        seq_len = model_inputs.shape[1]
                        current_position_ids = torch.arange(
                            seq_len, dtype=torch.long, device=self.device
                        ).unsqueeze(0)
                        current_attention_mask = torch.ones_like(model_inputs)
                    else:
                        # KV cache enabled: feed only new token after first step
                        if step == 0:
                            model_inputs = generated
                            current_position_ids = torch.arange(
                                input_length, dtype=torch.long, device=self.device
                            ).unsqueeze(0)
                        else:
                            model_inputs = generated[:, -1:]
                            current_position_ids = torch.tensor(
                                [[input_length + step - 1]], dtype=torch.long, device=self.device
                            )
                        current_attention_mask = torch.ones_like(model_inputs)
                    
                    with torch.no_grad():
                        if self.supports_kv_cache:
                            outputs = self.model(
                                input_ids=model_inputs,
                                attention_mask=current_attention_mask,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_hidden_states=True,
                                return_dict=True,
                                position_ids=current_position_ids
                            )
                            past_key_values = getattr(outputs, 'past_key_values', None)
                        else:
                            outputs = self.model(
                                input_ids=model_inputs,
                                attention_mask=current_attention_mask,
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True,
                                position_ids=current_position_ids
                            )
                        
                        logits = None
                        max_prob = None
                        
                        # Handle early exit if enabled
                        if early_exit_threshold is not None and outputs.hidden_states is not None:
                            for layer_idx, hidden in enumerate(outputs.hidden_states):
                                if hidden is None:
                                    continue
                                
                                layer_logits = self.model.lm_head(hidden[:, -1, :])
                                probs = torch.softmax(layer_logits, dim=-1)
                                layer_max_prob, _ = torch.max(probs, dim=-1)
                                
                                if (layer_max_prob > early_exit_threshold).any().item():
                                    logits = layer_logits
                                    max_prob = layer_max_prob
                                    if debug:
                                        print(f"Early exit at layer {layer_idx} "
                                              f"with confidence {max_prob.item():.4f}")
                                    break
                        
                        # Use final layer logits if no early exit
                        if logits is None:
                            logits = outputs.logits[:, -1, :]
                            probs = torch.softmax(logits, dim=-1)
                            max_prob, _ = torch.max(probs, dim=-1)
                        
                        # Apply repetition penalty
                        if repetition_penalty != 1.0 and step > 0:
                            for token_id in set(generated[0].tolist()):
                                logits[0, token_id] /= repetition_penalty
                        
                        # Sample next token
                        if temperature > 0:
                            # Apply temperature
                            logits = logits / temperature
                            
                            # Top-p (nucleus) sampling
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(
                                torch.softmax(sorted_logits, dim=-1), dim=-1
                            )
                            
                            # Remove tokens with cumulative probability above threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            if sorted_indices_to_remove[..., 1:].any().item():
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                            
                            # Apply mask to logits
                            mask = torch.ones_like(logits, dtype=torch.bool)
                            for i in range(sorted_indices.shape[0]):
                                for j in range(sorted_indices.shape[1]):
                                    if sorted_indices_to_remove[i, j]:
                                        idx = sorted_indices[i, j]
                                        if idx < logits.shape[-1]:
                                            mask[i, idx] = False
                            
                            logits = logits.masked_fill(~mask, float('-inf'))
                            
                            # Sample from distribution
                            probs = torch.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            # Greedy sampling
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        # Append to generated sequence
                        generated = torch.cat([generated, next_token], dim=1)
                        
                        if debug:
                            print(f"Step {step}: token={next_token.item()}, "
                                  f"text={self.tokenizer.decode(generated[0])}")
                        
                        # Check for position limit
                        if input_length + step + 1 >= self.config.max_position_embeddings:
                            if debug:
                                print(f"Reached maximum position limit")
                            break
                        
                        # Check for EOS token
                        if next_token.item() == self.tokenizer.eos_token_id:
                            if debug:
                                print("Generated EOS token, stopping")
                            break
            
            # Decode and return
            output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return output_text
    
    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()