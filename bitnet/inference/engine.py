"""
Inference engine for BitNet model with TRUE early exit optimization.
"""

from typing import Dict, List, Optional, Union, Set, Tuple
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
    Inference engine for BitNet model with optimized generation and TRUE early exit.
    
    Args:
        model_path: Path to pretrained model (directory or .safetensors file)
        tokenizer_path: Path to tokenizer
        device: Device to use for inference
        model_type: Type of model to load ('bitnet', 'bitnet2', or 'auto')
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
        
        # Debug: Print key analysis
        self._analyze_checkpoint_keys(state_dict)
        
        # Normalize state dict keys (remove training wrapper prefixes)
        state_dict = self._normalize_state_dict(state_dict)
        
        # Auto-detect model type
        if model_type == "auto":
            self.model_type = self._detect_model_type(state_dict)
            print(f"Auto-detected model type: {self.model_type}")
        
        # Infer model architecture from state dict
        self._infer_config_from_state_dict(state_dict)
        
        # Print final configuration
        self._print_config()
        
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
        
        # Convert to BF16 for inference on CUDA
        if torch.cuda.is_available() and self.device == "cuda":
            self.model = self.model.to(torch.bfloat16)
            print(f"Model converted to BF16 for inference")
        
        print(f"Model loaded successfully on {self.device}")
        
        # Load tokenizer
        print(f"\nLoading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Tokenizer loaded: vocab_size={len(self.tokenizer)}")
        
        # Initialize KV cache
        self.kv_cache = None
    
    def _analyze_checkpoint_keys(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Analyze and print checkpoint key structure."""
        print("\n=== CHECKPOINT KEY ANALYSIS ===")
        
        # Group keys by prefix
        key_groups = {}
        for key in sorted(state_dict.keys()):
            parts = key.split('.')
            if len(parts) >= 2:
                prefix = f"{parts[0]}.{parts[1]}"
            else:
                prefix = parts[0] if parts else key
            
            if prefix not in key_groups:
                key_groups[prefix] = []
            key_groups[prefix].append(key)
        
        print(f"\nKey prefixes found ({len(key_groups)} groups):")
        
        for prefix, keys in sorted(key_groups.items()):
            print(f"\n{prefix}.* ({len(keys)} keys):")
            display_keys = keys if len(keys) <= 5 else keys[:5]
            for key in display_keys:
                print(f"  {key}: {state_dict[key].shape}")
            if len(keys) > 5:
                print(f"  ... and {len(keys) - 5} more")
        
        # Critical weight check
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
            print("  Missing critical model weights for inference.")
        
        print("\n=== END CHECKPOINT ANALYSIS ===\n")
    
    def _print_config(self) -> None:
        """Print model configuration."""
        print(f"\nFinal Configuration:")
        print(f"  vocab_size: {self.config.vocab_size}")
        print(f"  hidden_size: {self.config.hidden_size}")
        print(f"  num_hidden_layers: {self.config.num_hidden_layers}")
        print(f"  num_attention_heads: {self.config.num_attention_heads}")
        print(f"  max_position_embeddings: {self.config.max_position_embeddings}")
        print(f"  mlp_ratio: {self.config.mlp_ratio}")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve the model path to a .safetensors file."""
        if os.path.isfile(model_path):
            if model_path.endswith('.safetensors'):
                return model_path
            else:
                raise ValueError(f"Only .safetensors format is supported, got: {model_path}")
        
        if os.path.isdir(model_path):
            preferred_names = ['model.safetensors', 'pytorch_model.safetensors']
            for name in preferred_names:
                candidate = os.path.join(model_path, name)
                if os.path.exists(candidate):
                    return candidate
            
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
                
                valid_keys = set(DefaultConfig.__dataclass_fields__.keys())
                filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
                
                self.config = DefaultConfig.from_dict(filtered_config)
                print(f"Loaded config from config.json")
            except Exception as e:
                print(f"Warning: Failed to parse config.json: {e}")
    
    def _normalize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize state dict by removing common prefixes."""
        def strip_prefix(name: str) -> str:
            prefixes = ["model.", "module.", "_orig_mod."]
            for prefix in prefixes:
                if name.startswith(prefix):
                    return name[len(prefix):]
            return name
        
        normalized = {strip_prefix(key): value for key, value in state_dict.items()}
        print(f"Normalized {len(state_dict)} keys")
        return normalized
    
    def _detect_model_type(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect whether this is BitNetModel or BitNetModel2."""
        for key in state_dict.keys():
            if "h_bitlinear" in key.lower() or "hadamard" in key.lower():
                return "bitnet2"
        return "bitnet"
    
    def _infer_config_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Infer model configuration from state dict."""
        def find_key(candidates: List[str]) -> Optional[str]:
            for key in candidates:
                if key in state_dict:
                    return key
            return None
        
        # Infer from embeddings
        embed_tokens_key = find_key(["embed_tokens.weight", "tok_embeddings.weight", "wte.weight"])
        if embed_tokens_key:
            vocab_size, hidden_size = state_dict[embed_tokens_key].shape
            self.config.vocab_size = vocab_size
            self.config.hidden_size = hidden_size
            print(f"Inferred: vocab_size={vocab_size}, hidden_size={hidden_size}")
        
        # Infer max_position_embeddings
        embed_pos_key = find_key(["embed_positions.weight", "wpe.weight"])
        if embed_pos_key:
            max_pos = state_dict[embed_pos_key].shape[0]
            self.config.max_position_embeddings = max_pos
            print(f"Inferred max_position_embeddings: {max_pos}")
        
        # Count layers
        layer_count = 0
        for key in state_dict.keys():
            if 'layers.' in key:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        layer_count = max(layer_count, layer_idx + 1)
        
        if layer_count > 0:
            self.config.num_hidden_layers = layer_count
            print(f"Inferred num_hidden_layers: {layer_count}")
        
        # Infer attention heads
        attention_keys = [k for k in state_dict.keys() if 'q_proj.weight' in k]
        if attention_keys:
            q_proj_shape = state_dict[attention_keys[0]].shape
            if q_proj_shape[0] == self.config.hidden_size:
                for head_dim in [64, 128, 96]:
                    if self.config.hidden_size % head_dim == 0:
                        self.config.num_attention_heads = self.config.hidden_size // head_dim
                        print(f"Inferred num_attention_heads: {self.config.num_attention_heads}")
                        break
        
        # Infer num_kv_heads (GQA)
        kv_keys = [k for k in state_dict.keys() if 'k_proj.weight' in k]
        if kv_keys:
            k_proj_shape = state_dict[kv_keys[0]].shape
            kv_hidden_size = k_proj_shape[0]
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            num_kv_heads = kv_hidden_size // head_dim
            self.config.num_kv_heads = num_kv_heads
            print(f"Inferred num_kv_heads: {num_kv_heads}")
        
        # Infer mlp_ratio
        mlp_keys = [k for k in state_dict.keys() if 'up_proj.weight' in k]
        if mlp_keys:
            intermediate_size = state_dict[mlp_keys[0]].shape[0]
            mlp_ratio = intermediate_size / self.config.hidden_size
            self.config.mlp_ratio = mlp_ratio
            print(f"Inferred mlp_ratio: {mlp_ratio:.2f}")
        else:
            self.config.mlp_ratio = 2.0
            print(f"Using default mlp_ratio: 2.0")
    
    def _initialize_model(self) -> Tuple:
        """Initialize the appropriate model."""
        if self.model_type == "bitnet2":
            model = BitNetModel2(self.config)
            print("Initialized BitNetModel2")
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
        """Intelligently remap state dict keys to match model."""
        remapped = {}
        checkpoint_keys = set(state_dict.keys())
        
        # Strategy 1: Direct match
        direct_matches = checkpoint_keys & model_keys
        for key in direct_matches:
            remapped[key] = state_dict[key]
        
        print(f"\nKey Mapping:")
        print(f"  Direct matches: {len(direct_matches)}")
        
        # Strategy 2: Known mappings
        key_mappings = {
            "tok_embeddings.weight": "embed_tokens.weight",
            "wte.weight": "embed_tokens.weight",
            "wpe.weight": "embed_positions.weight",
            "output.weight": "lm_head.weight",
            "ln_f.weight": "layer_norm.weight",
            "ln_f.bias": "layer_norm.bias",
        }
        
        mapped_count = 0
        for old_key, new_key in key_mappings.items():
            if old_key in state_dict and new_key not in remapped:
                remapped[new_key] = state_dict[old_key]
                mapped_count += 1
        
        print(f"  Known mappings: {mapped_count}")
        
        # Strategy 3: Pattern-based remapping
        remaining_keys = checkpoint_keys - direct_matches - set(key_mappings.keys())
        pattern_mapped = 0
        
        for key in remaining_keys:
            new_key = key
            
            if key.startswith('h.'):
                new_key = 'layers.' + key[2:]
            elif key.startswith('transformer.h.'):
                new_key = 'layers.' + key[14:]
            elif key.startswith('blocks.'):
                new_key = 'layers.' + key[7:]
            
            if new_key != key and new_key in model_keys:
                remapped[new_key] = state_dict[key]
                pattern_mapped += 1
        
        print(f"  Pattern matches: {pattern_mapped}")
        
        # Keep remaining keys
        for key in remaining_keys:
            if key not in remapped:
                remapped[key] = state_dict[key]
        
        print(f"  Total mapped: {len(remapped)}")
        
        return remapped
    
    def _validate_shapes(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Validate checkpoint shapes match model shapes."""
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
                    print(f"  OK {key}: {actual_shape}")
            else:
                print(f"  MISSING {key}")
        
        if mismatches:
            print("\nShape mismatches:")
            for mismatch in mismatches:
                print(mismatch)
            raise ValueError("Shape mismatch detected!")
    
    def _fix_weight_scales(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fix scalar weight_scale tensors."""
        fixed = {}
        for key, value in state_dict.items():
            if "weight_scale" in key and isinstance(value, torch.Tensor):
                if value.shape == torch.Size([]):
                    fixed[key] = value.view(1)
                else:
                    fixed[key] = value
            else:
                fixed[key] = value
        return fixed
    
    def _report_loading_issues(self, missing_keys: List[str], unexpected_keys: List[str]) -> None:
        """Report loading issues."""
        ignorable_missing = [
            "kv_cache", "early_exit_curriculum", "layer_skipping",
            "rotary_emb.cos_cached", "rotary_emb.sin_cached",
        ]
        
        ignorable_unexpected = ["optimizer", "scheduler", "step", "epoch"]
        
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
            for key in actual_missing[:10]:
                print(f"  - {key}")
            if len(actual_missing) > 10:
                print(f"  ... and {len(actual_missing) - 10} more")
        
        if actual_unexpected:
            print(f"\n⚠ Unexpected keys ({len(actual_unexpected)}):")
            for key in actual_unexpected[:10]:
                print(f"  - {key}")
            if len(actual_unexpected) > 10:
                print(f"  ... and {len(actual_unexpected) - 10} more")
        
        if not actual_missing and not actual_unexpected:
            print("OK All keys loaded successfully!")
    
    def _forward_with_early_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        threshold: float,
        debug: bool = False
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Forward pass with TRUE early exit.
        
        Returns:
            Tuple of (logits, exit_layer, confidence)
        """
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)
        position_embeddings = self.model.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeddings
        
        # Process layers incrementally
        for layer_idx, layer in enumerate(self.model.layers):
            # Forward through this layer
            layer_output = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            
            # Handle tuple return (with cache)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
            
            # Check if we should exit early (after layer 4)
            if layer_idx >= 4:
                # Apply layer norm
                normed_states = self.model.layer_norm(hidden_states)
                
                # Get logits for last token
                logits = self.model.lm_head(normed_states[:, -1, :])
                
                # Calculate confidence
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max().item()
                
                # Exit if confidence exceeds threshold
                if confidence > threshold:
                    if debug:
                        print(f"Early exit at layer {layer_idx} with confidence {confidence:.4f}")
                    return logits, layer_idx, confidence
        
        # If didn't exit early, use final layer output
        hidden_states = self.model.layer_norm(hidden_states)
        logits = self.model.lm_head(hidden_states[:, -1, :])
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max().item()
        
        if debug:
            print(f"Used all layers ({len(self.model.layers)}) with confidence {confidence:.4f}")
        
        return logits, len(self.model.layers) - 1, confidence
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        early_exit_threshold: Optional[float] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        debug: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Generate text from a prompt with TRUE early exit support.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            early_exit_threshold: Confidence threshold for early exit (None to disable)
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            debug: Enable debug logging
            
        Returns:
            Generated text, or (generated_text, stats_dict) if debug=True
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        prompt_len = input_ids.shape[1]
        actual_max_length = min(max_length, self.config.max_position_embeddings - prompt_len - 1)
        
        if actual_max_length <= 0:
            if debug:
                print(f"Cannot generate: prompt length {prompt_len} >= max positions")
            return prompt
        
        generated = input_ids
        exit_layers = []
        confidences = []
        
        # Generation loop
        for step in range(actual_max_length):
            # Prepare inputs
            seq_len = generated.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            attention_mask = torch.ones_like(generated)
            
            with torch.no_grad():
                if early_exit_threshold is not None and early_exit_threshold > 0.0:
                    # TRUE early exit: stops computing at confident layer
                    logits, exit_layer, confidence = self._forward_with_early_exit(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        threshold=early_exit_threshold,
                        debug=debug
                    )
                    exit_layers.append(exit_layer)
                    confidences.append(confidence)
                else:
                    # Standard generation: use full model
                    outputs = self.model(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True
                    )
                    logits = outputs.logits[:, -1, :]
                    exit_layers.append(len(self.model.layers) - 1)
                    
                    probs = torch.softmax(logits, dim=-1)
                    confidences.append(probs.max().item())
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and step > 0:
                    for token_id in set(generated[0].tolist()):
                        logits[0, token_id] /= repetition_penalty
                
                # Sample next token
                if temperature > 0:
                    logits = logits / temperature
                    
                    # Top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        if sorted_indices_to_remove[..., 1:].any():
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                        
                        mask = torch.ones_like(logits, dtype=torch.bool)
                        for i in range(sorted_indices.shape[0]):
                            for j in range(sorted_indices.shape[1]):
                                if sorted_indices_to_remove[i, j]:
                                    idx = sorted_indices[i, j]
                                    if idx < logits.shape[-1]:
                                        mask[i, idx] = False
                        
                        logits = logits.masked_fill(~mask, float('-inf'))
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    if debug:
                        print("Generated EOS token")
                    break
                
                # Check position limit
                if generated.shape[1] >= self.config.max_position_embeddings:
                    if debug:
                        print("Reached position limit")
                    break
        
        # Decode output
        output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        if debug:
            import statistics
            stats = {
                'num_tokens': len(exit_layers),
                'avg_exit_layer': statistics.mean(exit_layers) if exit_layers else 0,
                'avg_confidence': statistics.mean(confidences) if confidences else 0,
                'exit_layers': exit_layers,
                'confidences': confidences
            }
            return output_text, stats
        
        return output_text
    
    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self.kv_cache = None 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()