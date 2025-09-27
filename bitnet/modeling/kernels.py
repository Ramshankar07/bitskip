"""
CUDA kernels for BitNet operations.
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Initialize CUDA availability flag
CUDA_AVAILABLE = False

# Initialize CUDA functions as None
bitnet_linear_cuda = None
bitnet_ffn_cuda = None
layer_skip_decision_cuda = None
layer_skip_attention_cuda = None
layer_skip_ffn_cuda = None
ternary_quantize_cuda = None
activation_quantize_cuda = None
bitlinear_forward_cuda = None
squared_relu_cuda = None
fwht_cuda = None
early_exit_loss_cuda = None
attention_scores_cuda = None
attention_output_cuda = None

class BitNetCudaKernelManager:
    """
    Manages CUDA kernel loading for BitNet operations.
    Provides singleton-like behavior.
    """
    
    _instance: Optional['BitNetCudaKernelManager'] = None
    _kernels_loaded: bool = False
    _cuda_module: Optional[Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BitNetCudaKernelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._load_kernels()
    
    def _load_kernels(self) -> None:
        """Load CUDA kernels."""
        if self._kernels_loaded and self._cuda_module is not None:
            return
        
        try:
            logger.info("Loading BitNet CUDA kernels...")
            from bitnet.kernels.bitnet_kernels import (
                bitnet_linear_cuda,
                bitnet_ffn_cuda,
                ternary_quantize_cuda,
                activation_quantize_cuda,
                bitlinear_forward_cuda,
                squared_relu_cuda,
                fwht_cuda,
                early_exit_loss_cuda,
                attention_scores_cuda,
                attention_output_cuda
            )
            from bitnet.kernels.layer_skip_kernels import (
                layer_skip_decision_cuda,
                layer_skip_attention_cuda,
                layer_skip_ffn_cuda
            )
            
            self._cuda_module = {
                'bitnet_linear_cuda': bitnet_linear_cuda,
                'bitnet_ffn_cuda': bitnet_ffn_cuda,
                'ternary_quantize_cuda': ternary_quantize_cuda,
                'activation_quantize_cuda': activation_quantize_cuda,
                'bitlinear_forward_cuda': bitlinear_forward_cuda,
                'squared_relu_cuda': squared_relu_cuda,
                'fwht_cuda': fwht_cuda,
                'early_exit_loss_cuda': early_exit_loss_cuda,
                'attention_scores_cuda': attention_scores_cuda,
                'attention_output_cuda': attention_output_cuda,
                'layer_skip_decision_cuda': layer_skip_decision_cuda,
                'layer_skip_attention_cuda': layer_skip_attention_cuda,
                'layer_skip_ffn_cuda': layer_skip_ffn_cuda
            }
            
            self._kernels_loaded = True
            
        except ImportError as e:
            # Silently handle CUDA kernel import failure
            self._kernels_loaded = False
            self._cuda_module = None
    
    @property
    def is_available(self) -> bool:
        """Check if CUDA kernels are available."""
        return self._kernels_loaded and self._cuda_module is not None and torch.cuda.is_available()
    
    @property
    def module(self) -> Optional[Any]:
        """Get the loaded CUDA module."""
        return self._cuda_module if self.is_available else None
    
    def get_kernel(self, kernel_name: str) -> Optional[Any]:
        """Get a specific kernel function."""
        if not self.is_available:
            return None
        
        return self._cuda_module.get(kernel_name)

# Global instance for backward compatibility
bitnet_kernels = BitNetCudaKernelManager()

# Try to import CUDA extensions
try:
    from bitnet.kernels.bitnet_kernels import (
        bitnet_linear_cuda,
        bitnet_ffn_cuda,
        ternary_quantize_cuda,
        activation_quantize_cuda,
        bitlinear_forward_cuda,
        squared_relu_cuda,
        fwht_cuda,
        early_exit_loss_cuda,
        attention_scores_cuda,
        attention_output_cuda
    )
    from bitnet.kernels.layer_skip_kernels import (
        layer_skip_decision_cuda,
        layer_skip_attention_cuda,
        layer_skip_ffn_cuda
    )
    CUDA_AVAILABLE = True
except ImportError as e:
    # Silently handle CUDA kernel import failure
    pass

def early_exit_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    layer_idx: int,
    total_layers: int
) -> torch.Tensor:
    """
    Early exit loss computation with CUDA acceleration.
    
    Args:
        logits: Logits tensor
        targets: Target tensor
        layer_idx: Current layer index
        total_layers: Total number of layers
        
    Returns:
        Loss tensor
    """
    if CUDA_AVAILABLE and logits.is_cuda and early_exit_loss_cuda is not None:
        return early_exit_loss_cuda(logits, targets, layer_idx, total_layers)
    else:
        # CPU fallback
        # Weight loss by layer depth
        layer_weight = (layer_idx + 1) / total_layers
        # Compute cross entropy loss
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        # Scale loss by layer weight
        return loss * layer_weight

def attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Attention scores computation with CUDA acceleration.
    
    Args:
        query: Query tensor
        key: Key tensor
        scale: Scaling factor
        
    Returns:
        Attention scores tensor
    """
    if CUDA_AVAILABLE and query.is_cuda and attention_scores_cuda is not None:
        return attention_scores_cuda(query, key, scale)
    else:
        # CPU fallback
        return torch.matmul(query, key.transpose(-2, -1)) / scale

def attention_output(
    attention_weights: torch.Tensor,
    value: torch.Tensor
) -> torch.Tensor:
    """
    Attention output computation with CUDA acceleration.
    
    Args:
        attention_weights: Attention weights tensor
        value: Value tensor
        
    Returns:
        Attention output tensor
    """
    if CUDA_AVAILABLE and attention_weights.is_cuda and attention_output_cuda is not None:
        return attention_output_cuda(attention_weights, value)
    else:
        # CPU fallback
        return torch.matmul(attention_weights, value)

def fwht(input: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform with CUDA acceleration.
    
    Args:
        input: Input tensor
        
    Returns:
        Transformed tensor
    """
    if CUDA_AVAILABLE and input.is_cuda and fwht_cuda is not None:
        return fwht_cuda(input)
    else:
        # CPU fallback
        n = input.size(-1)
        if not (n & (n - 1) == 0):
            raise ValueError("Input size must be a power of 2")
        
        x = input.clone()
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x[..., j], x[..., j + h] = x[..., j] + x[..., j + h], x[..., j] - x[..., j + h]
            h *= 2
        return x / torch.sqrt(torch.tensor(n, dtype=x.dtype))

def squared_relu(input: torch.Tensor) -> torch.Tensor:
    """
    Squared ReLU activation with CUDA acceleration.
    
    Args:
        input: Input tensor
        
    Returns:
        Output tensor
    """
    if CUDA_AVAILABLE and input.is_cuda and squared_relu_cuda is not None:
        return squared_relu_cuda(input)
    else:
        # CPU fallback
        relu_output = torch.nn.functional.relu(input)
        return relu_output * relu_output

def bitlinear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """
    BitLinear forward pass with CUDA acceleration.
    
    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor
        scale: Scaling factor
        
    Returns:
        Output tensor
    """
    if CUDA_AVAILABLE and input.is_cuda and bitlinear_forward_cuda is not None:
        return bitlinear_forward_cuda(input, weight, bias, scale)
    else:
        # CPU fallback
        # First quantize the weights
        quantized_weight, weight_scale = ternary_quantize(weight)
        # Then quantize the input
        quantized_input, input_scale = activation_quantize(input)
        # Perform linear operation
        output = torch.nn.functional.linear(quantized_input, quantized_weight, bias)
        # Apply scaling
        output = output * (weight_scale * input_scale * scale)
        return output

def activation_quantize(input: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Activation quantization with CUDA acceleration.
    
    Args:
        input: Input tensor to quantize
        bits: Number of bits for quantization (default: 8)
        
    Returns:
        Tuple of (quantized tensor, scaling factor)
    """
    if CUDA_AVAILABLE and input.is_cuda and activation_quantize_cuda is not None:
        return activation_quantize_cuda(input, bits)
    else:
        # CPU fallback
        abs_input = torch.abs(input)
        max_val = torch.max(abs_input)
        scale = max_val / (2 ** (bits - 1) - 1)
        quantized = torch.round(input / scale) * scale
        return quantized, scale

def ternary_quantize(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ternary weight quantization with CUDA acceleration.
    
    Args:
        weights: Weight tensor to quantize
        
    Returns:
        Tuple of (quantized weights, scaling factor)
    """
    if CUDA_AVAILABLE and weights.is_cuda and ternary_quantize_cuda is not None:
        return ternary_quantize_cuda(weights)
    else:
        # CPU fallback
        abs_weights = torch.abs(weights)
        threshold = 0.7 * torch.mean(abs_weights)
        mask = abs_weights > threshold
        quantized = torch.zeros_like(weights)
        quantized[mask] = torch.sign(weights[mask])
        scale = torch.mean(abs_weights[mask]) if mask.any().item() else torch.tensor(1.0)
        return quantized, scale

def bitnet_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """
    BitNet linear layer with CUDA acceleration.
    
    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor
        scale: Scaling factor
        
    Returns:
        Output tensor
    """
    if CUDA_AVAILABLE and input.is_cuda and bitnet_linear_cuda is not None:
        return bitnet_linear_cuda(input, weight, bias, scale)
    else:
        # CPU fallback
        return torch.nn.functional.linear(input, weight, bias)

def bitnet_ffn(
    input: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """
    BitNet feed-forward network with CUDA acceleration.
    
    Args:
        input: Input tensor
        weight1: First weight tensor
        weight2: Second weight tensor
        bias1: Optional first bias tensor
        bias2: Optional second bias tensor
        scale: Scaling factor
        
    Returns:
        Output tensor
    """
    if CUDA_AVAILABLE and input.is_cuda and bitnet_ffn_cuda is not None:
        return bitnet_ffn_cuda(input, weight1, weight2, bias1, bias2, scale)
    else:
        # CPU fallback
        hidden = torch.nn.functional.linear(input, weight1, bias1)
        hidden = torch.nn.functional.gelu(hidden)
        return torch.nn.functional.linear(hidden, weight2, bias2)

def layer_skip_decision(
    input: torch.Tensor,
    skip_prob: float,
    training: bool = True
) -> torch.Tensor:
    """
    Layer skip decision with CUDA acceleration.
    
    Args:
        input: Input tensor
        skip_prob: Skip probability
        training: Whether in training mode
        
    Returns:
        Skip decision tensor
    """
    if CUDA_AVAILABLE and input.is_cuda and layer_skip_decision_cuda is not None:
        return layer_skip_decision_cuda(input, skip_prob, training)
    else:
        # CPU fallback
        if training:
            return torch.rand_like(input) < skip_prob
        return torch.zeros_like(input, dtype=torch.bool)

def layer_skip_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    skip_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Layer skip attention with CUDA acceleration.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        skip_mask: Skip mask tensor
        attention_mask: Optional attention mask
        scale: Scaling factor
        
    Returns:
        Attention output tensor
    """
    if CUDA_AVAILABLE and query.is_cuda and layer_skip_attention_cuda is not None:
        return layer_skip_attention_cuda(query, key, value, skip_mask, attention_mask, scale)
    else:
        # CPU fallback
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output * (~skip_mask).float().unsqueeze(-1)

def layer_skip_ffn(
    input: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    skip_mask: torch.Tensor,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Layer skip feed-forward network with CUDA acceleration.
    
    Args:
        input: Input tensor
        weight1: First weight tensor
        weight2: Second weight tensor
        skip_mask: Skip mask tensor
        bias1: Optional first bias tensor
        bias2: Optional second bias tensor
        scale: Scaling factor
        
    Returns:
        Output tensor
    """
    if CUDA_AVAILABLE and input.is_cuda and layer_skip_ffn_cuda is not None:
        return layer_skip_ffn_cuda(input, weight1, weight2, skip_mask, bias1, bias2, scale)
    else:
        # CPU fallback
        hidden = torch.nn.functional.linear(input, weight1, bias1)
        hidden = torch.nn.functional.gelu(hidden)
        output = torch.nn.functional.linear(hidden, weight2, bias2)
        return output * (~skip_mask).float().unsqueeze(-1)

# Export all functions
__all__ = [
    'bitnet_kernels',  # For backward compatibility
    'ternary_quantize',
    'activation_quantize',
    'bitlinear_forward',
    'squared_relu',
    'fwht',
    'early_exit_loss',  # Added early_exit_loss
    'attention_scores',  # Added attention_scores
    'attention_output',  # Added attention_output
    'bitnet_linear',
    'bitnet_ffn',
    'layer_skip_decision',
    'layer_skip_attention',
    'layer_skip_ffn',
    'CUDA_AVAILABLE'
]