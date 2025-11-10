# BitSkip: An Empirical Analysis of Quantization and Early Exit Composition
https://arxiv.org/abs/2510.23766
#### [<code>HF Checkpoint 🚀</code>](https://huggingface.co/Ram07/bitskip-v1-earlyexit) | [<code>Technical Report 📝</code>](https://arxiv.org/abs/2510.23766) 

A PyTorch implementation of BitNet with advanced early exit mechanisms, learnable routing decisions, and joint optimization of quantization and routing losses.

## 🚀 Key Features

### 1. **Learnable Gating Network for Early Exit**
- **Architecture**: `Gate_l(x) = σ(Linear(ReLU(Linear(LayerNorm(x)))))`
- **Input**: Layer output activation `h_l`
- **Output**: Scalar probability `p_exit ∈ [0, 1]`
- **Training**: Straight-through estimator for differentiable Bernoulli sampling
- **Inference**: Threshold-based decision (`p_exit > 0.5`)

### 2. **Joint Loss Optimization**
- **Task Loss**: Standard cross-entropy loss for language modeling
- **Quantization Loss**: MSE between original and quantized weights
- **Routing Loss**: Target cost loss encouraging efficient early exit
- **Early Exit Loss**: Additional supervision for early exit mechanisms

### 3. **BitNet Architecture**
- **BitLinear Layers**: Ternary weight quantization (-1, 0, 1)
- **H-BitLinear Layers**: Hadamard transform + LayerNorm + quantization
- **Squared ReLU Activation**: `f(x) = ReLU(x)²`
- **Gradient Checkpointing**: Memory-efficient training

## 📁 Project Structure

```
BitSkip/
├── bitnet/
│   ├── modeling/
│   │   ├── routing.py          # 🆕 Gating Network & Routing Loss
│   │   ├── transformer.py      # 🔄 Updated with routing modules
│   │   ├── model.py           # 🔄 Updated with routing loss collection
│   │   ├── bitlinear.py       # BitLinear implementation
│   │   ├── h_bitlinear.py     # H-BitLinear implementation
│   │   ├── attention.py       # BitNet attention
│   │   ├── feed_forward.py    # 🔄 Updated with quantization loss collection
│   │   └── ...
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── kernels/
│   ├── training/
│   └── utils/
├── train_quadratic.py         # 🔄 Updated with routing loss
├── train_quadratic_hbitlinear.py  # 🔄 Updated with routing loss
├── train_no_dropout.py        # 🔄 Updated with routing loss
├── train_early_exit.py        # 🔄 Updated with routing loss
├── train_layer_dropout.py     # 🔄 Updated with routing loss
└── requirements.txt
```

## 🏗️ Architecture Details

### Routing Module

```python
class RoutingModule(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )
    
    def forward(self, hidden_states, training=True):
        x = self.layer_norm(hidden_states)
        p_exit_logits = self.gate_mlp(x)
        p_exit = torch.sigmoid(p_exit_logits)
        
        if training:
            z_exit = self._straight_through_sampling(p_exit)
        else:
            z_exit = (p_exit > 0.5).float()
        
        return p_exit, z_exit
```

### Routing Loss (Target Cost Loss)

```python
class RoutingLoss(nn.Module):
    def __init__(self, target_exit_layer: float = 6.0, num_layers: int = 12):
        super().__init__()
        self.target_exit_layer = target_exit_layer
        self.num_layers = num_layers
    
    def forward(self, p_exit_list):
        # Compute expected exit layer
        expected_exit_layer = self._compute_expected_exit_layer(p_exit_list)
        
        # Target cost loss: encourage expected exit layer to be close to target
        routing_loss = F.mse_loss(expected_exit_layer, self.target_exit_layer)
        return routing_loss
```

## 🔧 Training Scripts

### Available Training Configurations

1. **`train_quadratic.py`** - Quadratic Schedule + Early Exit + Routing
2. **`train_quadratic_hbitlinear.py`** - H-BitLinear + Quadratic Schedule + Early Exit + Routing
3. **`train_no_dropout.py`** - Pure BitNet (Control) + Routing
4. **`train_early_exit.py`** - Early Exit + Routing
5. **`train_layer_dropout.py`** - Layer Dropout + Routing

### Joint Loss Function

```python
def compute_joint_loss(model, batch, lambda_q=0.1, lambda_r=0.05):
    # Standard forward pass
    outputs = model(**batch)
    task_loss = outputs.loss
    
    # Collect quantization losses
    quantization_info = model.collect_quantization_losses()
    quant_losses = list(quantization_info.values()) if quantization_info else []
    
    # Collect routing losses
    routing_loss = model.collect_routing_losses()
    
    # Combine losses
    total_loss = task_loss
    if quant_losses:
        total_loss += lambda_q * torch.stack(quant_losses).mean()
    if routing_loss > 0:
        total_loss += lambda_r * routing_loss
    
    return total_loss, {
        'task': task_loss.item(),
        'quant': torch.stack(quant_losses).mean().item() if quant_losses else 0,
        'route': routing_loss.item() if routing_loss > 0 else 0
    }
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Quadratic Schedule with Routing
python train_quadratic.py --lambda_q 0.1 --lambda_r 0.05 --num_steps 1000

# H-BitLinear with Routing
python train_quadratic_hbitlinear.py --lambda_q 0.1 --lambda_r 0.05 --num_steps 1000

# Early Exit with Routing
python train_early_exit.py --lambda_q 0.1 --lambda_r 0.05 --num_steps 1000
```

### Command Line Arguments

- `--lambda_q`: Weight for quantization loss (default: 0.1)
- `--lambda_r`: Weight for routing loss (default: 0.05)
- `--target_exit_layer`: Target average exit layer (default: 6.0)
- `--num_steps`: Number of training steps
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate

## 📊 Loss Components

### 1. **Task Loss** (`L_pred`)
- Standard cross-entropy loss for language modeling
- Accumulated from all exit points with layer-wise weights

### 2. **Quantization Loss** (`L_quant`)
- MSE between original full-precision weights and quantized weights
- Computed for all BitLinear and H-BitLinear layers
- Formula: `||W - W̃||²` where `W` is original and `W̃` is quantized

### 3. **Routing Loss** (`L_route`)
- Target cost loss encouraging efficient early exit
- Encourages expected exit layer to be close to target
- Formula: `MSE(expected_exit_layer, target_exit_layer)`

### 4. **Early Exit Loss** (where applicable)
- Additional supervision for early exit mechanisms
- Layer-wise weighted loss computation

## 🔬 Technical Details

### Quantization Process

1. **Weight Quantization**: Ternary quantization (-1, 0, 1) with scaling
2. **Activation Quantization**: Multi-bit quantization with per-token scaling
3. **Straight-Through Estimator**: Differentiable quantization during training

### Routing Decision Process

1. **Training**: Straight-through estimator for differentiable sampling
2. **Inference**: Threshold-based decision (`p_exit > 0.5`)
3. **Expected Exit Layer**: `Σ(l * p_exit_at_l)` where `p_exit_at_l = p_exit_l * ∏(1 - p_exit_i)`

### Memory Optimization

- **Gradient Checkpointing**: Reduces memory usage during training
- **Efficient Quantization**: Reduces model size and memory footprint
- **Early Exit**: Reduces computational cost for easier examples

## 📈 Expected Benefits

1. **Computational Efficiency**: Learnable early exit reduces average computation
2. **Memory Efficiency**: Quantization reduces model size and memory usage
3. **Performance**: Joint optimization maintains task performance while improving efficiency
4. **Flexibility**: Configurable target exit layer and loss weights

## 🐛 Recent Fixes

### Quantization Loss Fix
- **Issue**: Quantization loss was always 0 due to incorrect collection method
- **Solution**: Implemented proper `collect_quantization_losses()` methods in all model components
- **Result**: Quantization loss now properly computed and logged

### Routing Loss Implementation
- **Issue**: Routing loss was not implemented
- **Solution**: Added complete gating network architecture with target cost loss
- **Result**: Learnable early exit decisions with efficient routing

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{bitskip,
  title={BitSkip: An Empirical Analysis of Quantization and Early Exit Composition},
  author={Ramshankar Bhuvaneswaran, Handan Liu},
  year={2025},
  url={https://github.com/ramshankar07/bitskip}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions and support, please open an issue on GitHub.
