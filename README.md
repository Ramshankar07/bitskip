# BitSkip: An Empirical Analysis of Quantization and Early Exit Composition

[**Technical Report**](https://arxiv.org/abs/2510.23766) | [**HF Checkpoint**](https://huggingface.co/Ram07/bitskip-v1-earlyexit)

BitSkip is a PyTorch implementation of a BitNet transformer that jointly optimizes ternary weight quantization, multi-bit activation quantization, learnable early exit routing, and curriculum-based layer skipping. The v2 codebase extends the original paper with auxiliary routing and quantization losses, wandb integration, and Modal cloud training support.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [V2 Ablation Experiments](#v2-ablation-experiments)
- [Modal Cloud Training](#modal-cloud-training)
- [Loss Components](#loss-components)
- [Technical Details](#technical-details)
- [Configuration Reference](#configuration-reference)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Citation](#citation)
- [License](#license)

---

## Architecture

BitSkip is a 125M-parameter causal language model built on a 12-layer transformer with the following components:

| Component | Description |
|-----------|-------------|
| **BitLinear** | Ternary weight quantization (-1, 0, 1) with 8-bit per-token activation quantization and STE |
| **H-BitLinear** | Hadamard transform + LayerNorm + 4-bit activation quantization with FWHT CUDA kernel |
| **GQA** | Grouped Query Attention (12 query heads, 4 KV heads) with RoPE |
| **Squared ReLU** | Activation function: `f(x) = ReLU(x)^2` |
| **Layer Skipping** | Curriculum-based quadratic/linear/uniform dropout schedule during training |
| **Routing Module** | Learnable gating network for early exit: `Gate(x) = sigma(Linear(ReLU(Linear(LayerNorm(x)))))` |

**Data flow through one transformer block:**

```
Input --> BitNetGQA (attention) --> SublayerNorm + Residual
      --> BitFeedForward        --> SublayerNorm + Residual
      --> RoutingModule         --> p_exit, z_exit
      --> Output
```

**Two model variants:**

- `BitNetModel` (`model.py`) — uses `BitLinear` layers (8-bit activations)
- `BitNetModel2` (`model2.py`) — uses `HBitLinear` layers (4-bit activations, Hadamard transform)

---

## Project Structure

```
bitskip/
├── bitnet/
│   ├── modeling/
│   │   ├── model.py              # BitNetModel (standard BitLinear)
│   │   ├── model2.py             # BitNetModel2 (H-BitLinear / Hadamard)
│   │   ├── transformer.py        # BitTransformerBlock
│   │   ├── transformer2.py       # BitTransformerBlock2 (Hadamard variant)
│   │   ├── bitlinear.py          # Ternary weights + 8-bit activation quantization
│   │   ├── h_bitlinear.py        # Hadamard + LayerNorm + 4-bit activation quantization
│   │   ├── gqa_attention.py      # Grouped Query Attention with RoPE
│   │   ├── gqa_attention2.py     # GQA variant for H-BitLinear
│   │   ├── feed_forward.py       # BitLinear-based FFN
│   │   ├── feed_forward2.py      # H-BitLinear-based FFN
│   │   ├── routing.py            # RoutingModule + RoutingLoss (target cost)
│   │   ├── layer_skipping.py     # Layer dropout with quadratic/linear/uniform schedules
│   │   ├── rope.py               # Rotary Position Embeddings
│   │   ├── subln.py              # SublayerNorm with residual connections
│   │   └── kernels/
│   │       ├── __init__.py       # FWHT PyTorch fallback
│   │       ├── fwht.cu           # CUDA kernel for Fast Walsh-Hadamard Transform
│   │       └── fwht.cpp          # CUDA binding
│   ├── evaluation/
│   │   └── evaluator.py          # Perplexity, accuracy, cross-entropy evaluation
│   ├── inference/
│   │   └── engine.py             # Inference engine with early exit + self-speculative decode
│   ├── training/
│   │   └── trainer.py            # MemoryEfficientTrainer (AMP, gradient checkpointing)
│   └── utils/
│       ├── default_config.py     # DefaultConfig dataclass (all model/training params)
│       ├── checkpoint.py         # Checkpoint save/load utilities
│       ├── lr_schedule.py        # WSD scheduler with SMA
│       └── wandb_logger.py       # Weights & Biases logging wrapper
│
├── experiments/
│   ├── modal_train.py            # Modal B200 GPU training script
│   ├── modal_run_experiments.py  # Modal batch experiment runner (concurrent)
│   ├── run_v2_ablations.py       # V2 ablation study runner (4 stages)
│   └── bitskip_wikitext2/
│       ├── train.py              # Main training script (local)
│       ├── config.py             # ExperimentConfig dataclass
│       ├── model_configs.py      # Model size presets (125M)
│       ├── model_factory.py      # Model instantiation from config
│       ├── run_experiments_parallel.py  # Paper ablation runner (6 stages)
│       ├── run_experiments.py    # Sequential experiment runner
│       ├── view_results.py       # Results visualization
│       └── run_stage_*.slurm     # SLURM job scripts (stages 1-6)
│
├── evals/
│   ├── bitnet_huggingface_benchmark.py
│   ├── huggingface_layerskip_benchmark.py
│   ├── inference_benchmark_1b_improved.py
│   └── vllm_layerskip_benchmark.py
│
├── scripts/                      # Utilities (upload, convert, debug, benchmark)
├── archive/                      # Archived 1B/2B training scripts
├── results/                      # Experiment results and benchmark JSONs
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ramshankar07/bitskip.git
cd bitskip
pip install -r requirements.txt
```

For Modal cloud training:

```bash
pip install modal
modal token new
# If using wandb:
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

---

## Quick Start

### Train locally (WikiText-2, INT8, early exit enabled)

```bash
cd experiments/bitskip_wikitext2

python train.py \
    --model_id my_first_run \
    --dataset wikitext2 \
    --precision int8 \
    --early_exit_lambda 0.3 \
    --p_max 0.5 \
    --num_steps 2000 \
    --batch_size 16 \
    --wandb
```

### Train on Modal (B200 GPU)

```bash
modal run experiments/modal_train.py \
    --model-id my_modal_run \
    --dataset wikitext2 \
    --precision int8 \
    --batch-size 128 \
    --num-steps 5000 \
    --wandb
```

### Train with v2 auxiliary losses

```bash
python experiments/bitskip_wikitext2/train.py \
    --model_id v2_experiment \
    --precision int8 \
    --use_hadamard \
    --early_exit_lambda 0.3 \
    --p_max 0.5 \
    --lambda_q 0.05 \
    --lambda_r 0.05 \
    --num_steps 5000 \
    --wandb
```

---

## Training

### Command Line Arguments

**Core:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | (required) | Unique experiment name |
| `--output_dir` | `./results` | Directory for checkpoints and results |
| `--dataset` | `wikitext2` | Dataset: `wikitext2`, `wikitext103`, `ptb`, `mix` |
| `--precision` | `fp16` | Weight precision: `fp16`, `int8`, `int4` |
| `--use_hadamard` | off | Use H-BitLinear (Hadamard transform) |

**Early Exit / Layer Skipping:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_early_exit` | off | Disable early exit completely |
| `--early_exit_lambda` | `0.0` | Early exit loss weight (paper uses 0.3) |
| `--p_max` | `0.0` | Maximum layer dropout probability |
| `--dropout_schedule` | `quadratic` | Schedule: `quadratic`, `linear`, `uniform` |

**V2 Auxiliary Losses:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda_q` | `0.0` | Quantization loss weight (MSE between original and ternary weights) |
| `--lambda_r` | `0.0` | Routing loss weight (target cost loss for early exit) |

**Training:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | `16` | Batch size per device |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--learning_rate` | `6e-4` | Learning rate |
| `--num_steps` | config | Total training steps |
| `--eval_every_steps` | config | Validation frequency |
| `--seed` | `42` | Random seed |
| `--compile` | off | Enable `torch.compile` (max-autotune mode) |

**Logging:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb_project` | `bitskip-v2` | wandb project name |

### Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| `wikitext2` | 8.4 MB | WikiText-2 (small, fast iteration) |
| `wikitext103` | 517 MB | WikiText-103 (standard LM benchmark) |
| `ptb` | ~5 MB | Penn Treebank |
| `mix` | ~260 MB | 50% TinyStories + 50% WikiText-103 |

### Early Stopping

Training includes automatic early stopping:
- Validates every `eval_every_steps` steps
- Patience: 3 evaluations without improvement (configurable in `config.py`)
- Saves best model checkpoint as `model.safetensors`
- Results written to `results.txt` in the run directory

---

## V2 Ablation Experiments

The v2 codebase adds routing loss and quantization loss as auxiliary objectives. The ablation runner tests these in 4 stages, mirroring the paper's ablation methodology:

### Stages

| Stage | Sweep Parameter | Values | Purpose |
|-------|----------------|--------|---------|
| V2-1 | `lambda_r` (routing loss) | 0.0, 0.01, 0.05, 0.1, 0.2 | Find best routing loss weight |
| V2-2 | `lambda_q` (quantization loss) | 0.0, 0.01, 0.05, 0.1, 0.2 | Find best quantization loss weight |
| V2-3 | Routing on/off | on vs off at p_max {0.2, 0.5, 0.7} | Measure routing module impact |
| V2-4 | Golden config | best params, 3 seeds | Final v2 configuration with variance |

Each stage sweeps across INT8 with and without Hadamard transform.

### Run locally

```bash
# Dry run to see all experiments
python experiments/run_v2_ablations.py --dry_run

# Run all stages
python experiments/run_v2_ablations.py \
    --output_dir ./results/v2 \
    --dataset wikitext2 \
    --num_steps 2000 \
    --wandb

# Run specific stage
python experiments/run_v2_ablations.py --stage 1

# Override best values from prior stages
python experiments/run_v2_ablations.py \
    --stage 2 \
    --best_lambda_r 0.05
```

### Run on Modal (concurrent)

```bash
# All stages, all experiments launched in parallel per stage
modal run experiments/modal_run_experiments.py --wandb

# Specific stage
modal run experiments/modal_run_experiments.py --stage 1

# Dry run
modal run experiments/modal_run_experiments.py --dry-run
```

### Paper Ablation Runner (v1, 6 stages)

The original paper ablation suite is still available:

```bash
cd experiments/bitskip_wikitext2

# Run all 6 stages with 4 parallel workers
python run_experiments_parallel.py

# View results
python view_results.py --best
```

---

## Modal Cloud Training

The Modal scripts provide one-command cloud GPU training on NVIDIA B200 GPUs.

### Single experiment

```bash
modal run experiments/modal_train.py \
    --model-id my_run \
    --dataset wikitext2 \
    --precision int8 \
    --use-hadamard \
    --early-exit-lambda 0.3 \
    --p-max 0.5 \
    --lambda-q 0.05 \
    --lambda-r 0.05 \
    --batch-size 128 \
    --num-steps 5000 \
    --wandb
```

### Batch experiments

```bash
modal run experiments/modal_run_experiments.py \
    --stage 1 \
    --num-steps 5000 \
    --wandb
```

### Infrastructure

- **GPU**: NVIDIA B200 (192 GB HBM3e)
- **Storage**: `modal.Volume` named `bitskip-data` for checkpoints and cached datasets
- **Secrets**: `wandb-secret` for Weights & Biases API key
- **Batch size**: 128 recommended for B200 (vs 16 for local)
- **torch.compile**: Enabled by default with `max-autotune` mode

---

## Loss Components

### Paper Loss (v1)

```
L_total = L_main + lambda * L_early_exit
```

where `lambda = 0.3` (default).

### V2 Extended Loss

```
L_total = L_main + lambda * L_early_exit + lambda_q * L_quant + lambda_r * L_route
```

### Individual Components

**1. Task Loss (`L_main`)**

Standard causal language modeling cross-entropy with shifted labels.

**2. Early Exit Loss (`L_early_exit`)**

Intermediate supervision at each transformer layer using a shared LM head:

```
L_early_exit = sum(w_i * CE(lm_head(h_i), y))   for i = 1..L
```

Layer weights follow the paper formula: `w_i = (i+1)/L`, normalized so `sum(w_i) = 1`. Later layers receive proportionally more weight.

**3. Quantization Loss (`L_quant`) — v2**

MSE between full-precision and quantized weights across all BitLinear layers:

```
L_quant = (1/N) * sum(MSE(W, Q(W)))
```

where `Q(W)` is the ternary quantization: values above `0.5 * mean(|W|)` map to +1, below `-0.5 * mean(|W|)` map to -1, rest to 0, scaled by `mean(|W|)`.

**4. Routing Loss (`L_route`) — v2**

Target cost loss encouraging the model to exit at a target layer on average:

```
p_exit_at_l = p_exit_l * prod(1 - p_exit_i, i < l)
expected_exit = sum(l * p_exit_at_l)
L_route = MSE(expected_exit, target_exit_layer)
```

where `target_exit_layer = num_layers / 2` by default.

---

## Technical Details

### Weight Quantization (Ternary)

```python
alpha = mean(|W|)
W_q[W >  0.5 * alpha] = +1
W_q[W < -0.5 * alpha] = -1
W_q[otherwise]         =  0
W_q = W_q * alpha  # rescale
```

Gradients flow through the Straight-Through Estimator (STE): `W_q = W - W.detach() + W_q.detach()`.

### Activation Quantization

Per-token dynamic scaling to N-bit integer range:

```python
s = max(|x|, dim=-1)          # per-token scale
max_val = 2^(bits-1) - 1      # 127 for 8-bit, 7 for 4-bit
x_int = round(x * max_val / s).clamp(-max_val, max_val)
x_q = x_int * s / max_val     # dequantize
```

- **BitLinear**: 8-bit activations (default)
- **H-BitLinear**: 4-bit activations with Hadamard transform pre/post

### Hadamard Transform (H-BitLinear)

```
Forward: Input -> Pad(power-of-2) -> LayerNorm -> Quantize(4-bit) -> FWHT
         -> QuantizeWeights(ternary) -> Linear -> InverseFWHT -> Unpad
```

The Fast Walsh-Hadamard Transform uses O(n log n) complexity with a custom CUDA kernel. Falls back to PyTorch when CUDA is unavailable. Scaling factor: `1/sqrt(n)`.

### Layer Skipping

Three dropout schedules determine per-layer skip probability during training:

| Schedule | Formula | Shape |
|----------|---------|-------|
| Quadratic | `p(l) = p_max * (l/L)^2` | Aggressive skip of later layers |
| Linear | `p(l) = p_max * (l/L)` | Uniform increase |
| Uniform | `p(l) = p_max` | Equal skip probability |

Layer skipping is **training-only**. During inference, all layers execute (early exit handles compute savings).

### Routing Module

Each transformer block has a learnable gating network:

```
Gate(x) = sigmoid(Linear(ReLU(Linear(LayerNorm(x)))))
```

- **Training**: Bernoulli sampling with STE (`z_hard.detach() + z_soft - z_soft.detach()`)
- **Inference**: Threshold decision (`p_exit > 0.5`)

### Grouped Query Attention

12 query heads share 4 key/value head groups (3:1 ratio), reducing KV cache memory by 3x. Position encoding uses Rotary Position Embeddings (RoPE).

---

## Configuration Reference

### Model Architecture (125M preset)

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 50257 (GPT-2 tokenizer) |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `num_kv_heads` | 4 |
| `head_dim` | 64 |
| `mlp_ratio` | 4.0 (intermediate = 3072) |
| `max_position_embeddings` | 512 |
| `weight_bits` | 2 (ternary) |
| `activation_bits` | 8 (BitLinear) / 4 (H-BitLinear) |

### Paper Naming Convention

| Config ID | Weights | Activations | Hadamard |
|-----------|---------|-------------|----------|
| FP16-Baseline | FP16 | FP16 | No |
| BitSkip-W1.58A8 | Ternary | 8-bit | No |
| BitSkip-W1.58A8-H | Ternary | 8-bit | Yes |
| BitSkip-W1.58A4-H | Ternary | 4-bit | Yes |

### Default Loss Weights

| Parameter | Paper (v1) | Code Default | Description |
|-----------|-----------|--------------|-------------|
| `early_exit_lambda` | 0.3 | 0.0 | Early exit loss weight |
| `lambda_q` | — | 0.0 | Quantization loss weight (v2) |
| `lambda_r` | — | 0.0 | Routing loss weight (v2) |

---

## Evaluation and Benchmarking

### Inference Engine

The inference engine (`bitnet/inference/engine.py`) supports:

- SafeTensors checkpoint loading with auto model detection
- Early exit inference at a specified layer
- Self-speculative decoding (draft with early exit, verify with full model)
- Temperature, top-k, top-p sampling with repetition penalty

```python
from bitnet.inference.engine import BitNetInferenceEngine

engine = BitNetInferenceEngine("path/to/model.safetensors")
logits = engine.model.early_exit_inference(input_ids, exit_layer=6)
```

### Benchmark Scripts

```bash
# HuggingFace model benchmarks
python evals/bitnet_huggingface_benchmark.py

# Layer skip benchmarks
python evals/huggingface_layerskip_benchmark.py

# Inference timing at different exit thresholds
python evals/inference_benchmark_1b_improved.py
```

Benchmark results are stored as JSON files in `results/`.

### View Experiment Results

```bash
cd experiments/bitskip_wikitext2
python view_results.py           # All results
python view_results.py --best    # Best per stage
python view_results.py --summary # Summary statistics
```

---

## Citation

```bibtex
@misc{bitskip,
  title={BitSkip: An Empirical Analysis of Quantization and Early Exit Composition},
  author={Ramshankar Bhuvaneswaran and Handan Liu},
  year={2025},
  url={https://arxiv.org/abs/2510.23766}
}
```

## License

This project is licensed under the MIT License.
