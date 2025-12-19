# BitSkip Experiment Plan (WikiText-2)

This directory contains experiments to investigate the interaction between quantization and early-exit mechanisms in transformer language models.

## Structure

- `train.py`: Main training script supporting all configurations.
- `run_experiments.py`: Script to generate and run all experiment commands.
- `results/`: Directory to store logs and checkpoints.

## Experiments

1. **Baselines**: FP16, INT8, INT4 (w/ and w/o Hadamard) - No Early Exit.
2. **Early Exit Loss Weight (λ)**: Ablation of λ [0.0 - 0.7].
3. **Dropout Probability (p_max)**: Ablation of p_max [0.0 - 0.7].
4. **Dropout Schedule**: Quadratic vs Linear vs Uniform.
5. **Full Comparison**: Best combos of the above.
6. **Hadamard Analysis**: Analysis of Hadamard transform effects.

## Usage

To run all experiments:
```bash
python3 run_experiments.py
```

To run a single experiment:
```bash
python3 train.py --model_id B1 --precision fp16 --no_early_exit
```
