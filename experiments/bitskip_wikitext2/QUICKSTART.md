# Quick Start Guide - Parallel Experiment Runner

## Simple Usage

### Run All Stages Concurrently

```bash
# Basic usage (4 parallel workers)
python run_experiments_parallel.py

# With more workers (if you have more GPUs/CPUs)
python run_experiments_parallel.py --max_workers 8

# With torch.compile for faster training
python run_experiments_parallel.py --compile
```

### Run Specific Stage

```bash
# Run only stage 1 (Baselines)
python run_experiments_parallel.py --stage 1

# Run only stage 2 (Lambda Ablation)
python run_experiments_parallel.py --stage 2
```

## View Results

### View All Results
```bash
python view_results.py
```

### View Best Results Per Stage
```bash
python view_results.py --best
```

### View Summary Statistics
```bash
python view_results.py --summary
```

### Clear Cache (if needed)
```bash
python view_results.py --clear
```

## How It Works

1. **Stage 1** runs all baseline experiments in parallel
2. **Stage 2** runs lambda ablation experiments in parallel
3. **Stage 3** waits for Stage 2, reads best lambda from cache, runs p_max ablation
4. **Stage 4** waits for Stage 3, reads best lambda & p_max, runs schedule ablation
5. **Stage 5** waits for Stage 4, reads all best params, runs full comparison
6. **Stage 6** is a placeholder for analysis

## Key Features

- ✅ **Automatic Resume**: Skips already completed experiments
- ✅ **Results Caching**: Fast parameter selection from JSON cache
- ✅ **Progress Tracking**: See experiments complete in real-time
- ✅ **Error Handling**: Continues even if some experiments fail

## Example Workflow

```bash
# 1. Start all experiments
python run_experiments_parallel.py --max_workers 4

# 2. Check progress (in another terminal)
python view_results.py --summary

# 3. View best results so far
python view_results.py --best

# 4. If interrupted, just run again - it will skip completed experiments
python run_experiments_parallel.py --max_workers 4
```

## Performance

With 4 parallel workers, expect:
- **Stage 1**: ~5-10 minutes (5 experiments)
- **Stage 2**: ~30-60 minutes (12 experiments)
- **Stage 3**: ~20-40 minutes (8 experiments)
- **Stage 4**: ~15-30 minutes (6 experiments)
- **Stage 5**: ~45-90 minutes (15 experiments)

**Total**: ~2-4 hours (vs 8-16 hours sequential)

