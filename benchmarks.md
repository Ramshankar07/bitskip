# BitSkip V2 Ablation Benchmarks

**Dataset:** WikiText-2, 500 training steps
**Hardware:** Modal B200 GPU
**Branch:** `v2-Implementation`
**Date:** 2026-03-04

## Summary


| Run             | Model        | Fix Applied                           | noH Best Val PPL | H Best Val PPL | H Status              |
| --------------- | ------------ | ------------------------------------- | ---------------- | -------------- | --------------------- |
| Run 1 (pre-fix) | 125M (768h)  | None (no causal mask)                 | ~16 (invalid)    | ~1,794         | Collapsed             |
| Run 2           | 125M (768h)  | Causal mask + H-BitLinear fixes       | **251.11**       | ~1,494         | Collapsed             |
| Run 3           | 85M_H (512h) | activation_bits fix + power-of-2 dims | **252.93**       | ~1,491         | Collapsed (unchanged) |
| Run 4           | 85M_H (512h) | **squared_relu added to HBitLinear**  | —                | **957.03**     | **Learning (fixed!)** |


**Run 4 — Root cause found and fixed:**

The "Hadamard collapse" was caused by a **missing activation function** in `HBitLinear`. `BitLinear` (used by noH models) applies `squared_relu` at the end of every projection — including attention q/k/v/o. `HBitLinear` had no activation, causing every projection output to lack the non-linearity that the architecture relied on.

**Fix:** Added `squared_relu` to the end of `HBitLinear.forward()`, matching `BitLinear`'s behavior.

**Result:** H model PPL dropped from ~1,503 (collapsed) to **957** (early-stopped at step 400). The model is now genuinely learning. Gap to noH (254) remains — likely needs hyperparameter tuning and longer training now that the architecture is functional.

**Diagnostic tests that isolated the root cause:**

1. FWHT unit test: H(H(x))=x passed for all sizes (kernel is correct)
2. H model with no quantization: still collapsed (quant not the cause)
3. H model with no output FWHT + standard init: still collapsed (FWHT not the cause)
4. H model with ALL Hadamard + quant disabled (HBitLinear = LayerNorm→Linear): **still collapsed** — proved the bug was in the Model2 architecture, not HBitLinear's special features
5. Side-by-side diff of BitLinear vs HBitLinear revealed the missing `squared_relu`

---

## Run 4: squared_relu Fix (Root Cause of Hadamard Collapse)

**Model:** 85M_H (512 hidden, 12 layers, 8 heads, 4 KV heads, 2048 FFN)
**Fix:** Added `squared_relu` activation to the end of `HBitLinear.forward()`.


| Experiment          | H   | Config                     | Seed | Best Val PPL | Final Val PPL | Steps            | Notes                   |
| ------------------- | --- | -------------------------- | ---- | ------------ | ------------- | ---------------- | ----------------------- |
| V2_sqrelu_fix_H_s42 | Yes | lambda_r=0.0, lambda_q=0.0 | 42   | **957.03**   | **957.03**    | 400 (early stop) | **No longer collapsed** |


**Comparison (same config, lambda_r=0.0):**

- noH (Run 3): Final Val PPL **254.56**
- H before fix (Run 3): Final Val PPL **1,503.47** (collapsed)
- H after fix (Run 4): Final Val PPL **957.03** (36% improvement, learning)

---

## Composition Ablation (4-way) + Weight Insights

**Setup:** 85M_H, WikiText-2, best config (lambda_r=0.2, lambda_q=0.05, p_max=0.7, early_exit_lambda=0.3), seed 7. 10k max steps, eval every 100. Models pushed to Hugging Face (Ram07/*, private) with weight analysis.

### Composition results


| Config                            | Best Val PPL | Steps |
| --------------------------------- | ------------ | ----- |
| Baseline (no H, no EE)            | 228.77       | 1800  |
| H-only (Hadamard, no EE)          | **185.46**   | 1800  |
| EE-only (no Hadamard, early exit) | 252.37       | 2100  |
| BitSkip (H + EE)                  | 216.06       | 1800  |


**Takeaway:** H-only achieves best PPL (185.46); EE-only is worst (252.37). BitSkip (H+EE) sits between baseline and H-only.

### Weight insights (per model)


| Model            | Best Val PPL | keys | params     | scale [min, max] | mean_scale | Sparsity (mean / max) | layers_high(≥50%) | Ternary frac +1 | Ternary frac −1 |
| ---------------- | ------------ | ---- | ---------- | ---------------- | ---------- | --------------------- | ----------------- | --------------- | --------------- |
| comp_baseline_s7 | 228.77       | 72   | 34,603,008 | [0.0010, 0.0215] | 0.005125   | 30.40% / 57.19%       | 4                 | 34.01%          | 35.60%          |
| comp_H_only_s7   | 185.46       | 72   | 34,603,008 | [0.0053, 0.0230] | 0.015055   | 35.59% / 40.03%       | 0                 | 32.05%          | 32.36%          |
| comp_EE_only_s7  | 252.37       | 72   | 34,603,008 | [0.0010, 0.0281] | 0.003648   | 26.85% / 48.65%       | 0                 | 36.00%          | 37.15%          |
| comp_BitSkip_s7  | 216.06       | 72   | 34,603,008 | [0.0020, 0.0247] | 0.012157   | 35.26% / 46.30%       | 0                 | 32.32%          | 32.42%          |


**Analysis:**

- **H-only** has highest mean scale (0.0151) and no high-sparsity layers; scales are in a tighter band — consistent with Hadamard rotation.
- **Baseline** has lowest mean scale (0.0051), highest max sparsity (57.19%), and 4 layers with ≥50% sparsity.
- **EE-only** has lowest mean sparsity (26.85%) and most asymmetric ternary (36.0% +1, 37.15% −1).
- **BitSkip** is between baseline and H-only on scale and sparsity; ternary is nearly balanced (32.32% / 32.42%).

```text
====================================================================================
================
COMPOSITION ABLATION — FULL-PRECISION vs 1.5-BIT COMPARISON
====================================================================================
================
Config                         FP PPL  1.5b PPL   Δ PPL%  Mean Scale  Sparsity  
Hi-Sparse
------------------------------------------------------------------------------------
----------------
Baseline (no H, no EE)         228.76    451.03  +97.17%    0.005125    30.4%       
4
H-only (Hadamard)              185.45    389.65 +110.11%    0.015055    35.6%       
0
EE-only (Early Exit)           252.34    478.38  +89.58%    0.003648    26.9%       
0
BitSkip (H + EE)               216.06    764.36 +253.76%    0.012157    35.3%       
0
====================================================================================
================

WEIGHT DISTRIBUTION (ternary bucket fractions)
====================================================================================
================
Config                        +1 frac  -1 frac   0 frac  Min Scale  Max Scale  
#Layers
------------------------------------------------------------------------------------
----------------
Baseline (no H, no EE)         34.0%   35.6%   30.4%   0.001040   0.021546       72
H-only (Hadamard)              32.0%   32.4%   35.6%   0.005340   0.022986       72
EE-only (Early Exit)           36.0%   37.1%   26.9%   0.001016   0.028080       72
BitSkip (H + EE)               32.3%   32.4%   35.3%   0.002011   0.024732       72
====================================================================================
================

EARLY EXIT LAYER PPL — BitSkip (H + EE)
====================================================================================
================
Exit Layer       Loss        PPL
------------------------------------------------------------------------------------
----------------
         0    11.2706   78480.44
         1    11.3689   86584.36
         2    11.3250   82870.06
         3    11.2501   76888.66
         4    11.1302   68197.78
         5    10.9687   58026.51
         6    10.5851   39540.34
         7    10.1505   25602.75
         8     9.9153   20237.07
         9     9.6755   15922.90
        10     9.4269   12417.69
        11     5.3756     216.06
====================================================================================
================

Best full-precision PPL : H-only (Hadamard) (185.45)
Best 1.5-bit PPL       : H-only (Hadamard) (389.65)
Least PPL degradation  : EE-only (Early Exit) (+89.58%)
```

```text
Preparing wikitext2 validation set...
  Model config: size=85M_H, hadamard=True
  Validation batches: 31

Evaluating full-precision checkpoint...
Using BitNetModel2 (Hadamard)
  Loaded 415/415 model keys
  Eval full-precision: 100%|██████████| 31/31 [00:05<00:00,  5.19batch/s]
  Full-precision PPL: 197.87  (loss=5.2876)

Evaluating 1.5-bit (ternary) checkpoint...
Using BitNetModel2 (Hadamard)
  Loaded 415/415 model keys
  Eval 1.5-bit: 100%|██████████| 31/31 [00:02<00:00, 13.92batch/s]
  1.5-bit PPL: 490.17  (loss=6.1948)

Analyzing early exit PPL per layer (0..11)...
Using BitNetModel2 (Hadamard)
  Eval exit_layer=0: 100%|██████████| 31/31 [00:00<00:00, 47.93batch/s]
  Exit layer  0: PPL=2334.09 loss=7.7554
Using BitNetModel2 (Hadamard)
  Eval exit_layer=1: 100%|██████████| 31/31 [00:00<00:00, 40.12batch/s]
  Exit layer  1: PPL=2689.45 loss=7.8971
Using BitNetModel2 (Hadamard)
  Eval exit_layer=2: 100%|██████████| 31/31 [00:00<00:00, 31.02batch/s]
  Exit layer  2: PPL=2949.96 loss=7.9895
Using BitNetModel2 (Hadamard)
  Eval exit_layer=3: 100%|██████████| 31/31 [00:01<00:00, 27.31batch/s]
  Exit layer  3: PPL=3155.34 loss=8.0569
Using BitNetModel2 (Hadamard)
  Eval exit_layer=4: 100%|██████████| 31/31 [00:01<00:00, 24.93batch/s]
  Exit layer  4: PPL=3252.79 loss=8.0873
Using BitNetModel2 (Hadamard)
  Eval exit_layer=5: 100%|██████████| 31/31 [00:01<00:00, 22.35batch/s]
  Exit layer  5: PPL=3239.28 loss=8.0831
Using BitNetModel2 (Hadamard)
  Eval exit_layer=6: 100%|██████████| 31/31 [00:01<00:00, 20.45batch/s]
  Exit layer  6: PPL=3159.02 loss=8.0580
Using BitNetModel2 (Hadamard)
  Eval exit_layer=7: 100%|██████████| 31/31 [00:01<00:00, 18.55batch/s]
  Exit layer  7: PPL=3064.22 loss=8.0275
Using BitNetModel2 (Hadamard)
  Eval exit_layer=8: 100%|██████████| 31/31 [00:01<00:00, 16.77batch/s]
  Exit layer  8: PPL=2948.15 loss=7.9889
Using BitNetModel2 (Hadamard)
  Eval exit_layer=9: 100%|██████████| 31/31 [00:02<00:00, 15.45batch/s]
  Exit layer  9: PPL=2816.23 loss=7.9432
Using BitNetModel2 (Hadamard)
  Eval exit_layer=10:  87%|████████▋ | 27/31 [00:01<00:00, 17.11batch/s]⠙ Running (2
  Eval exit_layer=10: 100%|██████████| 31/31 [00:02<00:00, 14.96batch/s]
  Exit layer 10: PPL=2660.59 loss=7.8863
Using BitNetModel2 (Hadamard)
  Eval exit_layer=11: 100%|██████████| 31/31 [00:02<00:00, 14.06batch/s]
  Exit layer 11: PPL=197.87 loss=5.2876

============================================================
RESULTS: Ram07/ls_base_H_EE_s7
============================================================
  Full-precision PPL : 197.87
  1.5-bit PPL        : 490.17
  PPL degradation    : +147.72%
  Mean ternary scale : 0.010327
  Mean sparsity      : 35.13%
  High-sparsity layers (≥50%): 0
============================================================
      FP PPL=197.87 1.5b PPL=490.17
```

```text
Final Validation Perplexity: 233.89
Result: best_val_ppl=233.89 (steps=1700)
Pushing comp_EE_only_s7 -> Ram07/comp_EE_only_s7 ...
Processing Files (2 / 2)      : 100%|██████████|  697MB /  697MB, 11.3MB/s  
Processing Files (2 / 2)      : 100%|██████████|  697MB /  697MB, 1.96MB/s  
New Data Upload               : 100%|██████████|  569MB /  569MB, 1.96MB/s  
  .../model_1.5bit.safetensors: 100%|██████████|  349MB /  349MB            
  ...bbzqsba/model.safetensors: 100%|██████████|  349MB /  349MB            
  HF upload OK. Summary: {'quantizable_keys': 72, 'total_quantizable_params': 
34603008, 'mean_scale': 0.0035052261179468283, 'min_scale': 0.001032170606777072, 
'max_scale': 0.021438241004943848, 'mean_sparsity': 0.27162836657630074, 
'max_sparsity': 0.45767974853515625, 'layers_high_sparsity': 0, 'mean_frac_pos': 
0.36006059911515975, 'mean_frac_neg': 0.3683110343085395}
```

---

## Run 3: activation_bits Fix + 85M_H Model (power-of-2 dims)

**Model:** 85M_H (512 hidden, 12 layers, 8 heads, 4 KV heads, 2048 FFN)

### Fixes Applied

- `**gqa_attention2.py`:** All 4 HBitLinear projections (q/k/v/o_proj) now pass `activation_bits=activation_bits`. Previously defaulted to 4-bit regardless of config.
- **Model size:** Switched to `85M_H` — all dimensions are powers of 2 (512, 2048), so HBitLinear has **zero padding**. This eliminates the padding-corruption hypothesis.

### Stage 1: Routing Loss (lambda_r) Sweep

Fixed: lambda_q=0.0, best lambda_r carried forward.


| Experiment                   | H   | lambda_r | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes             |
| ---------------------------- | --- | -------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | ----------------- |
| V2_LambdaR_0.0_int8_noH_s42  | No  | 0.0      | 42   | 4.26       | 287.30    | 287.34       | 254.56        | 1.16   |                   |
| V2_LambdaR_0.01_int8_noH_s42 | No  | 0.01     | 42   | 4.27       | 288.60    | 288.64       | 254.27        | 1.19   |                   |
| V2_LambdaR_0.05_int8_noH_s42 | No  | 0.05     | 42   | 4.26       | 289.30    | 289.35       | 259.17        | 1.19   |                   |
| V2_LambdaR_0.1_int8_noH_s42  | No  | 0.1      | 42   | 4.30       | 288.20    | 288.24       | 257.29        | 1.18   |                   |
| V2_LambdaR_0.2_int8_noH_s42  | No  | 0.2      | 42   | 4.21       | 283.50    | 283.48       | 251.11        | 1.18   | **Best lambda_r** |
| V2_LambdaR_0.0_int8_H_s42    | Yes | 0.0      | 42   | 9.20       | 1,803.40  | 1,803.39     | 1,503.47      | 1.86   | Collapsed         |
| V2_LambdaR_0.01_int8_H_s42   | Yes | 0.01     | 42   | 9.34       | 1,797.30  | 1,797.34     | 1,566.13      | 1.84   | Collapsed         |
| V2_LambdaR_0.05_int8_H_s42   | Yes | 0.05     | 42   | 9.64       | 1,799.60  | 1,799.64     | 1,789.28      | 1.88   | Collapsed         |
| V2_LambdaR_0.1_int8_H_s42    | Yes | 0.1      | 42   | 9.64       | 1,800.90  | 1,800.86     | 1,784.72      | 1.84   | Collapsed         |
| V2_LambdaR_0.2_int8_H_s42    | Yes | 0.2      | 42   | 9.64       | 1,793.40  | 1,793.45     | 1,788.82      | 1.85   | Collapsed         |


### Stage 2: Quantization Loss (lambda_q) Sweep

Fixed: lambda_r=best from Stage 1, sweep lambda_q.


| Experiment                   | H   | lambda_q | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes             |
| ---------------------------- | --- | -------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | ----------------- |
| V2_LambdaQ_0.0_int8_noH_s42  | No  | 0.0      | 42   | 4.26       | 289.30    | 289.35       | 259.17        | 1.18   |                   |
| V2_LambdaQ_0.01_int8_noH_s42 | No  | 0.01     | 42   | 4.24       | 289.50    | 289.48       | 255.01        | 1.31   |                   |
| V2_LambdaQ_0.05_int8_noH_s42 | No  | 0.05     | 42   | 4.26       | 284.20    | 284.24       | 254.10        | 1.19   | **Best lambda_q** |
| V2_LambdaQ_0.1_int8_noH_s42  | No  | 0.1      | 42   | 4.26       | 286.00    | 285.95       | 254.29        | 1.22   |                   |
| V2_LambdaQ_0.2_int8_noH_s42  | No  | 0.2      | 42   | 4.28       | 291.60    | 291.57       | 257.89        | 1.22   |                   |
| V2_LambdaQ_0.0_int8_H_s42    | Yes | 0.0      | 42   | 9.64       | 1,795.70  | 1,795.65     | 1,792.12      | 1.98   | Collapsed         |
| V2_LambdaQ_0.01_int8_H_s42   | Yes | 0.01     | 42   | 9.64       | 1,795.60  | 1,795.59     | 1,793.58      | 1.87   | Collapsed         |
| V2_LambdaQ_0.05_int8_H_s42   | Yes | 0.05     | 42   | 9.64       | 1,798.10  | 1,798.08     | 1,788.50      | 1.87   | Collapsed         |
| V2_LambdaQ_0.1_int8_H_s42    | Yes | 0.1      | 42   | 9.64       | 1,794.80  | 1,794.80     | 1,793.04      | 1.85   | Collapsed         |
| V2_LambdaQ_0.2_int8_H_s42    | Yes | 0.2      | 42   | 9.64       | 1,796.60  | 1,796.58     | 1,789.71      | 2.03   | Collapsed         |


### Stage 3: Routing Configuration Sweep


| Experiment                            | H   | Routing | Skip Prob | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes            |
| ------------------------------------- | --- | ------- | --------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | ---------------- |
| V2_Routing_routeOFF_p0.7_int8_noH_s42 | No  | OFF     | 0.7       | 42   | 4.34       | 296.10    | 296.08       | 263.21        | 1.05   |                  |
| V2_Routing_routeOFF_p0.5_int8_noH_s42 | No  | OFF     | 0.5       | 42   | 4.26       | 287.30    | 287.34       | 254.56        | 1.15   |                  |
| V2_Routing_routeOFF_p0.2_int8_noH_s42 | No  | OFF     | 0.2       | 42   | 4.30       | 293.30    | 293.30       | 259.85        | 1.36   |                  |
| V2_Routing_routeON_p0.7_int8_noH_s42  | No  | ON      | 0.7       | 42   | 4.27       | 286.10    | 286.06       | 252.93        | 1.16   | **Best overall** |
| V2_Routing_routeON_p0.5_int8_noH_s42  | No  | ON      | 0.5       | 42   | 4.26       | 289.30    | 289.35       | 259.17        | 1.30   |                  |
| V2_Routing_routeON_p0.2_int8_noH_s42  | No  | ON      | 0.2       | 42   | 4.23       | 290.30    | 290.34       | 253.43        | 1.24   |                  |
| V2_Routing_routeOFF_p0.7_int8_H_s42   | Yes | OFF     | 0.7       | 42   | 9.24       | 1,772.40  | 1,772.37     | 1,491.62      | 1.67   | Collapsed        |
| V2_Routing_routeOFF_p0.5_int8_H_s42   | Yes | OFF     | 0.5       | 42   | 9.34       | 1,802.50  | 1,802.55     | 1,549.70      | 1.82   | Collapsed        |
| V2_Routing_routeOFF_p0.2_int8_H_s42   | Yes | OFF     | 0.2       | 42   | 9.26       | 1,798.10  | 1,798.09     | 1,497.70      | 1.94   | Collapsed        |
| V2_Routing_routeON_p0.7_int8_H_s42    | Yes | ON      | 0.7       | 42   | 9.64       | 1,798.20  | 1,798.17     | 1,795.24      | 1.95   | Collapsed        |
| V2_Routing_routeON_p0.5_int8_H_s42    | Yes | ON      | 0.5       | 42   | 9.64       | 1,802.10  | 1,802.07     | 1,789.32      | 1.83   | Collapsed        |
| V2_Routing_routeON_p0.2_int8_H_s42    | Yes | ON      | 0.2       | 42   | 9.64       | 1,801.50  | 1,801.46     | 1,790.84      | 1.99   | Collapsed        |


---

## Run 2: Post Causal Mask + H-BitLinear Fixes

### Stage 1: Routing Loss (lambda_r) Sweep

Fixed: lambda_q=0.0, best lambda_r carried forward.


| Experiment                   | H   | lambda_r | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes             |
| ---------------------------- | --- | -------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | ----------------- |
| V2_LambdaR_0.0_int8_noH_s42  | No  | 0.0      | 42   | 4.26       | 287.30    | 287.34       | 254.56        | 1.17   | Baseline          |
| V2_LambdaR_0.01_int8_noH_s42 | No  | 0.01     | 42   | 4.27       | 288.60    | 288.64       | 254.27        | 1.20   |                   |
| V2_LambdaR_0.05_int8_noH_s42 | No  | 0.05     | 42   | 4.25       | 287.60    | 287.60       | 255.20        | 1.20   |                   |
| V2_LambdaR_0.1_int8_noH_s42  | No  | 0.1      | 42   | 4.30       | 288.20    | 288.24       | 257.29        | 1.21   |                   |
| V2_LambdaR_0.2_int8_noH_s42  | No  | 0.2      | 42   | 4.21       | 283.50    | 283.48       | 251.11        | 1.36   | **Best lambda_r** |
| V2_LambdaR_0.0_int8_H_s42    | Yes | 0.0      | 42   | 9.39       | 1,783.20  | 1,783.22     | 1,606.79      | 1.84   | Collapsed         |
| V2_LambdaR_0.01_int8_H_s42   | Yes | 0.01     | 42   | 9.37       | 1,795.70  | 1,795.72     | 1,588.95      | 1.85   | Collapsed         |
| V2_LambdaR_0.05_int8_H_s42   | Yes | 0.05     | 42   | 9.64       | 1,794.40  | 1,794.37     | 1,791.19      | 2.00   | Collapsed         |
| V2_LambdaR_0.1_int8_H_s42    | Yes | 0.1      | 42   | 9.64       | 1,800.90  | 1,800.88     | 1,786.28      | 1.97   | Collapsed         |
| V2_LambdaR_0.2_int8_H_s42    | Yes | 0.2      | 42   | 9.64       | 1,795.10  | 1,795.10     | 1,782.13      | 2.02   | Collapsed         |


### Stage 2: Quantization Loss (lambda_q) Sweep

Fixed: lambda_r=best from Stage 1, sweep lambda_q.


| Experiment                   | H   | lambda_q | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes             |
| ---------------------------- | --- | -------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | ----------------- |
| V2_LambdaQ_0.0_int8_noH_s42  | No  | 0.0      | 42   | 4.26       | 289.30    | 289.35       | 259.17        | 1.36   | Baseline          |
| V2_LambdaQ_0.01_int8_noH_s42 | No  | 0.01     | 42   | 4.24       | 289.50    | 289.48       | 255.01        | 1.22   |                   |
| V2_LambdaQ_0.05_int8_noH_s42 | No  | 0.05     | 42   | 4.26       | 284.20    | 284.24       | 254.10        | 1.21   | **Best lambda_q** |
| V2_LambdaQ_0.1_int8_noH_s42  | No  | 0.1      | 42   | 4.26       | 286.00    | 285.95       | 254.29        | 1.22   |                   |
| V2_LambdaQ_0.2_int8_noH_s42  | No  | 0.2      | 42   | 4.28       | 291.60    | 291.57       | 257.89        | 1.23   |                   |
| V2_LambdaQ_0.0_int8_H_s42    | Yes | 0.0      | 42   | 9.64       | 1,795.80  | 1,795.83     | 1,786.53      | 2.01   | Collapsed         |
| V2_LambdaQ_0.01_int8_H_s42   | Yes | 0.01     | 42   | 9.64       | 1,804.10  | 1,804.14     | 1,793.15      | 2.03   | Collapsed         |
| V2_LambdaQ_0.05_int8_H_s42   | Yes | 0.05     | 42   | 9.64       | 1,801.10  | 1,801.13     | 1,788.60      | 1.88   | Collapsed         |
| V2_LambdaQ_0.1_int8_H_s42    | Yes | 0.1      | 42   | 9.64       | 1,802.60  | 1,802.63     | 1,789.19      | 1.84   | Collapsed         |
| V2_LambdaQ_0.2_int8_H_s42    | Yes | 0.2      | 42   | 9.64       | 1,793.90  | 1,793.93     | 1,791.11      | 2.04   | Collapsed         |


### Stage 3: Routing Configuration Sweep


| Experiment                            | H   | Routing | Skip Prob | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes            |
| ------------------------------------- | --- | ------- | --------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | ---------------- |
| V2_Routing_routeOFF_p0.7_int8_noH_s42 | No  | OFF     | 0.7       | 42   | 4.26       | 284.40    | 284.39       | 251.43        | 1.06   | **Best overall** |
| V2_Routing_routeOFF_p0.5_int8_noH_s42 | No  | OFF     | 0.5       | 42   | 4.26       | 287.30    | 287.34       | 254.56        | 1.27   |                  |
| V2_Routing_routeOFF_p0.2_int8_noH_s42 | No  | OFF     | 0.2       | 42   | 4.30       | 293.30    | 293.30       | 259.85        | 1.22   |                  |
| V2_Routing_routeON_p0.7_int8_noH_s42  | No  | ON      | 0.7       | 42   | 4.27       | 286.10    | 286.06       | 252.93        | 1.16   |                  |
| V2_Routing_routeON_p0.5_int8_noH_s42  | No  | ON      | 0.5       | 42   | 4.27       | 285.50    | 285.50       | 254.80        | 1.20   |                  |
| V2_Routing_routeON_p0.2_int8_noH_s42  | No  | ON      | 0.2       | 42   | 4.23       | 290.30    | 290.34       | 253.43        | 1.25   |                  |
| V2_Routing_routeOFF_p0.7_int8_H_s42   | Yes | OFF     | 0.7       | 42   | 9.31       | 1,794.70  | 1,794.67     | 1,527.32      | 1.66   | Collapsed        |
| V2_Routing_routeOFF_p0.5_int8_H_s42   | Yes | OFF     | 0.5       | 42   | 9.27       | 1,792.60  | 1,792.56     | 1,495.19      | 1.98   | Collapsed        |
| V2_Routing_routeOFF_p0.2_int8_H_s42   | Yes | OFF     | 0.2       | 42   | 9.28       | 1,795.30  | 1,795.30     | 1,529.59      | 1.90   | Collapsed        |
| V2_Routing_routeON_p0.7_int8_H_s42    | Yes | ON      | 0.7       | 42   | 9.64       | 1,797.80  | 1,797.79     | 1,795.50      | 1.92   | Collapsed        |
| V2_Routing_routeON_p0.5_int8_H_s42    | Yes | ON      | 0.5       | 42   | 9.64       | 1,794.10  | 1,794.07     | 1,790.61      | 1.84   | Collapsed        |
| V2_Routing_routeON_p0.2_int8_H_s42    | Yes | ON      | 0.2       | 42   | 9.64       | 1,797.30  | 1,797.33     | 1,789.66      | 1.95   | Collapsed        |


### Stage 4: Golden Config (H only, multi-seed)


| Experiment            | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes     |
| --------------------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | --------- |
| V2_Golden_int8_H_s42  | 42   | 9.64       | 1,794.90  | 1,794.94     | 1,789.00      | 1.92   | Collapsed |
| V2_Golden_int8_H_s123 | 123  | 9.66       | 1,797.00  | 1,796.97     | 1,790.85      | 1.86   | Collapsed |
| V2_Golden_int8_H_s456 | 456  | 9.48       | 1,803.20  | 1,803.20     | 1,795.51      | 1.86   | Collapsed |


---

## Run 1: Pre Causal Mask Fix (INVALID — bidirectional attention)

> **These results are invalid.** The model had no causal mask, so it attended to future tokens.
> PPL numbers are artificially low. Kept for reference only.

### Stage 1: lambda_r Sweep (Pre-Fix)


| Experiment                   | H   | lambda_r | Seed | Train Loss | Train PPL | Best Val PPL | Final Val PPL | s/step | Notes                     |
| ---------------------------- | --- | -------- | ---- | ---------- | --------- | ------------ | ------------- | ------ | ------------------------- |
| V2_LambdaR_0.0_int8_noH_s42  | No  | 0.0      | 42   | 1.97       | 34.90     | 34.86        | 8.48          | 1.12   | Baseline (no routing reg) |
| V2_LambdaR_0.01_int8_noH_s42 | No  | 0.01     | 42   | 2.02       | 36.00     | 36.03        | 10.09         | 1.23   |                           |
| V2_LambdaR_0.05_int8_noH_s42 | No  | 0.05     | 42   | 1.86       | 27.10     | 27.13        | 8.82          | 1.32   | Best lambda_r             |
| V2_LambdaR_0.1_int8_noH_s42  | No  | 0.1      | 42   | 2.28       | 37.70     | 37.69        | 14.41         | 1.15   |                           |
| V2_LambdaR_0.2_int8_noH_s42  | No  | 0.2      | 42   | 2.80       | 68.80     | 68.79        | 21.86         | 1.16   |                           |


### Stage 3: Routing Sweep (Pre-Fix)


| Experiment                            | H   | Routing | Skip Prob | Seed | Train Loss | Best Val PPL | Final Val PPL | Notes                  |
| ------------------------------------- | --- | ------- | --------- | ---- | ---------- | ------------ | ------------- | ---------------------- |
| V2_Routing_routeOFF_p0.7_int8_noH_s42 | No  | OFF     | 0.7       | 42   | 1.49       | 16.17        | 5.52          | Best overall (INVALID) |
| V2_Routing_routeOFF_p0.2_int8_noH_s42 | No  | OFF     | 0.2       | 42   | 1.36       | 17.20        | 5.58          |                        |


---

## Fixes Applied Between Run 1 and Run 2

1. **Causal mask added** to `gqa_attention.py` and `gqa_attention2.py` — prevents attending to future tokens
2. **Early exit utilities extracted** to `early_exit.py` — deduplication from model.py/model2.py
3. **H-BitLinear LayerNorm** — moved to unpadded dimensions (768 instead of 1024)
4. **H-BitLinear op order** — Hadamard now applied before quantization (was reversed)
5. **H-BitLinear weight init** — added 0.1x scaling to match BitLinear

---

## Key Observations

### Causal Mask Validation

- noH Best Val PPL: ~16 (Run 1, no causal mask) -> ~284 (Run 2, legitimate)
- Confirms causal mask is correctly enforced

### noH Model (Working)

- Remarkably stable across all hyperparameter settings (Best Val PPL 283-296)
- lambda_r and lambda_q have minimal impact at 500 steps — model is undertrained
- Best config: route ON, p=0.7, lambda_r=0.2, lambda_q=0.05 -> **Final Val PPL 252.93**
- Higher skip probability (p=0.7) consistently outperforms lower (p=0.2)

### H Model — Root Cause Found (Run 4)

- **Root cause:** `HBitLinear` was missing the `squared_relu` activation that `BitLinear` applies at the end of every projection. This meant all attention Q/K/V/O projections and FFN up/down projections lacked the non-linearity the architecture depends on.
- **Fix:** Added `squared_relu` to the end of `HBitLinear.forward()` in `h_bitlinear.py`.
- **Run 4 result:** H model PPL **957** (early-stopped at step 400) — down from ~1,503 (collapsed). The model is now learning.
- **Remaining gap:** H model PPL 957 vs noH best PPL 252. Likely causes:
  - The FWHT + quantization pipeline adds overhead that needs more steps or tuning
  - The output FWHT (sandwich pattern) may still be suboptimal — the paper applies FWHT only to the input
  - Hyperparameters (LR, batch size, steps) were tuned for noH and may not be optimal for H
- **Eliminated hypotheses (during diagnosis):**
  - ~~FWHT kernel bug~~ — H(H(x))=x unit test passed
  - ~~fp16 precision~~ — full float32 also collapsed (before fix)
  - ~~Quantization interaction~~ — no-quant mode also collapsed (before fix)
  - ~~Non-power-of-2 padding~~ — 85M_H (all power-of-2) still collapsed (before fix)
  - ~~Output FWHT (sandwich)~~ — input-only FWHT also collapsed (before fix)
  - ~~Weight init scale~~ — standard Kaiming (1.0) also collapsed (before fix)

