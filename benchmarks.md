# BitSkip V2 Ablation Benchmarks

**Dataset:** WikiText-2, 500 training steps
**Hardware:** Modal B200 GPU
**Branch:** `v2-Implementation`
**Date:** 2026-03-04

## Summary


| Run | Model | Fix Applied | noH Best Val PPL | H Best Val PPL | H Status |
|-----|-------|------------|-------------------|----------------|----------|
| Run 1 (pre-fix) | 125M (768h) | None (no causal mask) | ~16 (invalid) | ~1,794 | Collapsed |
| Run 2 | 125M (768h) | Causal mask + H-BitLinear fixes | **251.11** | ~1,494 | Collapsed |
| Run 3 | 85M_H (512h) | activation_bits fix + power-of-2 dims | **252.93** | ~1,491 | Collapsed (unchanged) |

**Run 3 changes:**
1. Fixed `activation_bits` not being passed to HBitLinear in `gqa_attention2.py` (all 4 projections defaulted to 4-bit instead of config's 8-bit).
2. Switched from 125M (768 hidden, 12 heads, 3072 FFN) to **85M_H** (512 hidden, 8 heads, 2048 FFN) — all power-of-2 dimensions, eliminating FWHT padding entirely.

Neither fix resolved the Hadamard collapse.

---

## Run 3: activation_bits Fix + 85M_H Model (power-of-2 dims)

**Model:** 85M_H (512 hidden, 12 layers, 8 heads, 4 KV heads, 2048 FFN)

### Fixes Applied

- **`gqa_attention2.py`:** All 4 HBitLinear projections (q/k/v/o_proj) now pass `activation_bits=activation_bits`. Previously defaulted to 4-bit regardless of config.
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

### H Model (Collapsed — 3 runs, 5 fixes, no improvement)

- ALL H-BitLinear experiments across Runs 1-3 collapsed to PPL ~1,490-1,800
- Fixes attempted with no effect:
  1. LayerNorm moved to unpadded dimensions
  2. Hadamard applied before quantization (was reversed)
  3. Weight init 0.1x scaling
  4. `activation_bits` passed to GQA2 attention projections (Run 3)
  5. Power-of-2 model dims via 85M_H (512h, 2048 FFN) — zero FWHT padding (Run 3)
- route-OFF H variants show marginal Final Val PPL improvement (~1,491-1,566 vs ~1,790) suggesting slight learning
- **Eliminated hypotheses:**
  - ~~Non-power-of-2 padding corrupts the Hadamard rotation~~ — Run 3 used 85M_H (all power-of-2), still collapsed
  - ~~activation_bits defaulting to 4-bit in attention~~ — fixed in Run 3, no effect
- **Remaining hypotheses:**
  - The FWHT kernel or inverse-FWHT implementation itself is incorrect
  - Ternary weight quantization + Hadamard rotation combined is fundamentally unstable
  - The HBitLinear forward pass order (norm → pad → FWHT → quant → linear → iFFWHT → unpad) has a logic error
  - Learning rate / optimizer settings may need Hadamard-specific tuning

