#!/usr/bin/env bash
set -euo pipefail

# Runs evals/inference_benchmark_1b_improved.py against four model outputs and saves results
# Models expected (default output dirs from training scripts):
#  - output-bitnet-1b/final_model                         (train_quadratic_1b_improved.py)
#  - output-quadratic-2b-hf/final_model                   (train_quadratic_2b.py)
#  - output-bitnet-hbitlinear-1b/final_model              (train_quadratic_hbitlinear_1b.py)
#  - output-quadratic-hbitlinear-2b-hf/final_model        (train_quadratic_hbitlinear_2b.py)

DEVICE=${DEVICE:-cuda}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
NUM_RUNS=${NUM_RUNS:-3}
OUT_DIR=${OUT_DIR:-./benchmark_results}

mkdir -p "${OUT_DIR}"

declare -A MODELS
MODELS[bitnet_1b]="./output-bitnet-1b/final_model"
MODELS[bitnet_2b]="./output-quadratic-2b-hf/final_model"
MODELS[hbitlinear_1b]="./output-bitnet-hbitlinear-1b/final_model"
MODELS[hbitlinear_2b]="./output-quadratic-hbitlinear-2b-hf/final_model"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }

for name in "${!MODELS[@]}"; do
  model_path="${MODELS[$name]}"
  if [[ ! -d "${model_path}" && ! -f "${model_path}" ]]; then
    echo "[WARN] Skipping ${name}: path not found -> ${model_path}" >&2
    continue
  fi

  json_out="${OUT_DIR}/inference_${name}_$(ts).json"
  log_out="${OUT_DIR}/inference_${name}_$(ts).log"

  echo "[INFO] Running benchmark for ${name} -> ${model_path}"
  echo "[INFO] Output: ${json_out} (JSON), ${log_out} (stdout log)"

  python -u evals/inference_benchmark_1b_improved.py \
    --model_path "${model_path}" \
    --device "${DEVICE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_runs "${NUM_RUNS}" \
    --output_file "${json_out}" \
    | tee "${log_out}"
done

echo "[DONE] Benchmarks completed. Results saved under ${OUT_DIR}"


