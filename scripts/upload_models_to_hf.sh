#!/usr/bin/env bash
set -euo pipefail

# Upload all final models to Hugging Face Hub using scripts/upload_to_hf.py
#
# Usage:
#   export HUGGINGFACE_TOKEN=...    # Required
#   bash scripts/upload_models_to_hf.sh [--private] [--revision main]
#
# It looks for these local directories (default training outputs):
#   ./output-bitnet-1b/final_model
#   ./output-quadratic-2b-hf/final_model
#   ./output-bitnet-hbitlinear-1b/final_model
#   ./output-quadratic-hbitlinear-2b-hf/final_model
#
# Repos created/used:
#   Ram07/bitnet-1b
#   Ram07/bitnet-2b  
#   Ram07/hbitnet-1b
#   Ram07/hbitnet-2b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${HUGGINGFACE_TOKEN:-}" ]]; then
  echo "[ERROR] HUGGINGFACE_TOKEN is not set" >&2
  exit 1
fi

# Set default HF_USER to Ram07
HF_USER="${HF_USER:-Ram07}"

PRIVATE_FLAG=""
REVISION="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --private)
      PRIVATE_FLAG="--private"
      shift
      ;;
    --revision)
      REVISION="${2:-main}"
      shift 2
      ;;
    *)
      echo "[WARN] Unknown arg: $1" >&2
      shift
      ;;
  esac
done

declare -A MODEL_DIRS
# Updated to match your new directory structure
MODEL_DIRS[bitnet-1b]="${REPO_ROOT}/safetensors/bitnet-1b"
MODEL_DIRS[bitnet-2b]="${REPO_ROOT}/safetensors/bitnet-2b"
MODEL_DIRS[hbitlinear-1b]="${REPO_ROOT}/safetensors/hbitnet-1b"
MODEL_DIRS[hbitlinear-2b]="${REPO_ROOT}/safetensors/hbitnet-2b"

upload_file() {
  local file_path="$1"
  local repo_id="$2"
  local dest_path="$3"
  if [[ ! -f "$file_path" ]]; then
    echo "[WARN] Missing file: $file_path (skipping)" >&2
    return 0
  fi
  echo "[INFO] Uploading $file_path -> $repo_id:$dest_path (rev: $REVISION)"
  python "${REPO_ROOT}/scripts/upload_to_hf.py" \
    --file "$file_path" \
    --repo-id "$repo_id" \
    --path-in-repo "$dest_path" \
    --revision "$REVISION" \
    $PRIVATE_FLAG
}

for name in "${!MODEL_DIRS[@]}"; do
  MODEL_PATH="${MODEL_DIRS[$name]}"
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[WARN] Skipping $name: folder not found -> $MODEL_PATH" >&2
    continue
  fi

  REPO_ID="${HF_USER}/${name}"
  echo "\n[INFO] Processing $name"
  echo "[INFO] Local folder: $MODEL_PATH"
  echo "[INFO] Target repo: $REPO_ID (rev: $REVISION) ${PRIVATE_FLAG:+[private]}"

  # Upload SafeTensors files at repo root
  upload_file "${MODEL_PATH}/config.json" "$REPO_ID" "config.json"
  upload_file "${MODEL_PATH}/README.md" "$REPO_ID" "README.md"
  
  # Upload conversion info (try both possible names)
  if [[ -f "${MODEL_PATH}/conversion_info.json" ]]; then
    upload_file "${MODEL_PATH}/conversion_info.json" "$REPO_ID" "conversion_info.json"
  else
    echo "[WARN] No conversion info file found in ${MODEL_PATH}" >&2
  fi
  
  # Upload the .safetensors file (find the exact filename)
  SAFETENSORS_FILE=$(find "${MODEL_PATH}" -name "*.safetensors" -type f | head -1)
  if [[ -n "$SAFETENSORS_FILE" ]]; then
    SAFETENSORS_FILENAME=$(basename "$SAFETENSORS_FILE")
    upload_file "$SAFETENSORS_FILE" "$REPO_ID" "$SAFETENSORS_FILENAME"
  else
    echo "[WARN] No .safetensors file found in ${MODEL_PATH}" >&2
  fi
done

echo "\n[DONE] Uploads completed."


