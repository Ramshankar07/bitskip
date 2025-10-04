#!/usr/bin/env bash
set -euo pipefail

# Check SafeTensors directory structure before upload
# Usage: bash scripts/check_safetensors_structure.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "🔍 Checking SafeTensors directory structure..."
echo "Repository root: $REPO_ROOT"
echo ""

declare -A MODEL_DIRS
MODEL_DIRS[bitnet-1b]="${REPO_ROOT}/safetensors/bitnet-1b"
MODEL_DIRS[bitnet-2b]="${REPO_ROOT}/safetensors/bitnet-2b"
MODEL_DIRS[hbitlinear-1b]="${REPO_ROOT}/safetensors/hbitnet-1b"
MODEL_DIRS[hbitlinear-2b]="${REPO_ROOT}/safetensors/hbitnet-2b"

ALL_GOOD=true

for name in "${!MODEL_DIRS[@]}"; do
  MODEL_DIR="${MODEL_DIRS[$name]}"
  
  echo "📁 Checking $name:"
  echo "   Path: $MODEL_DIR"
  
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "   ❌ Directory not found!"
    ALL_GOOD=false
    continue
  fi
  
  # Check required files
  if [[ -f "$MODEL_DIR/config.json" ]]; then
    echo "   ✅ config.json found"
  else
    echo "   ❌ config.json missing"
    ALL_GOOD=false
  fi
  
  SAFETENSORS_FILE=$(find "$MODEL_DIR" -name "*.safetensors" -type f | head -1)
  if [[ -n "$SAFETENSORS_FILE" ]]; then
    file_size=$(du -h "$SAFETENSORS_FILE" | cut -f1)
    echo "   ✅ SafeTensors file found: $(basename "$SAFETENSORS_FILE") ($file_size)"
  else
    echo "   ❌ No .safetensors file found"
    ALL_GOOD=false
  fi
  
  if [[ -f "$MODEL_DIR/README.md" ]]; then
    echo "   ✅ README.md found"
  else
    echo "   ⚠️  README.md missing (will be created)"
  fi
  
  if [[ -f "$MODEL_DIR/conversion_info.json" ]]; then
    echo "   ✅ conversion_info.json found"
  else
    echo "   ⚠️  conversion_info.json missing (optional)"
  fi
  
  echo ""
done

if [[ "$ALL_GOOD" == "true" ]]; then
  echo "🎉 All models are ready for upload!"
  echo ""
  echo "To upload to Hugging Face, run:"
  echo "export HUGGINGFACE_TOKEN=..."
  echo "bash scripts/upload_models_to_hf.sh"
  echo ""
  echo "Or upload individually:"
  echo "bash scripts/upload_models_to_hf.sh --private  # for private repos"
else
  echo "⚠️  Some models are missing files. Please check the directories."
fi
