#!/usr/bin/env bash
set -euo pipefail

# BitNet to SafeTensors Conversion Script
# Converts all 4 BitNet models to SafeTensors format

# Configuration
INPUT_DIR="./"
OUTPUT_DIR="./safetensors_models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER_SCRIPT="${SCRIPT_DIR}/convert_bitnet_to_safetensors.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if model directory exists
check_model_exists() {
    local model_path="$1"
    if [[ -d "$model_path" ]]; then
        if [[ -f "$model_path/config.json" && -f "$model_path/model.pt" ]]; then
            return 0
        else
            print_warning "Model directory exists but missing config.json or model.pt: $model_path"
            return 1
        fi
    else
        print_warning "Model directory not found: $model_path"
        return 1
    fi
}

# Function to check if all models exist
check_all_models_exist() {
    local all_exist=true
    
    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"
        if ! check_model_exists "$model_path"; then
            all_exist=false
        fi
    done
    
    if [[ "$all_exist" == "false" ]]; then
        print_error "Some models are missing or incomplete. Please check the model directories."
        return 1
    fi
    
    return 0
}

# Main conversion function
main() {
    print_status "Starting BitNet to SafeTensors conversion for all 4 models"
    print_status "Input directory: $INPUT_DIR"
    print_status "Output directory: $OUTPUT_DIR"
    print_status "Converter script: $CONVERTER_SCRIPT"
    
    # Check if converter script exists
    if [[ ! -f "$CONVERTER_SCRIPT" ]]; then
        print_error "Converter script not found: $CONVERTER_SCRIPT"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if all models exist first
    print_status "Checking if all models exist..."
    if ! check_all_models_exist; then
        print_error "Cannot proceed with conversion due to missing models"
        exit 1
    fi
    
    print_status "All models found! Starting conversion..."
    echo "----------------------------------------"
    
    # Run the converter script once for all models
    python "$CONVERTER_SCRIPT" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR"
    
    local conversion_exit_code=$?
    
    if [[ $conversion_exit_code -eq 0 ]]; then
        print_success "Conversion completed successfully!"
    else
        print_error "Conversion failed with exit code: $conversion_exit_code"
        exit 1
    fi
    
    # Print final summary
    echo
    print_status "Conversion completed!"
    print_success "Converted models saved to: $OUTPUT_DIR"
    print_status "Each model has its own subdirectory with SafeTensors files"
    
    # Show usage instructions
    echo
    print_status "Usage with HuggingFace Transformers:"
    echo "  from transformers import AutoTokenizer, AutoModelForCausalLM"
    echo "  from safetensors.torch import load_file"
    echo "  state_dict = load_file('$OUTPUT_DIR/bitnet-1b/bitnet-1b.safetensors')"
    echo "  model = AutoModelForCausalLM.from_pretrained('.', state_dict=state_dict)"
    
    print_success "All conversions completed successfully!"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "BitNet to SafeTensors Conversion Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR    Input directory (default: ./)"
            echo "  --output-dir DIR   Output directory (default: ./safetensors_models)"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "This script converts all 4 BitNet models to SafeTensors format:"
            echo "  - bitnet-1b"
            echo "  - bitnet-2b" 
            echo "  - hbitlinear-1b"
            echo "  - hbitlinear-2b"
            echo ""
            echo "Example:"
            echo "  $0 --output-dir ./my_safetensors_models"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
