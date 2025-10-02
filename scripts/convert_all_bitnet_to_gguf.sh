#!/usr/bin/env bash
set -euo pipefail

# BitNet to GGUF Conversion Script
# Converts all 4 BitNet models to GGUF format

# Configuration
INPUT_DIR="./"
OUTPUT_DIR="./gguf_models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER_SCRIPT="${SCRIPT_DIR}/convert_bitnet_to_gguf.py"

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

# Function to convert a single model
convert_model() {
    local model_name="$1"
    local model_path="$2"
    
    print_status "Converting $model_name from $model_path"
    
    if check_model_exists "$model_path"; then
        # Run the converter script for this specific model
        python "$CONVERTER_SCRIPT" \
            --input-dir "$model_path" \
            --output-dir "$OUTPUT_DIR/$model_name"
        
        if [[ $? -eq 0 ]]; then
            print_success "Successfully converted $model_name"
        else
            print_error "Failed to convert $model_name"
            return 1
        fi
    else
        print_error "Skipping $model_name - model not found or incomplete"
        return 1
    fi
}

# Main conversion function
main() {
    print_status "Starting BitNet to GGUF conversion for all 4 models"
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
    
    # Define the 4 models to convert
    declare -A MODELS
    MODELS[bitnet-1b]="output-bitnet-1b/final_model"
    MODELS[bitnet-2b]="output-quadratic-2b-hf/final_model"
    MODELS[hbitlinear-1b]="output-bitnet-hbitlinear-1b/final_model"
    MODELS[hbitlinear-2b]="output-quadratic-hbitlinear-2b-hf/final_model"
    
    # Track conversion results
    local successful=0
    local failed=0
    local total=${#MODELS[@]}
    
    print_status "Found $total models to convert"
    echo "----------------------------------------"
    
    # Convert each model
    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"
        
        print_status "Processing model $((successful + failed + 1))/$total: $model_name"
        
        if convert_model "$model_name" "$model_path"; then
            ((successful++))
        else
            ((failed++))
        fi
        
        echo "----------------------------------------"
    done
    
    # Print final summary
    echo
    print_status "Conversion Summary:"
    print_success "Successfully converted: $successful/$total models"
    
    if [[ $failed -gt 0 ]]; then
        print_error "Failed conversions: $failed/$total models"
    fi
    
    if [[ $successful -gt 0 ]]; then
        print_success "Converted models saved to: $OUTPUT_DIR"
        print_status "Each model has its own subdirectory with GGUF files"
        
        # List converted models
        echo
        print_status "Converted models:"
        for model_name in "${!MODELS[@]}"; do
            if [[ -d "$OUTPUT_DIR/$model_name" ]]; then
                print_success "  ✅ $model_name"
            else
                print_error "  ❌ $model_name"
            fi
        done
        
        # Show usage instructions
        echo
        print_status "Usage with llama-cpp-python:"
        echo "  from llama_cpp import Llama"
        echo "  model = Llama(model_path='$OUTPUT_DIR/bitnet-1b/bitnet-1b.gguf')"
        echo "  response = model('Hello world', max_tokens=100)"
    fi
    
    # Exit with appropriate code
    if [[ $failed -eq 0 ]]; then
        print_success "All conversions completed successfully!"
        exit 0
    else
        print_error "Some conversions failed. Check the output above for details."
        exit 1
    fi
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
            echo "BitNet to GGUF Conversion Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR    Input directory (default: ./)"
            echo "  --output-dir DIR   Output directory (default: ./gguf_models)"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "This script converts all 4 BitNet models to GGUF format:"
            echo "  - bitnet-1b"
            echo "  - bitnet-2b" 
            echo "  - hbitlinear-1b"
            echo "  - hbitlinear-2b"
            echo ""
            echo "Example:"
            echo "  $0 --output-dir ./my_gguf_models"
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
