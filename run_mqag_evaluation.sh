#!/bin/bash

# MQAG Models Comprehensive Evaluation Runner
# ==========================================
# This script provides a comprehensive interface for running MQAG model evaluations
# with various configurations and options.

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/mqag_evaluation.py"
DEFAULT_OUTPUT="comprehensive_mqag_results.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() {
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

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Help function
show_help() {
    cat << EOF
MQAG Models Comprehensive Evaluation Runner

USAGE:
    $0 [OPTIONS] [COMMAND]

COMMANDS:
    test        Run quick test evaluation with sample data
    batch       Run batch evaluation with multiple configurations
    check       Check dependencies and system requirements
    help        Show this help message

OPTIONS:
    --viquad-path PATH      Path to UIT-ViQuAD dataset JSON file
    --vsmrc-path PATH       Path to VSMRC dataset JSON file
    --batch-size N          Batch size for evaluation (default: 4)
    --max-samples N         Maximum samples to evaluate (default: no limit)
    --output FILE           Output JSON file (default: $DEFAULT_OUTPUT)
    --device DEVICE         Device to use (cuda/cpu, default: auto-detect)
    --use-huggingface       Try to load datasets from HuggingFace
    --verbose               Enable verbose logging
    --quiet                 Suppress non-essential output

EXAMPLES:
    # Quick test
    $0 test

    # Batch evaluation
    $0 batch

    # Check dependencies
    $0 check

    # Full evaluation with local files
    $0 --viquad-path data/viquad.json --vsmrc-path data/vsmrc.json

    # Evaluation with HuggingFace datasets
    $0 --use-huggingface --max-samples 100 --device cuda

    # Small-scale evaluation
    $0 --max-samples 50 --batch-size 2 --device cpu

SYSTEM REQUIREMENTS:
    - Python 3.7+
    - PyTorch
    - Transformers
    - NumPy
    - Optional: datasets (for HuggingFace loading)

EOF
}

# Check if Python script exists
check_python_script() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        print_info "Please make sure you're running this script from the correct directory"
        exit 1
    fi
}

# Check Python and dependencies
check_dependencies() {
    print_info "Checking system dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed"
        return 1
    fi
    
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python3 version: $python_version"
    
    # Check Python dependencies
    python3 "$PYTHON_SCRIPT" --check-deps
    local dep_status=$?
    
    if [[ $dep_status -eq 0 ]]; then
        print_success "All dependencies are available"
        return 0
    else
        print_error "Some dependencies are missing"
        print_info "Please install required packages:"
        print_info "  pip install torch transformers numpy"
        print_info "  pip install datasets  # (optional, for HuggingFace support)"
        return 1
    fi
}

# Run test evaluation
run_test() {
    print_header "RUNNING TEST EVALUATION"
    print_info "This will run a quick test with sample Vietnamese data"
    
    local cmd="python3 '$PYTHON_SCRIPT' --test"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose"
    fi
    
    print_info "Executing: $cmd"
    
    if eval "$cmd"; then
        print_success "Test evaluation completed successfully!"
        return 0
    else
        print_error "Test evaluation failed!"
        return 1
    fi
}

# Run batch evaluation
run_batch() {
    print_header "RUNNING BATCH EVALUATION"
    print_info "This will run multiple evaluation configurations"
    
    local cmd="python3 '$PYTHON_SCRIPT' --batch"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose"
    fi
    
    print_info "Executing: $cmd"
    
    if eval "$cmd"; then
        print_success "Batch evaluation completed successfully!"
        return 0
    else
        print_error "Batch evaluation failed!"
        return 1
    fi
}

# Run full evaluation
run_evaluation() {
    print_header "RUNNING FULL EVALUATION"
    
    local cmd="python3 '$PYTHON_SCRIPT'"
    
    # Add arguments
    [[ -n "$VIQUAD_PATH" ]] && cmd="$cmd --viquad-path '$VIQUAD_PATH'"
    [[ -n "$VSMRC_PATH" ]] && cmd="$cmd --vsmrc-path '$VSMRC_PATH'"
    [[ -n "$BATCH_SIZE" ]] && cmd="$cmd --batch-size $BATCH_SIZE"
    [[ -n "$MAX_SAMPLES" ]] && cmd="$cmd --max-samples $MAX_SAMPLES"
    [[ -n "$OUTPUT" ]] && cmd="$cmd --output '$OUTPUT'"
    [[ -n "$DEVICE" ]] && cmd="$cmd --device $DEVICE"
    [[ "$USE_HUGGINGFACE" == "true" ]] && cmd="$cmd --use-huggingface"
    
    print_info "Configuration:"
    [[ -n "$VIQUAD_PATH" ]] && print_info "  ViQuAD path: $VIQUAD_PATH"
    [[ -n "$VSMRC_PATH" ]] && print_info "  VSMRC path: $VSMRC_PATH"
    [[ -n "$BATCH_SIZE" ]] && print_info "  Batch size: $BATCH_SIZE"
    [[ -n "$MAX_SAMPLES" ]] && print_info "  Max samples: $MAX_SAMPLES"
    [[ -n "$OUTPUT" ]] && print_info "  Output file: $OUTPUT"
    [[ -n "$DEVICE" ]] && print_info "  Device: $DEVICE"
    [[ "$USE_HUGGINGFACE" == "true" ]] && print_info "  HuggingFace: enabled"
    
    print_info "Executing: $cmd"
    
    if eval "$cmd"; then
        print_success "Evaluation completed successfully!"
        [[ -n "$OUTPUT" ]] && print_info "Results saved to: $OUTPUT"
        return 0
    else
        print_error "Evaluation failed!"
        return 1
    fi
}

# Monitor system resources
monitor_resources() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
            print_info "  $line"
        done
    fi
    
    if command -v free &> /dev/null; then
        print_info "Memory Status:"
        free -h | head -2 | while read line; do
            print_info "  $line"
        done
    fi
}

# Main execution
main() {
    # Initialize variables
    COMMAND=""
    VIQUAD_PATH=""
    VSMRC_PATH=""
    BATCH_SIZE=""
    MAX_SAMPLES=""
    OUTPUT=""
    DEVICE=""
    USE_HUGGINGFACE="false"
    VERBOSE="false"
    QUIET="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            test|batch|check|help)
                COMMAND="$1"
                shift
                ;;
            --viquad-path)
                VIQUAD_PATH="$2"
                shift 2
                ;;
            --vsmrc-path)
                VSMRC_PATH="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --max-samples)
                MAX_SAMPLES="$2"
                shift 2
                ;;
            --output)
                OUTPUT="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --use-huggingface)
                USE_HUGGINGFACE="true"
                shift
                ;;
            --verbose)
                VERBOSE="true"
                shift
                ;;
            --quiet)
                QUIET="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_info "Use '$0 help' for usage information"
                exit 1
                ;;
        esac
    done
    
    # Show header
    if [[ "$QUIET" != "true" ]]; then
        print_header "MQAG Models Comprehensive Evaluation"
        print_info "Vietnamese Question-Answer Generation Models Evaluation"
        print_info "Script: $PYTHON_SCRIPT"
        print_info "Working Directory: $(pwd)"
        echo
    fi
    
    # Check Python script
    check_python_script
    
    # Handle commands
    case "$COMMAND" in
        help)
            show_help
            exit 0
            ;;
        check)
            check_dependencies
            exit $?
            ;;
        test)
            check_dependencies || exit 1
            if [[ "$VERBOSE" == "true" ]]; then
                monitor_resources
            fi
            run_test
            exit $?
            ;;
        batch)
            check_dependencies || exit 1
            if [[ "$VERBOSE" == "true" ]]; then
                monitor_resources
            fi
            run_batch
            exit $?
            ;;
        "")
            # No command, run full evaluation
            check_dependencies || exit 1
            if [[ "$VERBOSE" == "true" ]]; then
                monitor_resources
            fi
            run_evaluation
            exit $?
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            print_info "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'print_error "Interrupted by user"; exit 130' INT TERM

# Run main function
main "$@"
