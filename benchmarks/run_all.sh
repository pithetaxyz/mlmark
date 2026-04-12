#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 -o <output_dir> [-v <venv_path>]"
    echo "  -o  Directory where results will be saved (required)"
    echo "  -v  Python venv to activate (default: \$HOME/rocm)"
    exit 1
}

RESULTS_DIR=""
VENV="$HOME/rocm"

while getopts "o:v:" opt; do
    case $opt in
        o) RESULTS_DIR="$OPTARG" ;;
        v) VENV="$OPTARG" ;;
        *) usage ;;
    esac
done

[[ -z "$RESULTS_DIR" ]] && usage

source "$VENV/bin/activate"

# gfx1035 (Radeon 680M / RDNA 2) is not in the PyTorch ROCm wheel's default target list.
# Override to gfx1030 kernels which are binary-compatible on this architecture.
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd "$(dirname "$0")"
exec python3 run_all.py -o "$RESULTS_DIR"
