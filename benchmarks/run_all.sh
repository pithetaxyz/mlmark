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

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT="$RESULTS_DIR/run_${TIMESTAMP}"

mkdir -p "$OUT"
source "$VENV/bin/activate"

# gfx1035 (Radeon 680M / RDNA 2) is not in the PyTorch ROCm wheel's default target list.
# Override to gfx1030 kernels which are binary-compatible on this architecture.
export HSA_OVERRIDE_GFX_VERSION=10.3.0

SCRIPTS_DIR="$(dirname "$0")"

run_bench() {
    local script="$1"
    local name="$(basename "$script" .py)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 "$script" 2>&1 | tee "$OUT/${name}.log" | grep -v "^{" | grep -v "^}" | grep -v "^\[" | grep -v "^  \"" || true
    # Extract JSON block and save separately
    python3 "$script" 2>/dev/null | python3 -c "
import sys, json
data = sys.stdin.read()
start = data.find('{')
if start != -1:
    try:
        obj = json.loads(data[start:])
        print(json.dumps(obj, indent=2))
    except:
        pass
" > "$OUT/${name}.json" 2>/dev/null || true
}

echo "Benchmark run: $TIMESTAMP"
echo "Results dir:  $OUT"

run_bench "$SCRIPTS_DIR/bench_matmul.py"
run_bench "$SCRIPTS_DIR/bench_cnn.py"
run_bench "$SCRIPTS_DIR/bench_transformer.py"
run_bench "$SCRIPTS_DIR/bench_memory.py"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Done. Results saved to: $OUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
