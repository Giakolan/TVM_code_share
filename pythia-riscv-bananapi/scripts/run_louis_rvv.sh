#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"

export LD_LIBRARY_PATH="$PROJECT_DIR/runtime:${LD_LIBRARY_PATH:-}"
export KIWIPEDIA_MATMUL_SO="$PROJECT_DIR/kernels/libmatmul_louis_rvv.so"

echo "========================================"
echo "Pythia-70M Banana Pi - Louis RVV"
echo "========================================"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "KIWIPEDIA_MATMUL_SO=$KIWIPEDIA_MATMUL_SO"
echo

./bin/pythia_runner_riscv
