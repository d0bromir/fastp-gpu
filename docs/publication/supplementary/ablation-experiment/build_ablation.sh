#!/usr/bin/env bash
# Build the standalone 3-way-ablation tool (experiments/ablation_batch_cpu.cpp).
# Reuses the object files from a normal CPU-only `make` build (run first),
# excluding main.o/unittest.o to avoid a duplicate main().
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

if [[ ! -d obj ]] || [[ -z "$(ls obj/*.o 2>/dev/null)" ]]; then
    echo "Building CPU object files first (make WITH_CUDA=0)..." >&2
    make -j"$(nproc)" WITH_CUDA=0 fastp
fi

OBJS=$(ls obj/*.o | grep -v -E '/(main|unittest|cuda_unittest)\.o$')

g++ -std=c++11 -pthread -O3 -march=native -I./inc \
    "$SCRIPT_DIR/ablation_batch_cpu.cpp" \
    $OBJS \
    -o "$SCRIPT_DIR/ablation_batch_cpu" \
    -lisal -ldeflate -lpthread

echo "Built: $SCRIPT_DIR/ablation_batch_cpu"
