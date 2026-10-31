#!/usr/bin/env bash
# =============================================================================
# build_all.sh — Build all fastp variants for benchmarking
#
# Builds three binaries expected by run_benchmark.sh / run_benchmark_wsl.sh:
#
#   ../fastp_opengene/fastp   — upstream OpenGene reference (CPU, no profiling)
#   ./fastp-cpu               — d0bromir CPU build with profiling
#   ./fastp                   — d0bromir GPU build with profiling
#
# GPU arch: auto-detected from installed GPU via nvidia-smi.
#           Override with: CUDA_ARCH=XX ./build_all.sh
#
# Usage:
#   ./scripts/build_all.sh              # build all three
#   ./scripts/build_all.sh opengene     # only opengene reference
#   ./scripts/build_all.sh cpu          # only d0bromir CPU
#   ./scripts/build_all.sh gpu          # only d0bromir GPU
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENGENE_DIR="$(cd "$PROJ_DIR/.." && pwd)/fastp_opengene"

# ---------------------------------------------------------------------------
# Auto-detect GPU compute capability
# Priority: CUDA_ARCH env var > nvidia-smi query > fallback 80
# ---------------------------------------------------------------------------
detect_cuda_arch() {
    if command -v nvidia-smi &>/dev/null; then
        # Query compute capability of the first GPU (e.g. "8.0" -> "80")
        local cap
        cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
        if [[ -n "$cap" && "$cap" =~ ^[0-9]+$ ]]; then
            echo "$cap"
            return
        fi
    fi
    # Fallback if nvidia-smi is unavailable or returns nothing
    echo "80"
}

if [[ -z "${CUDA_ARCH:-}" ]]; then
    CUDA_ARCH="$(detect_cuda_arch)"
    echo "    [auto] Detected GPU compute capability: sm_${CUDA_ARCH}"
fi

NVCC_ARCH_FLAGS="-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"

# Parallel jobs
JOBS="${JOBS:-$(nproc)}"

# Set SKIP_TESTS=1 to skip post-build unit tests (NOT recommended).
SKIP_TESTS="${SKIP_TESTS:-0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step()  { echo ""; echo ">>> $*"; }
ok()    { echo "    [OK] $*"; }
fail()  { echo "    [FAIL] $*" >&2; exit 1; }

# Run a binary's built-in unit tests (`<binary> test`).  Hard-fail on error.
# Set SKIP_TESTS=1 in the env to bypass (not recommended).
post_build_unit_test() {
    local label="$1" bin="$2"
    if [[ "$SKIP_TESTS" == "1" ]]; then
        echo "    [WARN] SKIP_TESTS=1 — skipping unit tests for $label"
        return 0
    fi
    [[ -x "$bin" ]] || fail "$label: binary missing for unit tests ($bin)"
    echo "    [test] $label  $bin test"
    local out rc=0
    out=$(cd "$PROJ_DIR" && "$bin" test 2>&1) || rc=$?
    if (( rc != 0 )) || echo "$out" | grep -qiE '^FAILED|: failed'; then
        echo "$out"
        fail "$label: built-in unit tests FAILED — refusing to ship this binary"
    fi
    ok "$label: built-in unit tests passed"
}

TARGET="${1:-all}"

# ---------------------------------------------------------------------------
# 1. OpenGene reference
# ---------------------------------------------------------------------------
build_opengene() {
    step "Building OpenGene fastp reference → $OPENGENE_DIR"

    if [[ ! -d "$OPENGENE_DIR/.git" ]]; then
        step "Cloning OpenGene/fastp..."
        git clone --depth=1 https://github.com/OpenGene/fastp.git "$OPENGENE_DIR"
    else
        step "Updating existing clone..."
        git -C "$OPENGENE_DIR" pull --ff-only || true
    fi

    step "Building OpenGene fastp (CPU, no profiling)..."
    make -C "$OPENGENE_DIR" -j"$JOBS" clean 2>/dev/null || true
    make -C "$OPENGENE_DIR" -j"$JOBS"

    [[ -x "$OPENGENE_DIR/fastp" ]] || fail "OpenGene build produced no binary"
    ok "opengene fastp: $("$OPENGENE_DIR/fastp" --version 2>&1 | head -1)"
    post_build_unit_test "opengene" "$OPENGENE_DIR/fastp"
}

# ---------------------------------------------------------------------------
# 2. d0bromir CPU build (profiling enabled, no CUDA)
# ---------------------------------------------------------------------------
build_cpu() {
    step "Building d0bromir fastp CPU + profiling → $PROJ_DIR/fastp-cpu"

    make -C "$PROJ_DIR" -j"$JOBS" \
        WITH_CUDA=0 \
        PROFILING=1 \
        clean 2>/dev/null || true

    make -C "$PROJ_DIR" -j"$JOBS" \
        WITH_CUDA=0 \
        PROFILING=1

    [[ -x "$PROJ_DIR/fastp" ]] || fail "CPU build produced no binary"
    cp "$PROJ_DIR/fastp" "$PROJ_DIR/fastp-cpu"
    ok "fastp-cpu built: $(ls -lh "$PROJ_DIR/fastp-cpu" | awk '{print $5}')"
    post_build_unit_test "d0bromir_cpu" "$PROJ_DIR/fastp-cpu"
}

# ---------------------------------------------------------------------------
# 3. d0bromir GPU build (profiling + CUDA)
# ---------------------------------------------------------------------------
build_gpu() {
    step "Building d0bromir fastp GPU + profiling → $PROJ_DIR/fastp"

    if ! command -v nvcc &>/dev/null; then
        fail "nvcc not found — cannot build GPU version. Install CUDA Toolkit."
    fi

    make -C "$PROJ_DIR" -j"$JOBS" \
        WITH_CUDA=1 \
        PROFILING=1 \
        "NVCC_ARCH_FLAGS=$NVCC_ARCH_FLAGS" \
        clean 2>/dev/null || true

    make -C "$PROJ_DIR" -j"$JOBS" \
        WITH_CUDA=1 \
        PROFILING=1 \
        "NVCC_ARCH_FLAGS=$NVCC_ARCH_FLAGS"

    [[ -x "$PROJ_DIR/fastp" ]] || fail "GPU build produced no binary"
    ok "fastp (GPU) built: $(ls -lh "$PROJ_DIR/fastp" | awk '{print $5}')"
    # Convenience symlink so all three benchmark binaries appear with
    # consistent names (fastp-cpu / fastp-gpu) in PATH and ps listings.
    # The canonical drop-in-replacement name remains ./fastp.
    ln -sf fastp "$PROJ_DIR/fastp-gpu"
    post_build_unit_test "d0bromir_gpu" "$PROJ_DIR/fastp"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
echo "============================================================"
echo " fastp build_all.sh"
echo " Project  : $PROJ_DIR"
echo " OpenGene : $OPENGENE_DIR"
echo " CUDA arch: sm_$CUDA_ARCH"
echo " Jobs     : $JOBS"
echo " Target   : $TARGET"
echo "============================================================"

case "$TARGET" in
    all)
        build_opengene
        build_cpu
        build_gpu
        ;;
    opengene)
        build_opengene
        ;;
    cpu)
        build_cpu
        ;;
    gpu)
        build_gpu
        ;;
    *)
        echo "Unknown target '$TARGET'. Use: all | opengene | cpu | gpu"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# CPU↔GPU regression on testdata/ (only meaningful when both d0bromir builds
# are present in this run).  Hard-fail on mismatch.
# ---------------------------------------------------------------------------
if [[ "$SKIP_TESTS" != "1" && "$TARGET" == "all" ]]; then
    step "Running CPU↔GPU regression test on testdata/"
    "$SCRIPT_DIR/run_tests.sh" --regression-only --skip-opengene \
        || fail "CPU↔GPU regression FAILED — refusing to ship these binaries"
fi

echo ""
echo "============================================================"
echo " Build complete. Binaries:"
[[ -x "$OPENGENE_DIR/fastp"         ]] && echo "  opengene     : $OPENGENE_DIR/fastp"
[[ -x "$PROJ_DIR/fastp-cpu" ]] && echo "  d0bromir cpu : $PROJ_DIR/fastp-cpu"
[[ -x "$PROJ_DIR/fastp"             ]] && echo "  d0bromir gpu : $PROJ_DIR/fastp"
echo "============================================================"
