#!/usr/bin/env bash
# verify_public_publish.sh — Post-publish validation for d0bromir/fastp-gpu.
#
# Run this AFTER scripts/publish_public.sh to prove that what is on the
# public remote is buildable and functional. Three checks, all hard-fail:
#
#   1. SOURCE PARITY — clone the public repo into a fresh scratch dir and
#      diff every whitelisted source path (Makefile, src/, scripts/,
#      testdata/, docs/publication/figures, docs/publication/tables,
#      benchmark_results/, top-level docs/*.md, .github/workflows, .gitignore,
#      LICENSE, README.md) against the work repo. Same exclude rules as
#      publish_public.sh (paper-build scripts, *.fq.gz, *.bam, etc.).
#      Any difference is a publish bug — fail.
#
#   2. BUILD — from inside the freshly-cloned public repo:
#        a) make WITH_CUDA=0 PROFILING=0          -> CPU fastp
#        b) make WITH_CUDA=1 PROFILING=0          -> GPU fastp
#      Either build failing aborts.
#
#   3. SMOKE RUN — run each binary on testdata/R1.fq + testdata/R2.fq and
#      verify the output FASTQ is non-empty and the JSON report parses.
#      The two binaries' outputs are also diffed against each other to
#      catch GPU vs CPU divergence.
#
# Usage:
#   scripts/verify_public_publish.sh                # full validation
#   scripts/verify_public_publish.sh --cpu-only     # skip GPU build & run
#   scripts/verify_public_publish.sh --keep         # keep scratch dir on success
#
# Env overrides:
#   PUBLIC_REMOTE_URL  default: https://github.com/d0bromir/fastp-gpu.git
#   PUBLIC_BRANCH      default: master
#   VERIFY_DIR         default: <repo>/.publish-scratch/verify
#   CUDA_ARCH          default: auto-detected by build_all.sh logic
#   JOBS               default: $(nproc)

set -euo pipefail

REMOTE_URL="${PUBLIC_REMOTE_URL:-https://github.com/d0bromir/fastp-gpu.git}"
BRANCH="${PUBLIC_BRANCH:-master}"
CPU_ONLY=0
KEEP_SCRATCH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only) CPU_ONLY=1; shift ;;
        --keep)     KEEP_SCRATCH=1; shift ;;
        -h|--help)
            sed -n '2,33p' "$0"
            exit 0
            ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VERIFY_DIR="${VERIFY_DIR:-$REPO_ROOT/.publish-scratch/verify}"
PUBLIC_CLONE="$VERIFY_DIR/fastp-gpu"
JOBS="${JOBS:-$(nproc)}"

step()  { echo ""; echo ">>> $*"; }
ok()    { echo "    [OK] $*"; }
fail()  { echo "    [FAIL] $*" >&2; exit 1; }

# Mirror the publish-time whitelist and excludes. Keep in sync with
# scripts/publish_public.sh.
PUBLISH_PATHS=(
    Makefile
    README.md
    LICENSE
    .gitignore
    .github/workflows
    src
    scripts
    testdata
    docs/ARCHITECTURE.md
    docs/BUILD_WITH_CUDA.md
    docs/CUDA_ACCELERATION.md
    docs/INDEX.md
    docs/PERFORMANCE_SUMMARY.md
    docs/QUICK_REFERENCE.md
    docs/UPSTREAM_REJECTED.md
    docs/VERSION_MANAGEMENT.md
    docs/publication/figures
    docs/publication/tables
    benchmark_results
)
DIFF_EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.DS_Store'
    --exclude='*.aux'
    --exclude='*.out'
    --exclude='*.fq.gz'
    --exclude='*.fastq.gz'
    --exclude='*.bam'
    --exclude='*.sam'
    --exclude='build_paper.sh'
    --exclude='build_application_notes.sh'
    --exclude='build_st_sim_concept.sh'
    --exclude='build_tus_paper.sh'
)

cleanup() {
    local rc=$?
    if (( rc == 0 )) && (( KEEP_SCRATCH == 0 )); then
        rm -rf "$VERIFY_DIR" || true
    else
        echo "    (scratch retained at $VERIFY_DIR)"
    fi
}
trap cleanup EXIT

# ---------- 1. Fresh clone of the public repo ---------------------------
step "Cloning $REMOTE_URL ($BRANCH) into $PUBLIC_CLONE"
rm -rf "$VERIFY_DIR"
mkdir -p "$VERIFY_DIR"
git clone --depth=1 --branch "$BRANCH" "$REMOTE_URL" "$PUBLIC_CLONE" \
    || fail "could not clone public repo"
PUBLIC_SHA="$(git -C "$PUBLIC_CLONE" rev-parse --short HEAD)"
ok "public HEAD: $PUBLIC_SHA"

# ---------- 2. Source parity check --------------------------------------
step "Source parity: diff whitelisted paths (work repo vs public clone)"
parity_failed=0

# Build a list of "real" files (regular files, no broken symlinks) under the
# work-repo whitelist — these are the files the public repo MUST contain.
# Empty directories (e.g. uninitialized submodule mountpoints under
# src/libs/) and broken symlinks are not publishable artifacts and are
# excluded automatically. Excluded patterns mirror the publish-time list.
# Build a list of "real" files (regular files, no broken symlinks) under the
# work-repo whitelist — these are the files the public repo MUST contain.
# Empty directories (e.g. uninitialized submodule mountpoints under
# src/libs/) and broken symlinks are not publishable artifacts and are
# excluded automatically. Excluded patterns mirror the publish-time list.
EXCLUDE_DIR_NAMES=(__pycache__)
EXCLUDE_FILE_GLOBS=(
    '*.pyc' '.DS_Store' '*.aux' '*.out'
    '*.fq.gz' '*.fastq.gz' '*.bam' '*.sam'
    'build_paper.sh' 'build_application_notes.sh'
    'build_st_sim_concept.sh' 'build_tus_paper.sh'
)

# Returns 0 if the basename of $1 matches any EXCLUDE_FILE_GLOBS pattern.
file_excluded() {
    local base
    base="$(basename "$1")"
    local pat
    for pat in "${EXCLUDE_FILE_GLOBS[@]}"; do
        # shellcheck disable=SC2053
        [[ "$base" == $pat ]] && return 0
    done
    return 1
}

for p in "${PUBLISH_PATHS[@]}"; do
    work_path="$REPO_ROOT/$p"
    pub_path="$PUBLIC_CLONE/$p"
    if [[ ! -e "$work_path" ]]; then
        fail "whitelisted path missing in work repo: $p"
    fi
    if [[ -f "$work_path" ]]; then
        if [[ ! -f "$pub_path" ]]; then
            echo "    [DIFF] missing on public side: $p"
            parity_failed=1
        elif ! diff -q "$work_path" "$pub_path" >/dev/null 2>&1; then
            echo "    [DIFF] file differs: $p"
            parity_failed=1
        fi
        continue
    fi
    # Directory: enumerate regular files, prune excluded directories,
    # filter excluded file globs in bash.
    prune_args=()
    for d in "${EXCLUDE_DIR_NAMES[@]}"; do
        prune_args+=(-name "$d" -o)
    done
    # Drop trailing -o
    if (( ${#prune_args[@]} > 0 )); then
        unset 'prune_args[${#prune_args[@]}-1]'
    fi
    while IFS= read -r f; do
        # find -type f already excludes broken symlinks and directories.
        file_excluded "$f" && continue
        rel="${f#$work_path/}"
        pub_f="$pub_path/$rel"
        if [[ ! -f "$pub_f" ]]; then
            echo "    [DIFF] missing on public side: $p/$rel"
            parity_failed=1
        elif ! diff -q "$f" "$pub_f" >/dev/null 2>&1; then
            echo "    [DIFF] file differs: $p/$rel"
            parity_failed=1
        fi
    done < <(find "$work_path" \( "${prune_args[@]}" \) -prune -o -type f -print 2>/dev/null)
done
(( parity_failed == 1 )) && fail "source parity check FAILED"
ok "all whitelisted files match between work repo and public repo"

# ---------- 3. CPU build from public clone ------------------------------
step "Building CPU fastp from public clone"
make -C "$PUBLIC_CLONE" -j"$JOBS" WITH_CUDA=0 PROFILING=0 clean >/dev/null 2>&1 || true
if ! make -C "$PUBLIC_CLONE" -j"$JOBS" WITH_CUDA=0 PROFILING=0; then
    fail "CPU build from public repo FAILED"
fi
[[ -x "$PUBLIC_CLONE/fastp" ]] || fail "CPU build produced no binary"
cp "$PUBLIC_CLONE/fastp" "$PUBLIC_CLONE/fastp.cpu.verify"
ok "CPU binary: $($PUBLIC_CLONE/fastp.cpu.verify --version 2>&1 | head -1)"

# ---------- 4. GPU build from public clone ------------------------------
GPU_BIN=""
if (( CPU_ONLY == 0 )); then
    step "Building GPU fastp from public clone"
    if ! command -v nvcc &>/dev/null; then
        fail "nvcc not found — cannot validate GPU build (use --cpu-only to skip)"
    fi
    # Reuse build_all.sh's compute-capability detection logic if needed.
    if [[ -z "${CUDA_ARCH:-}" ]] && command -v nvidia-smi &>/dev/null; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d ' .')
    fi
    CUDA_ARCH="${CUDA_ARCH:-80}"
    NVCC_ARCH_FLAGS="-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
    make -C "$PUBLIC_CLONE" -j"$JOBS" WITH_CUDA=1 PROFILING=0 \
        "NVCC_ARCH_FLAGS=$NVCC_ARCH_FLAGS" clean >/dev/null 2>&1 || true
    if ! make -C "$PUBLIC_CLONE" -j"$JOBS" WITH_CUDA=1 PROFILING=0 \
            "NVCC_ARCH_FLAGS=$NVCC_ARCH_FLAGS"; then
        fail "GPU build from public repo FAILED"
    fi
    [[ -x "$PUBLIC_CLONE/fastp" ]] || fail "GPU build produced no binary"
    cp "$PUBLIC_CLONE/fastp" "$PUBLIC_CLONE/fastp.gpu.verify"
    GPU_BIN="$PUBLIC_CLONE/fastp.gpu.verify"
    ok "GPU binary: $($GPU_BIN --version 2>&1 | head -1)"
else
    echo "    [SKIP] --cpu-only: GPU build not validated"
fi

# ---------- 5. Smoke run on testdata ------------------------------------
R1="$PUBLIC_CLONE/testdata/R1.fq"
R2="$PUBLIC_CLONE/testdata/R2.fq"
[[ -s "$R1" && -s "$R2" ]] || fail "testdata/R1.fq or R2.fq missing in public clone"

run_smoke() {
    local label="$1" bin="$2"
    local outdir="$VERIFY_DIR/smoke_${label}"
    mkdir -p "$outdir"
    step "Smoke run: $label  ($bin)"
    pushd "$outdir" >/dev/null
    if ! "$bin" \
            -i "$R1" -I "$R2" \
            -o out_R1.fq.gz -O out_R2.fq.gz \
            -j report.json -h report.html \
            --thread 2 >run.log 2>&1; then
        echo "    --- last 30 lines of run.log ---"
        tail -30 run.log
        popd >/dev/null
        fail "$label: fastp run FAILED"
    fi
    [[ -s out_R1.fq.gz ]] || { popd >/dev/null; fail "$label: out_R1.fq.gz empty"; }
    [[ -s out_R2.fq.gz ]] || { popd >/dev/null; fail "$label: out_R2.fq.gz empty"; }
    [[ -s report.json ]] || { popd >/dev/null; fail "$label: report.json empty"; }
    # Validate JSON parses.
    python3 -c "import json,sys; json.load(open('report.json'))" \
        || { popd >/dev/null; fail "$label: report.json is not valid JSON"; }
    local reads
    reads="$(python3 -c "import json; d=json.load(open('report.json')); print(d['summary']['after_filtering']['total_reads'])")"
    ok "$label: produced output, after_filtering.total_reads=$reads"
    popd >/dev/null
}

run_smoke cpu "$PUBLIC_CLONE/fastp.cpu.verify"
if [[ -n "$GPU_BIN" ]]; then
    run_smoke gpu "$GPU_BIN"

    # CPU vs GPU agreement on the trimmed FASTQ.
    step "CPU vs GPU output agreement on testdata"
    cpu_md5_r1=$(zcat "$VERIFY_DIR/smoke_cpu/out_R1.fq.gz" | md5sum | awk '{print $1}')
    gpu_md5_r1=$(zcat "$VERIFY_DIR/smoke_gpu/out_R1.fq.gz" | md5sum | awk '{print $1}')
    cpu_md5_r2=$(zcat "$VERIFY_DIR/smoke_cpu/out_R2.fq.gz" | md5sum | awk '{print $1}')
    gpu_md5_r2=$(zcat "$VERIFY_DIR/smoke_gpu/out_R2.fq.gz" | md5sum | awk '{print $1}')
    if [[ "$cpu_md5_r1" != "$gpu_md5_r1" || "$cpu_md5_r2" != "$gpu_md5_r2" ]]; then
        echo "    R1: cpu=$cpu_md5_r1  gpu=$gpu_md5_r1"
        echo "    R2: cpu=$cpu_md5_r2  gpu=$gpu_md5_r2"
        fail "CPU and GPU produced different trimmed FASTQ on testdata"
    fi
    ok "CPU and GPU produced byte-identical trimmed FASTQ"
fi

echo
echo "=========================================================="
echo "  Public repo validation PASSED"
echo "    public HEAD : $PUBLIC_SHA"
echo "    remote      : $REMOTE_URL ($BRANCH)"
echo "    cpu-only    : $([[ $CPU_ONLY -eq 1 ]] && echo yes || echo no)"
echo "=========================================================="
