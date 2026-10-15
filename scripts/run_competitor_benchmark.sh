#!/usr/bin/env bash
# =============================================================================
# run_competitor_benchmark.sh — Compare fastp / fastp-gpu against real,
# obtainable CPU FASTQ-QC competitors on the same public benchmark datasets.
#
# Context: docs/publication/supplementary/gpu-fastq-qc-competitor-landscape.md
# verified that no other GPU-accelerated FASTQ QC/preprocessing tool is both
# currently obtainable and buildable (G-FastQC/G-CNV source is gone; the only
# real GPU tool found, CARE, does error correction, a different task).
# RabbitQCPlus, despite being CPU, was found to be x86-only (hard-coded SSE/AVX
# intrinsics in 8 core files, no ARM/NEON path) and cannot run on this host.
# This script therefore benchmarks the tools that are real, obtainable, and
# actually runnable here: FastQC, Falco, and our own fastp / fastp-gpu.
#
# Datasets: the private TST-15 clinical panel (Panel_SE_148M) used by
# scripts/run_benchmark.sh is not available for third-party comparison.
# The public equivalents already defined in that script's canonical dataset
# table are used instead (fetch via `scripts/download_bench_samples.sh panel`):
#   Panel_PDAC_PE_1M   DRR262998  (PDAC 4-gene panel, ~1M reads, MiSeq PE 156bp)
#   Panel_PDAC_PE_1.5M DRR263018  (PDAC 4-gene panel, ~1.5M reads, MiSeq PE 156bp)
#
# This is an exploratory competitor comparison, NOT the validated fastp vs
# opengene regression (scripts/run_benchmark.sh). It intentionally does not
# touch that script's validated CSVs (Rule 0a) — results are written to a
# separate, clearly labelled file.
#
# Usage:
#   ./scripts/run_competitor_benchmark.sh [-w THREADS] [-o OUTPUT_DIR]
#
# Requires (built once, paths overridable via env):
#   FASTP_CPU_BIN   (our fastp-cpu binary)
#   FASTP_GPU_BIN   (our fastp-gpu binary)
#   FASTQC_BIN      (FastQC wrapper script)
#   FALCO_BIN       (falco binary)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

THREADS="${THREADS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/benchmark_results/competitor_comparison}"

while getopts "w:o:h" opt; do
    case "$opt" in
        w) THREADS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option" >&2; exit 1 ;;
    esac
done

FASTQ_DIR="${FASTQ_DIR:-$HOME/FASTQ}"
PANEL_PUBLIC_DIR="${PANEL_PUBLIC_DIR:-$FASTQ_DIR/Panel_public}"

FASTP_CPU_BIN="${FASTP_CPU_BIN:-$HOME/tools/bin/fastp_d0bromir_cpu}"
FASTP_GPU_BIN="${FASTP_GPU_BIN:-$HOME/tools/bin/fastp_d0bromir_gpu}"
FASTQC_BIN="${FASTQC_BIN:-$HOME/tools/src/competitors/FastQC/fastqc}"
FALCO_BIN="${FALCO_BIN:-$HOME/tools/src/competitors/falco/build/falco}"

for f in "$FASTP_CPU_BIN" "$FASTP_GPU_BIN" "$FASTQC_BIN" "$FALCO_BIN"; do
    [[ -x "$f" ]] || { echo "ERROR: required binary not found or not executable: $f" >&2; exit 1; }
done

declare -A DS_R1 DS_R2 DS_LABEL
DS_R1[Panel_PDAC_PE_1M]="$PANEL_PUBLIC_DIR/DRR262998_1.fastq.gz"
DS_R2[Panel_PDAC_PE_1M]="$PANEL_PUBLIC_DIR/DRR262998_2.fastq.gz"
DS_LABEL[Panel_PDAC_PE_1M]="PDAC 4-gene panel ~1M reads (DRR262998)"
DS_R1[Panel_PDAC_PE_1.5M]="$PANEL_PUBLIC_DIR/DRR263018_1.fastq.gz"
DS_R2[Panel_PDAC_PE_1.5M]="$PANEL_PUBLIC_DIR/DRR263018_2.fastq.gz"
DS_LABEL[Panel_PDAC_PE_1.5M]="PDAC 4-gene panel ~1.5M reads (DRR263018)"

for ds in Panel_PDAC_PE_1M Panel_PDAC_PE_1.5M; do
    [[ -f "${DS_R1[$ds]}" ]] || { echo "ERROR: missing dataset file ${DS_R1[$ds]}" >&2; exit 1; }
    [[ -f "${DS_R2[$ds]}" ]] || { echo "ERROR: missing dataset file ${DS_R2[$ds]}" >&2; exit 1; }
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_DIR/$TIMESTAMP"
WORK_DIR="$RUN_DIR/work"
mkdir -p "$WORK_DIR"
CSV="$RUN_DIR/competitor_benchmark_${TIMESTAMP}.csv"
echo "dataset,tool,threads,walltime_s,max_rss_kb,exit_code" > "$CSV"

echo "=== Competitor benchmark $TIMESTAMP (threads=$THREADS) ==="
echo "Output: $RUN_DIR"

# Runs one command under /usr/bin/time -v, appends a row to $CSV.
# Args: dataset tool_label threads -- <command...>
run_timed() {
    local dataset="$1" tool="$2" threads="$3"; shift 3
    [[ "$1" == "--" ]] && shift
    local log="$WORK_DIR/${dataset}_${tool}.timelog"
    local wall="NA" rss="NA" rc="NA"
    if /usr/bin/time -v "$@" > "$WORK_DIR/${dataset}_${tool}.stdout" 2> "$log"; then
        rc=0
    else
        rc=$?
    fi
    wall=$(grep "Elapsed (wall clock)" "$log" | awk -F': ' '{print $2}' | \
           awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print $1 }')
    rss=$(grep "Maximum resident set size" "$log" | awk -F': ' '{print $2}')
    echo "$dataset,$tool,$threads,${wall:-NA},${rss:-NA},$rc" >> "$CSV"
    echo "  $dataset / $tool : wall=${wall:-NA}s rss=${rss:-NA}kB exit=$rc"
}

for ds in Panel_PDAC_PE_1M Panel_PDAC_PE_1.5M; do
    r1="${DS_R1[$ds]}"; r2="${DS_R2[$ds]}"
    echo "--- ${DS_LABEL[$ds]} ---"

    run_timed "$ds" fastp_cpu "$THREADS" -- \
        "$FASTP_CPU_BIN" -w "$THREADS" \
        -i "$r1" -I "$r2" \
        -o "$WORK_DIR/${ds}_fastp_cpu_1.fq.gz" -O "$WORK_DIR/${ds}_fastp_cpu_2.fq.gz" \
        -j "$WORK_DIR/${ds}_fastp_cpu.json" -h "$WORK_DIR/${ds}_fastp_cpu.html"

    run_timed "$ds" fastp_gpu "$THREADS" -- \
        "$FASTP_GPU_BIN" -w "$THREADS" \
        -i "$r1" -I "$r2" \
        -o "$WORK_DIR/${ds}_fastp_gpu_1.fq.gz" -O "$WORK_DIR/${ds}_fastp_gpu_2.fq.gz" \
        -j "$WORK_DIR/${ds}_fastp_gpu.json" -h "$WORK_DIR/${ds}_fastp_gpu.html"

    # FastQC: -t parallelizes across input files (2 files -> 2 threads is the
    # useful max); it does not multi-thread within a single file.
    fastqc_threads=$(( THREADS < 2 ? THREADS : 2 ))
    run_timed "$ds" fastqc "$fastqc_threads" -- \
        "$FASTQC_BIN" -t "$fastqc_threads" -o "$WORK_DIR" "$r1" "$r2"

    # Falco: unlike FastQC's per-process-per-file model, falco supports -t
    # directly and analyzes one file per invocation; run once per read file.
    mkdir -p "$WORK_DIR/${ds}_falco_r1" "$WORK_DIR/${ds}_falco_r2"
    run_timed "$ds" falco_r1 "$THREADS" -- \
        "$FALCO_BIN" -t "$THREADS" -o "$WORK_DIR/${ds}_falco_r1" "$r1"
    run_timed "$ds" falco_r2 "$THREADS" -- \
        "$FALCO_BIN" -t "$THREADS" -o "$WORK_DIR/${ds}_falco_r2" "$r2"
done

echo
echo "Done. CSV: $CSV"
