#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Full three-tool benchmark with profiling + validation
#
# Tools   : opengene fastp 1.3.3
#           fastp-d0bromir CPU (fastp-cpu)
#           fastp-d0bromir GPU (fastp)
#
# Datasets: Panel SE 148M  (S1A_S1_L001_R1)
#           WGS SE 6.3G    (ERR1044780 R1 only)
#           WGS PE 12.8G   (ERR1044780 R1+R2)
#           WGS PE 18.2G   (ERR1044319 R1+R2)
#           WGS PE 40G     (DRR216653 R1+R2)
#
# Thread counts: 1 2 4 8 16 32 (default; override with -t)
#
# Outputs per run:
#   - gzip FASTQ  (exercises parallel output compression)
#   - JSON report (used for validation)
#   - Verbose stderr (profiling stage timings)
#
# Validation (mandatory, hard-fails on mismatch):
#   1. Read-count check, every thread count: reads_in / reads_out of every
#      d0bromir run must equal the matching opengene baseline.
#   2. Deep correctness check, once per dataset at VALIDATE_THREADS
#      (default 8): the decompressed output FASTQ md5 must equal opengene's,
#      and a fixed list of biology fields in the JSON must match exactly.
#      Output files are kept only long enough to compute md5; on PASS they
#      are deleted, on FAIL they are kept for diagnostics.
#
# Override VALIDATE_THREADS=N to run deep validation at a different thread
# count.  If your -t selection does not include VALIDATE_THREADS, deep
# validation is skipped (with a warning); read-count validation still runs.
#
# Pre-flight (always run; can be bypassed with env vars but NOT recommended):
#   1. Rebuild all binaries to be benchmarked via scripts/build_all.sh.
#      (--use-og-csv: only the d0bromir CPU + GPU binaries are rebuilt.)
#      Bypass with SKIP_PREFLIGHT_BUILD=1.
#   2. Run scripts/run_tests.sh (built-in unit tests + CPU<->GPU regression
#      on testdata/).  Bypass with SKIP_PREFLIGHT_TESTS=1.
#   Any failure in either step aborts the benchmark with exit 2.
#
# Usage:
#   ./run_benchmark.sh                           # all datasets, all threads
#   ./run_benchmark.sh -d WGS_SE_6.3G            # single dataset
#   ./run_benchmark.sh -d WGS_SE_6.3G -d WGS_PE_40G   # multiple datasets
#   ./run_benchmark.sh -t 8                      # single thread count
#   ./run_benchmark.sh -t 4 -t 8 -t 16           # multiple thread counts
#   ./run_benchmark.sh -d WGS_PE_40G -t 8        # combine both filters
#   ./run_benchmark.sh --list                     # list available datasets
#
# Available dataset names:
#   Panel_SE_148M   Panel_PE_304M   Panel_PDAC_PE_1M   Panel_PDAC_PE_1.5M
#   WGS_SE_6.3G   WGS_PE_12.8G   WGS_PE_18.2G   WGS_PE_40G
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Single-instance enforcement.
#
# Reasons to refuse to start:
#   1. Another run_benchmark.sh is already holding the flock on
#      $LOCK_FILE.  Concurrent benchmarks would (a) thrash the page cache,
#      (b) race for the GPU, and (c) corrupt timing measurements.
#   2. Any standalone fastp invocation (`fastp -w ...`) is running on this
#      host that is NOT a child of our own benchmark.  Side fastp jobs
#      contend for the same CPU/GPU/RAM and have caused futex hangs in the
#      past.
#
# Override with FORCE_RUN=1 (use only if you really know what you're doing).
# ---------------------------------------------------------------------------
LOCK_FILE="${BENCH_LOCK_FILE:-/tmp/fastp_run_benchmark.lock}"
exec 9>"$LOCK_FILE" || { echo "ERROR: cannot open lock file $LOCK_FILE" >&2; exit 3; }
if ! flock -n 9; then
    other_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [[ "${FORCE_RUN:-0}" == "1" ]]; then
        echo "WARNING: another run_benchmark.sh holds the lock (PID=$other_pid); FORCE_RUN=1 set, continuing anyway." >&2
    else
        echo "ERROR: another run_benchmark.sh is already running (PID=$other_pid; lock=$LOCK_FILE)." >&2
        echo "       Use './scripts/check_status.sh' to inspect, or set FORCE_RUN=1 to override." >&2
        exit 3
    fi
fi
# Record our PID inside the lock for diagnostics; the flock keeps the lock
# regardless of file contents.
echo $$ >&9

# Detect any stray fastp not under our control.  Children of this script
# are launched with our PGID so we can match by PGID to exclude them.
my_pgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
stray_fastp=$(ps -e -o pid=,pgid=,comm=,args= 2>/dev/null \
    | awk -v g="$my_pgid" '$3=="fastp" && $2!=g {print}')
if [[ -n "$stray_fastp" ]]; then
    if [[ "${FORCE_RUN:-0}" == "1" ]]; then
        echo "WARNING: foreign fastp process(es) detected; FORCE_RUN=1 set, continuing." >&2
        echo "$stray_fastp" >&2
    else
        echo "ERROR: a foreign fastp process is running on this host:" >&2
        echo "$stray_fastp" >&2
        echo "       Concurrent fastp jobs invalidate timings and have caused hangs." >&2
        echo "       Stop them, or set FORCE_RUN=1 to override." >&2
        exit 3
    fi
fi

# Output goes to benchmark_results/<TS>/ as a self-contained run folder ready
# to be moved into the fastp-gpu_<v>/vs_opengene_<v>/<platform>/ layout.
RESULTS_ROOT="$PROJ_DIR/benchmark_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$RESULTS_ROOT/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

CSV="$RESULTS_DIR/full_benchmark_${TIMESTAMP}.csv"
PROFILE_LOG="$RESULTS_DIR/full_profiling_${TIMESTAMP}.log"
VALID_LOG="$RESULTS_DIR/full_validation_${TIMESTAMP}.log"
STATUS_FILE="$RESULTS_DIR/full_status_${TIMESTAMP}.txt"

OPENGENE="$PROJ_DIR/../fastp_opengene/fastp"
CPU_BIN="$PROJ_DIR/fastp-cpu"
GPU_BIN="$PROJ_DIR/fastp"
OG_CSV=""

# Per-run timeout (seconds).  Hard wall-time limit — kills the fastp subprocess
# and marks the row TIMEOUT in the CSV so the benchmark can continue.
# Set to 0 to disable.
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}

# Hang detection via CPU monitoring.  After HANG_GRACE_SECS of startup time,
# if process CPU stays below HANG_CPU_PCT% for HANG_IDLE_SECS consecutive
# seconds the process is killed and marked TIMEOUT.
HANG_GRACE_SECS=${HANG_GRACE_SECS:-20}   # ignore first N secs (I/O-heavy start)
HANG_IDLE_SECS=${HANG_IDLE_SECS:-30}     # how long low-CPU must persist
HANG_CPU_PCT=${HANG_CPU_PCT:-5}          # % threshold below which = idle

# Output-stagnation detector: kill a run whose output FASTQ has not grown
# for this many seconds (catches livelocks where CPU is high but no data
# is produced — e.g. spinning on a futex or hrtimer_nanosleep).
HANG_OUTPUT_IDLE_SECS=${HANG_OUTPUT_IDLE_SECS:-180}

# Crash / hang retry policy.  A run is retried up to MAX_ATTEMPTS times if
# the binary exits non-zero or the watchdog kills it.  The successful
# attempt is the one written to the CSV; the attempts column records how
# many tries were needed.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

# Resource sampler interval (seconds).  Each fastp run is sampled every N
# seconds for CPU%, RSS, GPU util/memory.  Aggregates (avg/peak) land in
# both the CSV and the profiling log.
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-1}

# Resume: when set, completed (dataset, tool, threads) triples present in
# the given CSV are skipped and the new rows are appended to the same CSV
# / profile / validation logs.  Auto-detection (newest CSV under the
# benchmark_results subtree) is enabled by default but disabled by
# specifying --no-resume.
RESUME_CSV=""
NO_RESUME=0

# Thread count at which deep validation (FASTQ md5 + JSON biology fields)
# runs once per dataset.  Read-count validation runs at every thread count
# regardless.
VALIDATE_THREADS=${VALIDATE_THREADS:-8}

# Biology fields whose values must match opengene exactly (deep validation).
VALIDATE_JSON_FIELDS=(
    summary.before_filtering.total_reads
    summary.before_filtering.total_bases
    summary.after_filtering.total_reads
    summary.after_filtering.total_bases
    summary.after_filtering.q20_rate
    summary.after_filtering.q30_rate
    filtering_result.passed_filter_reads
    filtering_result.low_quality_reads
    filtering_result.too_many_N_reads
    filtering_result.too_short_reads
    adapter_cutting.adapter_trimmed_reads
    adapter_cutting.adapter_trimmed_bases
)

# ---------------------------------------------------------------------------
# All available datasets
# ---------------------------------------------------------------------------
# Override via env: FASTQ_DIR (default $HOME/FASTQ), WGS_DIR (default
# $FASTQ_DIR/WGS), PANEL_PUBLIC_DIR (default $FASTQ_DIR/Panel_public).
FASTQ_DIR="${FASTQ_DIR:-$HOME/FASTQ}"
WGS_DIR="${WGS_DIR:-$FASTQ_DIR/WGS}"
RNA_DIR="${RNA_DIR:-$FASTQ_DIR/RNA}"
PANEL_PUBLIC_DIR="${PANEL_PUBLIC_DIR:-$FASTQ_DIR/Panel_public}"

declare -A DS_TYPE DS_R1 DS_R2 DS_LABEL

DS_TYPE[Panel_SE_148M]="SE"
DS_R1[Panel_SE_148M]="$FASTQ_DIR/S1A_S1_L001_R1_001.fastq.gz"
DS_R2[Panel_SE_148M]=""
DS_LABEL[Panel_SE_148M]="Panel SE 148M (S1A_S1_L001_R1)"

DS_TYPE[Panel_PE_304M]="PE"
DS_R1[Panel_PE_304M]="$FASTQ_DIR/S1A_S1_L001_R1_001.fastq.gz"
DS_R2[Panel_PE_304M]="$FASTQ_DIR/S1A_S1_L001_R2_001.fastq.gz"
DS_LABEL[Panel_PE_304M]="Panel PE 304M (S1A_S1_L001 R1+R2)"

DS_TYPE[WGS_SE_6.3G]="SE"
DS_R1[WGS_SE_6.3G]="$WGS_DIR/ERR1044780_1.fastq.gz"
DS_R2[WGS_SE_6.3G]=""
DS_LABEL[WGS_SE_6.3G]="WGS SE 6.3G (ERR1044780_1)"

DS_TYPE[WGS_PE_12.8G]="PE"
DS_R1[WGS_PE_12.8G]="$WGS_DIR/ERR1044780_1.fastq.gz"
DS_R2[WGS_PE_12.8G]="$WGS_DIR/ERR1044780_2.fastq.gz"
DS_LABEL[WGS_PE_12.8G]="WGS PE 12.8G (ERR1044780 R1+R2)"

DS_TYPE[WGS_PE_18.2G]="PE"
DS_R1[WGS_PE_18.2G]="$WGS_DIR/ERR1044319_1.fastq.gz"
DS_R2[WGS_PE_18.2G]="$WGS_DIR/ERR1044319_2.fastq.gz"
DS_LABEL[WGS_PE_18.2G]="WGS PE 18.2G (ERR1044319 R1+R2)"

DS_TYPE[WGS_PE_40G]="PE"
DS_R1[WGS_PE_40G]="$WGS_DIR/DRR216653_1.fastq.gz"
DS_R2[WGS_PE_40G]="$WGS_DIR/DRR216653_2.fastq.gz"
DS_LABEL[WGS_PE_40G]="WGS PE 40G (DRR216653 R1+R2)"

# Public panel alternatives for the private TST-15 panel (same technology:
# MiSeq PE amplicon; PDAC 4-gene hotspot panel KRAS/TP53/CDKN2A/SMAD4)
# Download with: ./scripts/download_bench_samples.sh panel
DS_TYPE[Panel_PDAC_PE_1M]="PE"
DS_R1[Panel_PDAC_PE_1M]="$PANEL_PUBLIC_DIR/DRR262998_1.fastq.gz"
DS_R2[Panel_PDAC_PE_1M]="$PANEL_PUBLIC_DIR/DRR262998_2.fastq.gz"
DS_LABEL[Panel_PDAC_PE_1M]="Panel PDAC PE 166MB (DRR262998, 1.0M reads, MiSeq 156bp)"

DS_TYPE[Panel_PDAC_PE_1.5M]="PE"
DS_R1[Panel_PDAC_PE_1.5M]="$PANEL_PUBLIC_DIR/DRR263018_1.fastq.gz"
DS_R2[Panel_PDAC_PE_1.5M]="$PANEL_PUBLIC_DIR/DRR263018_2.fastq.gz"
DS_LABEL[Panel_PDAC_PE_1.5M]="Panel PDAC PE 249MB (DRR263018, 1.5M reads, MiSeq 156bp)"

# HiSeq X Ten WGS PE 151bp — Personalis 1KGP (PRJEB6463)
DS_TYPE[WGS_HiSeqX_PE_5.7G]="PE"
DS_R1[WGS_HiSeqX_PE_5.7G]="$WGS_DIR/ERR1305370_1.fastq.gz"
DS_R2[WGS_HiSeqX_PE_5.7G]="$WGS_DIR/ERR1305370_2.fastq.gz"
DS_LABEL[WGS_HiSeqX_PE_5.7G]="WGS HiSeqX PE 5.7G (ERR1305370, 151bp)"

DS_TYPE[WGS_HiSeqX_PE_5.4G]="PE"
DS_R1[WGS_HiSeqX_PE_5.4G]="$WGS_DIR/ERR1305374_1.fastq.gz"
DS_R2[WGS_HiSeqX_PE_5.4G]="$WGS_DIR/ERR1305374_2.fastq.gz"
DS_LABEL[WGS_HiSeqX_PE_5.4G]="WGS HiSeqX PE 5.4G (ERR1305374, 151bp)"

# HiSeq 2000 BS-Seq SE 52bp — short-read stress test (PRJNA387623)
DS_TYPE[BS_HiSeq_SE_26.8G]="SE"
DS_R1[BS_HiSeq_SE_26.8G]="$WGS_DIR/SRR5589994.fastq.gz"
DS_R2[BS_HiSeq_SE_26.8G]=""
DS_LABEL[BS_HiSeq_SE_26.8G]="BS-Seq HiSeq SE 26.8G (SRR5589994, 52bp)"

# RNA-Seq across three platforms / read lengths
DS_TYPE[RNA_HiSeqX_PE_6.1G]="PE"
DS_R1[RNA_HiSeqX_PE_6.1G]="$RNA_DIR/SRR13074117_1.fastq.gz"
DS_R2[RNA_HiSeqX_PE_6.1G]="$RNA_DIR/SRR13074117_2.fastq.gz"
DS_LABEL[RNA_HiSeqX_PE_6.1G]="RNA-Seq HiSeqX PE 6.1G (SRR13074117, 152bp)"

DS_TYPE[RNA_NextSeq_PE_1.2G]="PE"
DS_R1[RNA_NextSeq_PE_1.2G]="$RNA_DIR/DRR346892_1.fastq.gz"
DS_R2[RNA_NextSeq_PE_1.2G]="$RNA_DIR/DRR346892_2.fastq.gz"
DS_LABEL[RNA_NextSeq_PE_1.2G]="RNA-Seq NextSeq PE 1.2G (DRR346892, 71bp)"

DS_TYPE[RNA_HiSeq2500_PE_3.1G]="PE"
DS_R1[RNA_HiSeq2500_PE_3.1G]="$RNA_DIR/SRR5186485_1.fastq.gz"
DS_R2[RNA_HiSeq2500_PE_3.1G]="$RNA_DIR/SRR5186485_2.fastq.gz"
DS_LABEL[RNA_HiSeq2500_PE_3.1G]="RNA-Seq HiSeq2500 PE 3.1G (SRR5186485, 150bp)"

ALL_DATASETS=(Panel_SE_148M Panel_PE_304M Panel_PDAC_PE_1M Panel_PDAC_PE_1.5M \
              RNA_NextSeq_PE_1.2G RNA_HiSeq2500_PE_3.1G \
              WGS_HiSeqX_PE_5.4G WGS_HiSeqX_PE_5.7G RNA_HiSeqX_PE_6.1G \
              WGS_SE_6.3G WGS_PE_12.8G WGS_PE_18.2G BS_HiSeq_SE_26.8G WGS_PE_40G)
ALL_THREADS=(1 2 4 8 16 32)

# ---------------------------------------------------------------------------
# Scenario matrix — different fastp option sets exercising different
# code paths for PhD-quality benchmarking.  One scenario per run; rerun the
# benchmark with --scenario NAME to cover the matrix.
# ---------------------------------------------------------------------------
declare -A SCENARIO_OPTS
SCENARIO_OPTS[default]=""
SCENARIO_OPTS[detect_adapter]="--detect_adapter_for_pe"
SCENARIO_OPTS[corr_overrep]="--correction --overrepresentation_analysis"
SCENARIO_OPTS[cut_right]="--cut_right --cut_window_size 4 --cut_mean_quality 20"
# PE-only scenarios are skipped on SE datasets.
declare -A SCENARIO_PE_ONLY
SCENARIO_PE_ONLY[detect_adapter]=1
SCENARIO_PE_ONLY[corr_overrep]=1   # --correction is PE-only
ALL_SCENARIOS=(default detect_adapter corr_overrep cut_right)
SCENARIO="${SCENARIO:-default}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SELECTED_DATASETS=()
SELECTED_THREADS=()

usage() {
    echo "Usage: $0 [-d DATASET]... [-t THREADS]... [--list] [--use-og-csv [FILE]] [--validate-threads N]"
    echo ""
    echo "Options:"
    echo "  -d DATASET           Run only this dataset (repeatable)"
    echo "  -t THREADS           Run only this thread count (repeatable)"
    echo "  --list               List available dataset names and exit"
    echo "  --use-og-csv [FILE]  Skip opengene runs; load baseline from FILE."
    echo "                       Disables deep (FASTQ md5 + JSON-fields)"
    echo "                       validation since opengene FASTQ is not produced."
    echo "  --validate-threads N Run deep validation at thread count N"
    echo "                       (default: \$VALIDATE_THREADS=$VALIDATE_THREADS)"
    echo "  --resume FILE        Resume an interrupted benchmark, appending"
    echo "                       to FILE.  By default, the newest unfinished"
    echo "                       CSV under benchmark_results/ is auto-detected."
    echo "                       Skips runs that already have a non-TIMEOUT,"
    echo "                       non-FAIL row in that CSV."
    echo "  --no-resume          Disable resume auto-detection."
    echo "  --max-attempts N     Retry crashed runs up to N times (default 3)."
    echo "  --timeout SECS       Hard per-run wall-time limit; the fastp subprocess"
    echo "                       is killed and the row marked TIMEOUT after this"
    echo "                       many seconds (default \$RUN_TIMEOUT=$RUN_TIMEOUT)."
    echo "                       Use 0 to disable the hard limit (output-stagnation"
    echo "                       and CPU-idle hang detectors stay active)."
    echo "  --hang-output-idle N Kill a run if its output FASTQ has not grown for N"
    echo "                       consecutive seconds (default \$HANG_OUTPUT_IDLE_SECS=$HANG_OUTPUT_IDLE_SECS)."
    echo "  --scenario NAME      Run a named option scenario for this benchmark"
    echo "                       (one scenario per invocation; defaults to 'default')."
    echo "                       Available: ${ALL_SCENARIOS[*]}"
    echo "  --list-scenarios     List scenarios + their fastp options and exit."
    echo "  -h, --help           Show this help"
    echo ""
    echo "Available datasets:"
    for ds in "${ALL_DATASETS[@]}"; do
        printf "  %-20s  %s  %s\n" "$ds" "${DS_TYPE[$ds]}" "${DS_LABEL[$ds]}"
    done
    echo ""
    echo "Default thread counts: ${ALL_THREADS[*]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)
            shift
            if [[ -z "${DS_TYPE[$1]:-}" ]]; then
                echo "ERROR: Unknown dataset '$1'"
                echo "Available: ${ALL_DATASETS[*]}"
                exit 1
            fi
            SELECTED_DATASETS+=("$1")
            shift
            ;;
        -t)
            shift
            SELECTED_THREADS+=("$1")
            shift
            ;;
        --list)
            echo "Available datasets:"
            for ds in "${ALL_DATASETS[@]}"; do
                printf "  %-20s  %s  %s\n" "$ds" "${DS_TYPE[$ds]}" "${DS_LABEL[$ds]}"
            done
            exit 0
            ;;
        --use-og-csv)
            shift
            if [[ $# -gt 0 && "${1:-}" != -* ]]; then
                OG_CSV="$1"
                shift
            else
                OG_CSV="$RESULTS_ROOT/fastp-gpu_v1.2.2/vs_opengene_v1.1.0/galaxy_arm_a100/20260417/full_benchmark_20260417.csv"
            fi
            ;;
        --validate-threads)
            shift
            VALIDATE_THREADS="$1"
            shift
            ;;
        --resume)
            shift
            RESUME_CSV="$1"
            shift
            ;;
        --no-resume)
            NO_RESUME=1
            shift
            ;;
        --max-attempts)
            shift
            MAX_ATTEMPTS="$1"
            shift
            ;;
        --timeout)
            shift
            if ! [[ "$1" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --timeout requires a non-negative integer (seconds), got '$1'"
                exit 1
            fi
            RUN_TIMEOUT="$1"
            shift
            ;;
        --hang-output-idle)
            shift
            if ! [[ "$1" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --hang-output-idle requires a non-negative integer (seconds), got '$1'"
                exit 1
            fi
            HANG_OUTPUT_IDLE_SECS="$1"
            shift
            ;;
        --scenario)
            shift
            if [[ -z "${SCENARIO_OPTS[$1]+x}" ]]; then
                echo "ERROR: Unknown scenario '$1'"
                echo "Available: ${ALL_SCENARIOS[*]}"
                exit 1
            fi
            SCENARIO="$1"
            shift
            ;;
        --list-scenarios)
            echo "Available scenarios:"
            for s in "${ALL_SCENARIOS[@]}"; do
                printf "  %-16s  %s\n" "$s" "${SCENARIO_OPTS[$s]:-(no extra options)}"
            done
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option '$1'"
            usage
            exit 1
            ;;
    esac
done

# Fall back to all if none selected
if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
    SELECTED_DATASETS=("${ALL_DATASETS[@]}")
fi
if [[ ${#SELECTED_THREADS[@]} -eq 0 ]]; then
    SELECTED_THREADS=("${ALL_THREADS[@]}")
fi

DATASETS=("${SELECTED_DATASETS[@]}")
THREAD_COUNTS=("${SELECTED_THREADS[@]}")

# If a non-default scenario was selected, suffix the output filenames so
# results from different scenarios coexist in the same timestamp folder.
if [[ "$SCENARIO" != "default" ]]; then
    CSV="$RESULTS_DIR/full_benchmark_${TIMESTAMP}_${SCENARIO}.csv"
    PROFILE_LOG="$RESULTS_DIR/full_profiling_${TIMESTAMP}_${SCENARIO}.log"
    VALID_LOG="$RESULTS_DIR/full_validation_${TIMESTAMP}_${SCENARIO}.log"
    STATUS_FILE="$RESULTS_DIR/full_status_${TIMESTAMP}_${SCENARIO}.txt"
fi

# Stores wall times for % calculation:  key = "ds:tool:threads"
declare -A WALL_TIMES
# Stores opengene reference JSON paths:  key = "ds" (only at VALIDATE_THREADS)
declare -A OG_JSON
# Stores opengene read counts for validation:  key = "ds:threads"
declare -A OG_READS_IN OG_READS_OUT
# Stores opengene reference FASTQ md5s (decompressed) for deep validation,
# only at VALIDATE_THREADS:  key = "ds"
declare -A OG_FQ1_MD5 OG_FQ2_MD5

# ---------------------------------------------------------------------------
# fq_md5 — md5 of decompressed FASTQ contents (gzip-blocking-independent).
# ---------------------------------------------------------------------------
fq_md5() {
    [[ -f "$1" ]] || { echo ""; return; }
    gunzip -c "$1" | md5sum | awk '{print $1}'
}

# ---------------------------------------------------------------------------
# json_field_get — extract a dotted-path field from JSON (returns empty on
# missing).  Floats normalized to 6 decimals to absorb formatting noise.
# ---------------------------------------------------------------------------
json_field_get() {
    python3 - "$1" "$2" <<'PY' 2>/dev/null || true
import json, sys
f, p = sys.argv[1], sys.argv[2]
d = json.load(open(f))
for k in p.split('.'):
    if isinstance(d, dict) and k in d:
        d = d[k]
    else:
        sys.exit(0)
if isinstance(d, float):
    print(f"{d:.6f}")
else:
    print(d)
PY
}

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
reads_in()  { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['summary']['before_filtering']['total_reads'])" "$1" 2>/dev/null || echo "0"; }
# Use filtering_result.passed_filter_reads — this is the authoritative count of reads
# actually written to output, and is consistent across both opengene and d0bromir.
reads_out() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['filtering_result']['passed_filter_reads'])" "$1" 2>/dev/null || echo "0"; }

# ---------------------------------------------------------------------------
# Drop page cache between runs for consistent I/O
# ---------------------------------------------------------------------------
drop_cache() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true; }

# ---------------------------------------------------------------------------
# Total run count (used by resume-detection and progress display)
# ---------------------------------------------------------------------------
if [[ -n "$OG_CSV" ]]; then
    TOTAL_RUNS=$(( ${#DATASETS[@]} * 2 * ${#THREAD_COUNTS[@]} ))
else
    TOTAL_RUNS=$(( ${#DATASETS[@]} * 3 * ${#THREAD_COUNTS[@]} ))
fi
RUN_NUM=0

# ---------------------------------------------------------------------------
# CSV header (extended with attempts + resource columns).
# Extra columns are appended at the end so legacy DictReader-based scripts
# still work.
# ---------------------------------------------------------------------------
CSV_HEADER="dataset,type,tool,threads,rep,walltime_s,reads_in,reads_out,pct_vs_opengene,validation,attempts,cpu_avg_pct,cpu_peak_pct,rss_peak_mib,gpu_util_avg_pct,gpu_util_peak_pct,gpu_mem_peak_mib,scenario"

# ---------------------------------------------------------------------------
# Resume support: if --resume FILE is given (or auto-detect newest CSV),
# load the set of (dataset, tool, threads) triples already completed and
# skip them.  New rows are appended to the same CSV / profile / valid log.
# ---------------------------------------------------------------------------
declare -A COMPLETED
resolve_resume_csv() {
    [[ "$NO_RESUME" == "1" ]] && return 0
    if [[ -z "$RESUME_CSV" ]]; then
        # Auto-detect: pick newest full_benchmark_*.csv under
        # benchmark_results/ that is (a) less than 24h old AND
        # (b) has fewer rows than TOTAL_RUNS (i.e. likely incomplete).
        local newest find_pat
        if [[ "$SCENARIO" == "default" ]]; then
            # Default scenario: match files NOT containing a scenario suffix.
            newest=$(find "$PROJ_DIR/benchmark_results" -name 'full_benchmark_*.csv' \
                     -mmin -1440 -printf '%T@ %p\n' 2>/dev/null \
                     | awk '{ b=$2; sub(/.*\//,"",b); if (b ~ /^full_benchmark_[0-9_]+\.csv$/) print }' \
                     | sort -rn | head -1 | awk '{print $2}')
        else
            find_pat="full_benchmark_*_${SCENARIO}.csv"
            newest=$(find "$PROJ_DIR/benchmark_results" -name "$find_pat" \
                     -mmin -1440 -printf '%T@ %p\n' 2>/dev/null \
                     | sort -rn | head -1 | awk '{print $2}')
        fi
        if [[ -n "$newest" && -f "$newest" ]]; then
            local rows; rows=$(( $(wc -l < "$newest") - 1 ))
            if (( rows > 0 && rows < TOTAL_RUNS )); then
                RESUME_CSV="$newest"
                echo "Auto-detected resume CSV: $RESUME_CSV ($rows rows; pass --no-resume to ignore)"
            fi
        fi
    fi
}
load_completed_runs() {
    [[ -z "$RESUME_CSV" || ! -f "$RESUME_CSV" ]] && return 0
    local n=0
    while IFS=, read -r ds tp tool thr rep wall ri ro pct vl rest; do
        [[ "$ds" == "dataset" ]] && continue
        # Treat OK / PASS / PASS+FQ / baseline / CSV / SKIP as completed.
        # TIMEOUT / FAIL / CRASH runs are NOT completed and will retry.
        case "${pct},${vl}" in
            *TIMEOUT*|*FAIL*|*CRASH*) continue ;;
        esac
        COMPLETED["${ds}:${tool}:${thr}"]=1
        n=$(( n + 1 ))
    done < "$RESUME_CSV"
    echo "Resume: marking $n completed runs as already-done."
}

if [[ -n "$RESUME_CSV" || "$NO_RESUME" != "1" ]]; then
    resolve_resume_csv
fi

if [[ -n "$RESUME_CSV" && -f "$RESUME_CSV" ]]; then
    # Re-use the same CSV / sibling logs so output is appended.
    CSV="$RESUME_CSV"
    OUT_DIR="$(dirname "$CSV")"
    base="$(basename "$CSV" .csv)"; ts="${base#full_benchmark_}"
    PROFILE_LOG="$OUT_DIR/full_profiling_${ts}.log"
    VALID_LOG="$OUT_DIR/full_validation_${ts}.log"
    STATUS_FILE="$OUT_DIR/full_status_${ts}.txt"
    load_completed_runs
    printf "\n=== Resumed: %s ===\n\n" "$(date)" >> "$PROFILE_LOG"
    printf "\n=== Resumed: %s ===\n\n" "$(date)" >> "$VALID_LOG"
else
    echo "$CSV_HEADER" > "$CSV"
    printf "=== Full Benchmark Log: %s ===\n\n" "$(date)" > "$PROFILE_LOG"
    printf "=== Validation Log: %s ===\n\n"    "$(date)" > "$VALID_LOG"
fi

# ---------------------------------------------------------------------------
# write_status — builds the human-readable STATUS_FILE
# ---------------------------------------------------------------------------
write_status() {
    local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
    # Compute progress from the CSV directly so the background ticker
    # subshell (which has a stale copy of $RUN_NUM from fork time) reports
    # the real number of completed runs.
    local csv_rows=0
    if [[ -s "$CSV" ]]; then
        csv_rows=$(( $(wc -l < "$CSV") - 1 ))
        (( csv_rows < 0 )) && csv_rows=0
    fi
    # Fall back to the in-shell counter when foreground caller is ahead of
    # the CSV (e.g. a row currently being written).
    local progress=$csv_rows
    (( RUN_NUM > progress )) && progress=$RUN_NUM
    {
        printf "============================================================\n"
        printf " BENCHMARK STATUS — %s\n" "$ts"
        printf " Progress : %d / %d runs\n" "$progress" "$TOTAL_RUNS"
        printf " Datasets : %s\n" "${DATASETS[*]}"
        printf " Threads  : %s\n" "${THREAD_COUNTS[*]}"
        printf " CSV      : %s\n" "$CSV"
        printf "============================================================\n\n"

        if [[ -s "$CSV" ]] && [[ $(wc -l < "$CSV") -gt 1 ]]; then
            printf "%-22s  %-4s  %-15s  %5s  %10s  %12s  %12s  %12s  %s\n" \
                "dataset" "type" "tool" "T" "wall_s" "reads_in" "reads_out" "vs_opengene" "valid"
            printf -- "%-22s  %-4s  %-15s  %5s  %10s  %12s  %12s  %12s  %s\n" \
                "----------------------" "----" "---------------" "-----" "----------" "------------" "------------" "------------" "------"
            tail -n +2 "$CSV" | while IFS=, read -r ds tp tool thr rep wt ri ro pct vl _rest; do
                printf "%-22s  %-4s  %-15s  %5s  %10.1f  %12s  %12s  %12s  %s\n" \
                    "$ds" "$tp" "$tool" "$thr" "$wt" "$ri" "$ro" "$pct" "$vl"
            done
        else
            echo "  (no runs completed yet)"
        fi

        if [[ -s "$PROFILE_LOG" ]]; then
            echo ""
            echo "--- Last profiling entries (tail) ---"
            grep -E "tool=|wall=|stage|Stage|seconds|time used|GPU|Init|Compress|Filter|Stats|Write" \
                "$PROFILE_LOG" 2>/dev/null | tail -30 || true
        fi
    } > "$STATUS_FILE"
}

# Background ticker every 3 minutes
ticker_fn() {
    # Release inherited lock fd so this background process does not keep
    # /tmp/fastp_run_benchmark.lock held after the parent exits.
    exec 9>&- 2>/dev/null || true
    while true; do
        sleep 180
        write_status
    done
}
ticker_fn &
TICKER_PID=$!
trap "kill \$TICKER_PID 2>/dev/null || true; rm -f /tmp/fastp_bench_watchdog_\$\$ /tmp/fastp_bench_*.json /tmp/fastp_bench_*_R[12].fq.gz /tmp/fastp_bench_*.stderr 2>/dev/null; echo 'Benchmark interrupted (CSV preserved at $CSV; rerun with --resume to continue).'" EXIT

# ---------------------------------------------------------------------------
# load_og_from_csv — populate WALL_TIMES / OG_READS_IN / OG_READS_OUT from
# a saved benchmark CSV, skipping opengene re-runs.
# Handles both legacy (no rep column) and current full_benchmark_*.csv formats.
# ---------------------------------------------------------------------------
load_og_from_csv() {
    local csv_file="$1"
    local loaded=0
    local hdr; hdr=$(head -1 "$csv_file")
    # Detect column layout:
    #   legacy:        dataset,type,tool,threads,walltime_s,reads_in,reads_out,...
    #   run_benchmark: dataset,type,tool,threads,rep,walltime_s,reads_in,reads_out,...
    local has_rep=0
    [[ "$hdr" == *",rep,"* ]] && has_rep=1

    while IFS=, read -r ds tp tool thr c5 c6 c7 rest; do
        [[ "$ds" == "dataset" ]] && continue
        [[ "$tool" != "opengene" ]] && continue
        # Skip stuck/failed opengene rows so we never compute % vs a
        # truncated baseline wall-time.
        if [[ "$rest" == *TIMEOUT* || "$rest" == *FAIL* || "$rest" == *CRASH* ]]; then
            continue
        fi
        local wall ri ro
        if [[ "$has_rep" -eq 1 ]]; then
            # c5=rep, c6=walltime_s, c7=reads_in, rest starts with reads_out
            wall="$c6"; ri="$c7"; ro="${rest%%,*}"
        else
            # c5=walltime_s, c6=reads_in, c7=reads_out
            wall="$c5"; ri="$c6"; ro="$c7"
        fi
        # Only accept positive numeric wall-times (defensive: a stuck row
        # could have wall=0 even without TIMEOUT marker).
        if ! awk -v w="$wall" 'BEGIN{exit !(w+0 > 0)}'; then
            continue
        fi
        WALL_TIMES["${ds}:opengene:${thr}"]="$wall"
        OG_READS_IN["${ds}:${thr}"]="$ri"
        OG_READS_OUT["${ds}:${thr}"]="$ro"
        (( loaded++ )) || true
    done < "$csv_file"
    echo "  Loaded $loaded opengene baseline entries from: $(basename "$csv_file")"
}

# Load opengene baselines from CSV if requested
if [[ -n "$OG_CSV" ]]; then
    if [[ ! -f "$OG_CSV" ]]; then
        echo "ERROR: --use-og-csv: file not found: $OG_CSV"
        exit 1
    fi
    load_og_from_csv "$OG_CSV"
    echo ""
fi

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Full Benchmark — Three-tool fastp comparison"
echo " Started   : $(date)"
echo " Datasets  : ${DATASETS[*]}"
echo " Threads   : ${THREAD_COUNTS[*]}"
echo " Scenario  : ${SCENARIO}  opts='${SCENARIO_OPTS[$SCENARIO]:-}'"
echo " Total runs: $TOTAL_RUNS"
echo " CSV       : $CSV"
echo " Profile   : $PROFILE_LOG"
echo " Validation: $VALID_LOG"
echo " Status    : $STATUS_FILE"
echo " Validation: read-count at every T; full FASTQ+JSON at T=$VALIDATE_THREADS"
if [[ -n "$OG_CSV" ]]; then
    echo " OG CSV    : $OG_CSV  (opengene runs skipped; deep validation disabled)"
else
    in_set=0
    for t in "${THREAD_COUNTS[@]}"; do [[ "$t" == "$VALIDATE_THREADS" ]] && in_set=1; done
    if [[ "$in_set" -eq 0 ]]; then
        echo " WARNING   : VALIDATE_THREADS=$VALIDATE_THREADS not in selected threads (${THREAD_COUNTS[*]});"
        echo "             deep validation will be SKIPPED. Re-run with -t $VALIDATE_THREADS to enable."
    fi
fi
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight rebuild — always rebuild the binaries that will be benchmarked
# so timings reflect the current source tree.  build_all.sh runs
# `<binary> test` after each build and aborts on any test failure.
# Set SKIP_PREFLIGHT_BUILD=1 to bypass (not recommended; only safe when you
# have just built the binaries yourself in the same session).
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREFLIGHT_BUILD:-0}" != "1" ]]; then
    echo "Pre-flight rebuild (scripts/build_all.sh):"
    if [[ -n "$OG_CSV" ]]; then
        # opengene baseline loaded from CSV — rebuild only d0bromir binaries.
        echo "  --use-og-csv set; rebuilding d0bromir CPU + GPU only"
        "$SCRIPT_DIR/build_all.sh" cpu \
            || { echo "  [FAIL] d0bromir CPU build failed — aborting benchmark." >&2; exit 2; }
        "$SCRIPT_DIR/build_all.sh" gpu \
            || { echo "  [FAIL] d0bromir GPU build failed — aborting benchmark." >&2; exit 2; }
    else
        # Full benchmark: rebuild all three (opengene + d0bromir CPU + GPU).
        "$SCRIPT_DIR/build_all.sh" all \
            || { echo "  [FAIL] build_all.sh failed — aborting benchmark." >&2; exit 2; }
    fi
    echo ""
fi

echo "Binary check:"
for pair in "opengene:$OPENGENE" "d0bromir_cpu:$CPU_BIN" "d0bromir_gpu:$GPU_BIN"; do
    lbl="${pair%%:*}"; pth="${pair##*:}"
    if [[ "$lbl" == "opengene" && -n "$OG_CSV" ]]; then
        echo "  [SKIP] $lbl  (baseline loaded from CSV)"
        continue
    fi
    if [[ ! -x "$pth" ]]; then echo "  [MISSING] $lbl -> $pth"; exit 1; fi
    echo "  [OK] $lbl  $(ls -lh "$pth" | awk '{print $5,$6,$7,$8}')  $pth"
done
echo ""
echo "Dataset check:"
for ds in "${DATASETS[@]}"; do
    [[ ! -f "${DS_R1[$ds]}" ]] && { echo "  [MISSING] ${DS_R1[$ds]}"; exit 1; }
    if [[ "${DS_TYPE[$ds]}" == "PE" ]]; then
        [[ ! -f "${DS_R2[$ds]}" ]] && { echo "  [MISSING] ${DS_R2[$ds]}"; exit 1; }
    fi
    sz=$(du -sh "${DS_R1[$ds]}" | cut -f1)
    echo "  [OK] $ds  ${DS_TYPE[$ds]}  R1=$(basename "${DS_R1[$ds]}") ($sz)"
done
echo ""

# ---------------------------------------------------------------------------
# Pre-flight test suite — built-in unit tests for every binary that will be
# benchmarked, plus CPU↔GPU regression on testdata/.  Hard-fails on any
# error; benchmarking a broken binary is worse than not benchmarking at all.
# Set SKIP_PREFLIGHT_TESTS=1 to bypass (not recommended).
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREFLIGHT_TESTS:-0}" != "1" ]]; then
    echo "Pre-flight test suite (scripts/run_tests.sh):"
    test_args=()
    [[ -n "$OG_CSV" ]] && test_args+=(--skip-opengene)
    if ! OPENGENE_BIN="$OPENGENE" CPU_BIN="$CPU_BIN" GPU_BIN="$GPU_BIN" \
            "$SCRIPT_DIR/run_tests.sh" "${test_args[@]}" -q; then
        echo "  [FAIL] pre-flight tests failed — aborting benchmark." >&2
        exit 2
    fi
    echo ""
fi

echo "All checks passed. Starting $TOTAL_RUNS runs..."
echo ""

# ---------------------------------------------------------------------------
# Resource sampler — every SAMPLE_INTERVAL seconds while $1 is alive,
# append one CSV row to $2:
#     epoch_s,cpu_pct,rss_mib,gpu_util_pct,gpu_mem_mib,input_file
# input_file is the basename of the primary FASTQ ($3) so the per-run
# CSV is self-identifying without relying on the filename.
# Children/threads are aggregated by ps's "tree" of $pid (rollup-style).
# When nvidia-smi is missing, GPU columns are empty.
# ---------------------------------------------------------------------------
start_resource_sampler() {
    local pid="$1" out="$2" input_file="${3:-}"
    # Header + content; basename only so the column stays narrow.
    local input_short=""
    [[ -n "$input_file" ]] && input_short=$(basename "$input_file")
    echo "epoch_s,cpu_pct,rss_mib,gpu_util_pct,gpu_mem_mib,input_file" > "$out"
    (
        # Release inherited lock fd; the sampler must not keep the
        # benchmark lock held if the parent dies.
        exec 9>&- 2>/dev/null || true
        # Disable strict-mode flags inside the sampler.  A transient ps /
        # nvidia-smi failure must not kill the sampler — we'd lose all
        # subsequent rows and the run would have empty resource data.
        set +e +u +o pipefail
        has_gpu=0
        command -v nvidia-smi >/dev/null 2>&1 && has_gpu=1
        while kill -0 "$pid" 2>/dev/null; do
            now=$(date +%s)
            # All processes in pid's process group (covers fastp's worker threads).
            # ps -L would give per-thread; instead aggregate over PGID of pid.
            pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            if [[ -n "$pgid" ]]; then
                read -r cpu rss < <(ps -e -o pgid=,pcpu=,rss= 2>/dev/null \
                    | awk -v g="$pgid" '$1==g {c+=$2; r+=$3} END {printf "%.1f %d", c+0, r+0}')
            else
                cpu=0; rss=0
            fi
            # rss is in KiB → convert to MiB; guard against empty.
            : "${rss:=0}"
            rss=$(( rss / 1024 ))
            : "${cpu:=0}"
            gpu_u=""; gpu_m=""
            if [[ "$has_gpu" -eq 1 ]]; then
                # Average util across all GPUs visible to this host;
                # peak GPU memory used by this PID is queried separately
                # via nvidia-smi --query-compute-apps.
                gpu_u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
                        | awk '{s+=$1;n++} END {if(n>0) printf "%.0f", s/n; else print "0"}')
                gpu_m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
                        | awk -v p="$pid" -F', *' '$1==p {s+=$2} END {printf "%d", s+0}')
                [[ -z "$gpu_u" ]] && gpu_u=0
                [[ -z "$gpu_m" ]] && gpu_m=0
            fi
            printf "%s,%s,%s,%s,%s,%s\n" "$now" "$cpu" "$rss" "$gpu_u" "$gpu_m" "$input_short" >> "$out"
            sleep "${SAMPLE_INTERVAL:-1}"
        done
    ) </dev/null >/dev/null 2>&1 &
    echo $!
}

# ---------------------------------------------------------------------------
# summarize_resource_csv — emit "cpu_avg cpu_peak rss_peak_mib gpu_avg gpu_peak gpu_mem_peak"
# ---------------------------------------------------------------------------
summarize_resource_csv() {
    local f="$1"
    [[ ! -s "$f" ]] && { echo "0 0 0 0 0 0"; return 0; }
    awk -F, 'NR==1 && $1=="epoch_s" {next}
    {
        if ($2+0 > cpu_peak) cpu_peak = $2+0
        cpu_sum += $2+0
        if ($3+0 > rss_peak) rss_peak = $3+0
        if ($4+0 > gpu_peak) gpu_peak = $4+0
        gpu_sum += $4+0
        if ($5+0 > gpu_mem_peak) gpu_mem_peak = $5+0
        n++
    }
    END {
        if (n == 0) { print "0 0 0 0 0 0"; exit }
        printf "%.1f %.1f %d %.1f %.1f %d\n",
            cpu_sum/n, cpu_peak, rss_peak, gpu_sum/n, gpu_peak, gpu_mem_peak
    }' "$f"
}

# ---------------------------------------------------------------------------
# run_one — execute one fastp run, capture timings + output
# ---------------------------------------------------------------------------
# Sets globals: _RUN_WALL _RUN_RI _RUN_RO _RUN_PCT _RUN_JSON _RUN_EXIT
run_one() {
    local tool_label="$1" bin="$2" ds="$3" threads="$4" rep="$5"

    local type="${DS_TYPE[$ds]}"
    local r1="${DS_R1[$ds]}"
    local r2="${DS_R2[$ds]:-}"
    local label="${DS_LABEL[$ds]}"

    local tmp_json tmp_out1 tmp_out2="" tmp_stderr
    tmp_json=$(mktemp /tmp/fastp_bench_XXXXXX.json)
    tmp_out1=$(mktemp /tmp/fastp_bench_XXXXXX_R1.fq.gz)
    tmp_stderr=$(mktemp /tmp/fastp_bench_XXXXXX.stderr)
    [[ "$type" == "PE" ]] && tmp_out2=$(mktemp /tmp/fastp_bench_XXXXXX_R2.fq.gz)

    local cmd=("$bin" -w "$threads" -j "$tmp_json" -h /dev/null --verbose)
    if [[ "$type" == "SE" ]]; then
        cmd+=(-i "$r1" -o "$tmp_out1")
    else
        cmd+=(-i "$r1" -I "$r2" -o "$tmp_out1" -O "$tmp_out2")
    fi
    # Append scenario options (split on whitespace; empty for 'default').
    local scenario_opts="${SCENARIO_OPTS[$SCENARIO]:-}"
    if [[ -n "$scenario_opts" ]]; then
        # shellcheck disable=SC2206
        local _opts_arr=( $scenario_opts )
        cmd+=( "${_opts_arr[@]}" )
    fi

    # Drop page cache before each run
    drop_cache

    local t_start t_end wall_s exit_code=0
    local t_start_sec; t_start_sec=$(date +%s)
    t_start=$(date +%s%N)

    # Launch fastp in background so the watchdog can monitor it.
    # Close fd 9 so the fastp process does not keep the benchmark lock\n    # held if the parent dies (fastp is long-running for big inputs).
    "${cmd[@]}" 9>&- > "$tmp_stderr" 2>&1 &
    local fastp_pid=$!

    # Resource sampler (writes per-second CSV to results dir)
    local sample_dir
    sample_dir="$(dirname "$CSV")/resources"
    mkdir -p "$sample_dir" 2>/dev/null || true
    local sample_csv="${sample_dir}/resource_${ds}_${tool_label}_T${threads}_$(date +%s).csv"
    local sampler_pid
    sampler_pid=$(start_resource_sampler "$fastp_pid" "$sample_csv" "$r1")

    # ---------------------------------------------------------------------------
    # Watchdog: kill if any of these conditions triggers after the startup
    # grace period (HANG_GRACE_SECS):
    #   (a) Wall-time hard limit exceeded (RUN_TIMEOUT, default 1800s).
    #   (b) Process CPU usage stays below HANG_CPU_PCT% for HANG_IDLE_SECS
    #       consecutive seconds  ("idle hang" — e.g. blocked on I/O).
    #   (c) Output FASTQ file size has not grown for HANG_OUTPUT_IDLE_SECS
    #       consecutive seconds  ("livelock hang" — process has plenty of
    #       CPU but is not producing data, e.g. spinning on a futex /
    #       hrtimer_nanosleep waiting for an event that will not arrive).
    #
    # CPU% is measured as (delta jiffies / delta time) × 100 across all
    # threads of the process group, read from /proc/<pid>/stat field 14+15.
    # Output size is the sum of tmp_out1 (+ tmp_out2 for PE).
    # ---------------------------------------------------------------------------
    local hang_killed=0
    local watchdog_log="$(dirname "$CSV")/watchdog_${ds}_${tool_label}_T${threads}_$(date +%s).log"
    # Pre-create the log file from the parent so a dead/dying subshell still
    # leaves a forensic breadcrumb.  If the file ends up empty after the run,
    # we know the watchdog subshell never wrote anything.
    : > "$watchdog_log" 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] PARENT pre-created watchdog_log; about to fork subshell" >> "$watchdog_log"
    (
        # FIRST LINE: write a marker so we know the subshell at least started.
        echo "[$(date +%H:%M:%S)] SUBSHELL ENTERED pid=$$" >> "$watchdog_log" 2>/dev/null
        # Release inherited lock fd; the watchdog must not keep the
        # benchmark lock held if the parent dies.
        exec 9>&- 2>/dev/null || true
        # Disable ALL strict-mode flags inside the watchdog.  The watchdog
        # must NEVER exit early on a transient command failure or unbound
        # variable — if it dies, the run hangs forever (this happened
        # once: stuck opengene WGS_SE went 2219 s without a kill).
        set +e +u +o pipefail
        # Self-logging: any unexpected exit must be diagnosable.
        trap 'echo "[$(date +%H:%M:%S)] WATCHDOG EXIT line=$LINENO rc=$? elapsed=${elapsed:-?} idle=${idle_streak:-?} out_idle=${out_idle_streak:-?}" >> "'"$watchdog_log"'"' EXIT
        echo "[$(date +%H:%M:%S)] WATCHDOG START pid=$$ fastp=$fastp_pid hard=${RUN_TIMEOUT:-1800} cpu_idle=${HANG_IDLE_SECS:-30}s out_idle=${HANG_OUTPUT_IDLE_SECS:-180}s" > "$watchdog_log"

        grace=${HANG_GRACE_SECS:-20}
        idle_limit=${HANG_IDLE_SECS:-30}
        cpu_thresh=${HANG_CPU_PCT:-5}
        out_idle_limit=${HANG_OUTPUT_IDLE_SECS:-180}
        hard_limit=${RUN_TIMEOUT:-1800}
        hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
        : "${hz:=100}"

        elapsed=0
        idle_streak=0
        out_idle_streak=0
        prev_jiffies=0
        prev_ts=0
        prev_out_size=0

        while :; do
            sleep 2
            # Stop if fastp is gone (normal completion or external kill).
            kill -0 "$fastp_pid" 2>/dev/null || {
                echo "[$(date +%H:%M:%S)] WATCHDOG fastp gone, exiting cleanly" >> "$watchdog_log"
                break
            }
            elapsed=$(( $(date +%s) - t_start_sec ))

            # ---- (a) Hard wall-time limit ----
            if [[ "${hard_limit:-0}" -gt 0 && "$elapsed" -ge "$hard_limit" ]]; then
                echo "[$(date +%H:%M:%S)] WATCHDOG HARD_TIMEOUT elapsed=${elapsed}s limit=${hard_limit}s" >> "$watchdog_log"
                kill -TERM "$fastp_pid" 2>/dev/null
                sleep 5
                kill -9   "$fastp_pid" 2>/dev/null
                echo "HARD_TIMEOUT elapsed=${elapsed}s limit=${hard_limit}s" > /tmp/fastp_bench_watchdog_$$
                break
            fi

            # Skip hang detection during grace period.
            if [[ "$elapsed" -lt "$grace" ]]; then
                idle_streak=0
                out_idle_streak=0
                prev_jiffies=0
                prev_ts=0
                prev_out_size=0
                continue
            fi

            # ---- (b) CPU-idle detector ----
            # /proc/PID/stat may transiently fail; treat as "skip this tick".
            stat_line=""
            if [[ -r "/proc/${fastp_pid}/stat" ]]; then
                stat_line=$(cat "/proc/${fastp_pid}/stat" 2>/dev/null)
            fi
            if [[ -n "$stat_line" ]]; then
                cur_jiffies=$(awk '{print $14+$15+$16+$17+0}' <<< "$stat_line" 2>/dev/null)
                : "${cur_jiffies:=0}"
                cur_ts=$(date +%s%N)
                if [[ "${prev_ts:-0}" -gt 0 ]]; then
                    delta_j=$(( cur_jiffies - prev_jiffies ))
                    delta_ns=$(( cur_ts - prev_ts ))
                    cpu_pct=0
                    if [[ "$delta_ns" -gt 0 ]]; then
                        cpu_pct=$(awk -v dj="$delta_j" -v hz="$hz" -v dns="$delta_ns" \
                            'BEGIN{ if (dns>0) printf "%d", (dj/hz)/(dns/1e9)*100; else print 0 }')
                        : "${cpu_pct:=0}"
                    fi
                    if [[ "$cpu_pct" -lt "$cpu_thresh" ]]; then
                        idle_streak=$(( idle_streak + 2 ))
                    else
                        idle_streak=0
                    fi
                    if [[ "$idle_streak" -ge "$idle_limit" ]]; then
                        echo "[$(date +%H:%M:%S)] WATCHDOG CPU_IDLE_HANG cpu=${cpu_pct}% streak=${idle_streak}s" >> "$watchdog_log"
                        kill -TERM "$fastp_pid" 2>/dev/null
                        sleep 5
                        kill -9   "$fastp_pid" 2>/dev/null
                        echo "HANG_DETECTED cpu=${cpu_pct}% cpu_idle=${idle_streak}s" > /tmp/fastp_bench_watchdog_$$
                        break
                    fi
                fi
                prev_jiffies="$cur_jiffies"
                prev_ts="$cur_ts"
            fi

            # ---- (c) Output-size stagnation detector ----
            cur_out_size=0
            sz1=$(stat -c %s "$tmp_out1" 2>/dev/null || echo 0)
            : "${sz1:=0}"
            cur_out_size=$(( cur_out_size + sz1 ))
            if [[ -n "${tmp_out2:-}" ]]; then
                sz2=$(stat -c %s "$tmp_out2" 2>/dev/null || echo 0)
                : "${sz2:=0}"
                cur_out_size=$(( cur_out_size + sz2 ))
            fi
            if [[ "${prev_out_size:-0}" -gt 0 && "$cur_out_size" -le "$prev_out_size" ]]; then
                out_idle_streak=$(( out_idle_streak + 2 ))
            else
                out_idle_streak=0
            fi
            if [[ "$out_idle_streak" -ge "$out_idle_limit" ]]; then
                echo "[$(date +%H:%M:%S)] WATCHDOG OUTPUT_IDLE_HANG idle=${out_idle_streak}s size=${cur_out_size}B" >> "$watchdog_log"
                kill -TERM "$fastp_pid" 2>/dev/null
                sleep 5
                kill -9   "$fastp_pid" 2>/dev/null
                echo "HANG_DETECTED output_idle=${out_idle_streak}s out_size=${cur_out_size}B" \
                    > /tmp/fastp_bench_watchdog_$$
                break
            fi
            prev_out_size="$cur_out_size"

            # Heartbeat every ~60 s into the watchdog log so we can see it's alive.
            if (( elapsed % 60 < 2 )); then
                echo "[$(date +%H:%M:%S)] WATCHDOG ALIVE elapsed=${elapsed}s cpu_idle=${idle_streak}s out_idle=${out_idle_streak}s out_size=${cur_out_size}B" >> "$watchdog_log"
            fi
        done
    ) &
    local watchdog_pid=$!

    wait "$fastp_pid" || exit_code=$?
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true

    # Summarize sampled resource usage
    local _r_summary cpu_avg cpu_peak rss_peak gpu_avg gpu_peak gpu_mem_peak
    _r_summary=$(summarize_resource_csv "$sample_csv")
    read -r cpu_avg cpu_peak rss_peak gpu_avg gpu_peak gpu_mem_peak <<< "$_r_summary"
    _RUN_CPU_AVG="$cpu_avg"
    _RUN_CPU_PEAK="$cpu_peak"
    _RUN_RSS_PEAK="$rss_peak"
    _RUN_GPU_AVG="$gpu_avg"
    _RUN_GPU_PEAK="$gpu_peak"
    _RUN_GPU_MEM_PEAK="$gpu_mem_peak"
    _RUN_SAMPLE_CSV="$sample_csv"

    t_end=$(date +%s%N)
    wall_s=$(awk "BEGIN{printf \"%.3f\",($t_end - $t_start)/1000000000}")

    # Check watchdog verdict
    local watchdog_msg=""
    if [[ -f /tmp/fastp_bench_watchdog_$$ ]]; then
        watchdog_msg=$(cat /tmp/fastp_bench_watchdog_$$)
        rm -f /tmp/fastp_bench_watchdog_$$
        exit_code=137   # treat as killed
    fi

    # exit_code 124 = timeout (SIGTERM); 137 = killed (SIGKILL / watchdog)
    if [[ $exit_code -eq 124 || $exit_code -eq 137 ]]; then
        local reason="TIMEOUT after ${wall_s}s (limit ${RUN_TIMEOUT}s)"
        [[ -n "$watchdog_msg" ]] && reason="HANG after ${wall_s}s ($watchdog_msg)"
        echo "  [KILLED] ${tool_label} ${ds} T=${threads} — ${reason}" | tee -a "$VALID_LOG"
        printf "  [HANG/TIMEOUT]\n"
        rm -f "$tmp_out1" "$tmp_stderr"
        [[ -n "$tmp_out2" ]] && rm -f "$tmp_out2"
        _RUN_WALL="$wall_s"; _RUN_RI=0; _RUN_RO=0; _RUN_PCT="TIMEOUT"; _RUN_JSON="$tmp_json"
        _RUN_EXIT="$exit_code"
        return 0
    fi

    local ri ro
    ri=$(reads_in  "$tmp_json")
    ro=$(reads_out "$tmp_json")

    # Detect non-watchdog crash (binary exited non-zero, or JSON missing /
    # empty, or reads_in==0 with the run actually attempted).  Mark CRASH
    # so the caller does NOT store this run as a usable baseline / md5
    # reference.  Without this, an opengene crash would record ri=ro=0
    # and every subsequent d0bromir validation would [FAIL] on read-count
    # mismatch and abort the entire benchmark.
    if [[ "$exit_code" -ne 0 || ! -s "$tmp_json" || "${ri:-0}" -le 0 ]]; then
        echo "  [CRASH] ${tool_label} ${ds} T=${threads} exit=${exit_code} ri=${ri} (json=$(stat -c %s "$tmp_json" 2>/dev/null || echo missing) bytes)" \
            | tee -a "$VALID_LOG"
        printf "  [CRASH]\n"
        rm -f "$tmp_out1" "$tmp_stderr"
        [[ -n "$tmp_out2" ]] && rm -f "$tmp_out2"
        _RUN_WALL="$wall_s"; _RUN_RI=0; _RUN_RO=0; _RUN_PCT="CRASH"
        _RUN_JSON="$tmp_json"; _RUN_OUT1=""; _RUN_OUT2=""
        _RUN_EXIT="$exit_code"
        return 0
    fi

    # % vs opengene.  Only compute when we have a positive, healthy
    # baseline wall-time; never compare against a stuck/timed-out opengene.
    local pct="baseline"
    if [[ "$tool_label" != "opengene" ]]; then
        local og_key="${ds}:opengene:${threads}"
        local og_w="${WALL_TIMES[$og_key]:-}"
        if [[ -n "$og_w" ]] && awk -v w="$og_w" 'BEGIN{exit !(w+0 > 0)}'; then
            pct=$(python3 -c "og=${og_w}; me=$wall_s; print(f'{(me-og)/og*100:+.1f}%')")
        else
            pct="N/A"
        fi
    fi

    # Write profile log entry
    {
        echo ""
        echo "================================================================"
        printf " Application  : %s (%s)\n" "$tool_label" "$bin"
        printf " Dataset      : %s\n" "$label"
        printf " Files        : R1=%s\n" "$r1"
        [[ -n "$r2" ]] && printf "               R2=%s\n" "$r2"
        printf " Type         : %s\n" "$type"
        printf " Threads      : %s\n" "$threads"
        printf " Wall time    : %s s\n" "$wall_s"
        printf " reads_in     : %s\n" "$ri"
        printf " reads_out    : %s\n" "$ro"
        printf " vs_opengene  : %s\n" "$pct"
        echo "--- Resource usage ---"
        printf " CPU avg/peak : %s%% / %s%%\n" "$cpu_avg" "$cpu_peak"
        printf " RSS peak     : %s MiB\n" "$rss_peak"
        printf " GPU util a/p : %s%% / %s%%\n" "$gpu_avg" "$gpu_peak"
        printf " GPU mem peak : %s MiB\n" "$gpu_mem_peak"
        printf " Sample CSV   : %s\n" "$sample_csv"
        echo "--- Profiling output ---"
        cat "$tmp_stderr"
        echo "================================================================"
    } >> "$PROFILE_LOG"

    # Only record this run's wall-time as a usable baseline if the run
    # actually succeeded.  A non-zero exit, missing JSON, or zero reads_in
    # means the timing is junk and must not be used for % comparisons.
    if [[ "$exit_code" -eq 0 && -s "$tmp_json" && "${ri:-0}" -gt 0 ]]; then
        WALL_TIMES["${ds}:${tool_label}:${threads}"]="$wall_s"
    fi

    # Caller is responsible for deleting output FASTQ + JSON (after deep
    # validation if needed).  Stderr is discarded immediately.
    rm -f "$tmp_stderr"

    _RUN_WALL="$wall_s"
    _RUN_RI="$ri"
    _RUN_RO="$ro"
    _RUN_PCT="$pct"
    _RUN_JSON="$tmp_json"
    _RUN_OUT1="$tmp_out1"
    _RUN_OUT2="$tmp_out2"
    # Preserve the real exit code of the fastp subprocess so run_with_retries
    # can detect non-watchdog crashes.
    _RUN_EXIT="$exit_code"
}

# ---------------------------------------------------------------------------
# free_run_outputs — delete files left by run_one for the most recent run.
# ---------------------------------------------------------------------------
free_run_outputs() {
    rm -f "${_RUN_JSON:-}" "${_RUN_OUT1:-}" "${_RUN_OUT2:-}"
    _RUN_JSON=""; _RUN_OUT1=""; _RUN_OUT2=""
}

# ---------------------------------------------------------------------------
# run_with_retries — call run_one up to MAX_ATTEMPTS times.  A run is
# considered failed (and retried) if (a) the binary exited non-zero with a
# non-watchdog code, or (b) the JSON output is missing/unreadable.
# Watchdog kills (TIMEOUT/HANG) are NOT retried — they would just hang again.
# Sets _RUN_ATTEMPTS to the attempt number that succeeded (or MAX_ATTEMPTS
# on persistent failure).
# ---------------------------------------------------------------------------
run_with_retries() {
    local tool_label="$1" bin="$2" ds="$3" threads="$4" rep="$5"
    local attempt
    _RUN_ATTEMPTS=0
    for (( attempt = 1; attempt <= MAX_ATTEMPTS; attempt++ )); do
        _RUN_ATTEMPTS="$attempt"
        run_one "$tool_label" "$bin" "$ds" "$threads" "$rep"
        # TIMEOUT / HANG: do not retry
        if [[ "${_RUN_PCT:-}" == "TIMEOUT" ]]; then
            return 0
        fi
        # Crash detection: non-zero exit OR JSON missing/bad
        local crashed=0
        if [[ "${_RUN_EXIT:-0}" != "0" ]]; then crashed=1; fi
        if [[ -z "${_RUN_JSON:-}" || ! -s "$_RUN_JSON" ]]; then crashed=1; fi
        if [[ "$crashed" -eq 0 ]]; then
            return 0
        fi
        echo "  [RETRY] attempt $attempt/$MAX_ATTEMPTS failed (exit=${_RUN_EXIT:-?}) ds=$ds tool=$tool_label T=$threads" \
            | tee -a "$VALID_LOG"
        free_run_outputs
        sleep 5
        drop_cache
    done
    # All attempts crashed
    _RUN_PCT="CRASH"
    return 0
}

# ---------------------------------------------------------------------------
# validate_against_opengene
# ---------------------------------------------------------------------------
validate_against_opengene() {
    local tool_label="$1" ds="$2" threads="$3" test_json="$4"

    case "${_RUN_PCT:-}" in
        TIMEOUT|CRASH) _VALID="${_RUN_PCT}"; return 0 ;;
    esac

    local og_ri og_ro
    og_ri="${OG_READS_IN[${ds}:${threads}]:-}"
    og_ro="${OG_READS_OUT[${ds}:${threads}]:-}"

    if [[ -z "$og_ri" || -z "$og_ro" ]]; then
        echo "[SKIP] $tool_label $ds T=$threads — no opengene baseline stored" | tee -a "$VALID_LOG"
        _VALID="SKIP"; return 0
    fi

    local te_ri te_ro
    te_ri=$(reads_in  "$test_json")
    te_ro=$(reads_out "$test_json")

    if [[ "$og_ri" == "$te_ri" && "$og_ro" == "$te_ro" ]]; then
        echo "[PASS] $tool_label $ds T=$threads  reads_in=$te_ri  reads_out=$te_ro" | tee -a "$VALID_LOG"
        _VALID="PASS"
    else
        echo "[FAIL] $tool_label $ds T=$threads  reads_in(test=$te_ri og=$og_ri)  reads_out(test=$te_ro og=$og_ro)" | tee -a "$VALID_LOG"
        echo "BENCHMARK ABORTED — read-count validation failure" | tee -a "$VALID_LOG"
        _VALID="FAIL"
        kill "$TICKER_PID" 2>/dev/null
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# validate_full_against_opengene — deep correctness check at VALIDATE_THREADS.
# Compares md5(decompressed FASTQ) and a fixed set of JSON biology fields.
# Hard-fails (exit 2) on any mismatch; output files retained on FAIL.
# Sets _VALID="PASS+FQ" on success.
# ---------------------------------------------------------------------------
validate_full_against_opengene() {
    local tool_label="$1" ds="$2" threads="$3"
    local test_json="$4" test_fq1="$5" test_fq2="${6:-}"

    case "${_RUN_PCT:-}" in
        TIMEOUT|CRASH) _VALID="${_RUN_PCT}"; return 0 ;;
    esac

    local og_json="${OG_JSON[$ds]:-}"
    local og_md5_1="${OG_FQ1_MD5[$ds]:-}"
    local og_md5_2="${OG_FQ2_MD5[$ds]:-}"

    if [[ -z "$og_json" || -z "$og_md5_1" ]]; then
        echo "[WARN] $tool_label $ds T=$threads — no opengene reference; falling back to read-count check" | tee -a "$VALID_LOG"
        validate_against_opengene "$tool_label" "$ds" "$threads" "$test_json"
        return 0
    fi

    local te_md5_1 te_md5_2=""
    te_md5_1=$(fq_md5 "$test_fq1")
    [[ -n "$test_fq2" ]] && te_md5_2=$(fq_md5 "$test_fq2")

    if [[ "$te_md5_1" != "$og_md5_1" ]]; then
        echo "[FAIL] $tool_label $ds T=$threads  R1 FASTQ md5 mismatch (test=$te_md5_1 og=$og_md5_1)" | tee -a "$VALID_LOG"
        echo "  test R1 retained at: $test_fq1" | tee -a "$VALID_LOG"
        echo "BENCHMARK ABORTED — FASTQ correctness failure" | tee -a "$VALID_LOG"
        _VALID="FAIL"
        kill "$TICKER_PID" 2>/dev/null
        exit 2
    fi
    if [[ -n "$test_fq2" && "$te_md5_2" != "$og_md5_2" ]]; then
        echo "[FAIL] $tool_label $ds T=$threads  R2 FASTQ md5 mismatch (test=$te_md5_2 og=$og_md5_2)" | tee -a "$VALID_LOG"
        echo "  test R2 retained at: $test_fq2" | tee -a "$VALID_LOG"
        echo "BENCHMARK ABORTED — FASTQ correctness failure" | tee -a "$VALID_LOG"
        _VALID="FAIL"
        kill "$TICKER_PID" 2>/dev/null
        exit 2
    fi

    local mismatches=""
    for field in "${VALIDATE_JSON_FIELDS[@]}"; do
        local og_v te_v
        og_v=$(json_field_get "$og_json"   "$field")
        te_v=$(json_field_get "$test_json" "$field")
        [[ -z "$og_v" && -z "$te_v" ]] && continue
        if [[ "$og_v" != "$te_v" ]]; then
            mismatches+=$'\n'"    $field: test=$te_v  og=$og_v"
        fi
    done
    if [[ -n "$mismatches" ]]; then
        echo "[FAIL] $tool_label $ds T=$threads  JSON biology fields differ:$mismatches" | tee -a "$VALID_LOG"
        echo "  test JSON retained at: $test_json" | tee -a "$VALID_LOG"
        echo "BENCHMARK ABORTED — JSON correctness failure" | tee -a "$VALID_LOG"
        _VALID="FAIL"
        kill "$TICKER_PID" 2>/dev/null
        exit 2
    fi

    echo "[PASS+FQ] $tool_label $ds T=$threads  md5(R1)=$te_md5_1${test_fq2:+  md5(R2)=$te_md5_2}  + ${#VALIDATE_JSON_FIELDS[@]} JSON fields equal" | tee -a "$VALID_LOG"
    _VALID="PASS+FQ"
}

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=== Dataset: $ds (${DS_TYPE[$ds]}) — ${DS_LABEL[$ds]} ==="

    # Skip PE-only scenarios for SE datasets.
    if [[ "${DS_TYPE[$ds]}" == "SE" && -n "${SCENARIO_PE_ONLY[$SCENARIO]:-}" ]]; then
        echo "  [SKIP] scenario '$SCENARIO' is PE-only; skipping SE dataset $ds"
        for threads in "${THREAD_COUNTS[@]}"; do
            echo "${ds},SE,opengene,${threads},1,0,0,0,baseline,SKIP,0,0,0,0,0,0,0,${SCENARIO}" >> "$CSV"
            echo "${ds},SE,d0bromir_cpu,${threads},1,0,0,0,N/A,SKIP,0,0,0,0,0,0,0,${SCENARIO}" >> "$CSV"
            echo "${ds},SE,d0bromir_gpu,${threads},1,0,0,0,N/A,SKIP,0,0,0,0,0,0,0,${SCENARIO}" >> "$CSV"
            RUN_NUM=$(( RUN_NUM + 3 ))
        done
        continue
    fi

    for threads in "${THREAD_COUNTS[@]}"; do
        echo "--- Threads: $threads ---"

        # Whether deep validation runs at this thread count.
        is_validate_t=0
        [[ "$threads" == "$VALIDATE_THREADS" && -z "$OG_CSV" ]] && is_validate_t=1

        # opengene
        if [[ -z "$OG_CSV" ]]; then
            RUN_NUM=$(( RUN_NUM + 1 ))
            if [[ -n "${COMPLETED[${ds}:opengene:${threads}]:-}" ]]; then
                printf "  [%2d/%2d] %-15s %-22s T=%-2s  [resumed]\n" "$RUN_NUM" "$TOTAL_RUNS" "opengene" "$ds" "$threads"
                # Re-load OG read counts + wall from the resume CSV so
                # subsequent d0bromir validation works.
                # Pick the newest non-TIMEOUT/FAIL/CRASH opengene row with
                # a positive wall-time so % comparisons never use a stuck
                # baseline.  Scan all matches, keep the last good one.
                _og_row=$(awk -F, -v d="$ds" -v t="$threads" '
                    $1==d && $3=="opengene" && $4==t {
                        bad=0
                        for (i=9; i<=NF; i++) {
                            if ($i ~ /TIMEOUT|FAIL|CRASH/) { bad=1; break }
                        }
                        if (bad) next
                        if ($6+0 <= 0) next
                        last=$0
                    }
                    END { if (last) print last }' "$RESUME_CSV")
                if [[ -n "$_og_row" ]]; then
                    IFS=, read -r _ _ _ _ _ _w _ri _ro _ <<< "$_og_row"
                    WALL_TIMES["${ds}:opengene:${threads}"]="$_w"
                    OG_READS_IN["${ds}:${threads}"]="$_ri"
                    OG_READS_OUT["${ds}:${threads}"]="$_ro"
                fi
                write_status
            else
                printf "  [%2d/%2d] %-15s %-22s T=%-2s  " "$RUN_NUM" "$TOTAL_RUNS" "opengene" "$ds" "$threads"
                run_one "opengene" "$OPENGENE" "$ds" "$threads" "1"
                # Only store baseline counts when opengene actually succeeded;
                # TIMEOUT / CRASH leave reads at 0 and would poison every
                # subsequent d0bromir validation.
                _og_ok=0
                case "${_RUN_PCT:-}" in
                    TIMEOUT|CRASH) _og_ok=0 ;;
                    *) _og_ok=1 ;;
                esac
                if [[ "$_og_ok" == "1" ]]; then
                    OG_READS_IN["${ds}:${threads}"]="$_RUN_RI"
                    OG_READS_OUT["${ds}:${threads}"]="$_RUN_RO"
                fi
                # At validation thread count: retain JSON, compute FQ md5s,
                # then drop FASTQ.  Reference JSON is freed end-of-dataset.
                if [[ "$is_validate_t" == "1" && "$_og_ok" == "1" ]]; then
                    OG_JSON["$ds"]="$_RUN_JSON"
                    OG_FQ1_MD5["$ds"]=$(fq_md5 "$_RUN_OUT1")
                    [[ -n "$_RUN_OUT2" ]] && OG_FQ2_MD5["$ds"]=$(fq_md5 "$_RUN_OUT2")
                    rm -f "$_RUN_OUT1" "$_RUN_OUT2"
                    _RUN_OUT1=""; _RUN_OUT2=""; _RUN_JSON=""
                else
                    free_run_outputs
                fi
                echo "${_RUN_WALL}s  reads_in=${_RUN_RI}  reads_out=${_RUN_RO}"
                # opengene baseline row: mark column 9 as N/A when the run
                # was killed/timed-out so downstream readers do not treat
                # the truncated wall-time as a usable baseline.
                _og_pct_col="baseline"
                case "${_RUN_PCT:-OK}" in
                    TIMEOUT|CRASH|FAIL*) _og_pct_col="N/A" ;;
                esac
                echo "${ds},${DS_TYPE[$ds]},opengene,${threads},1,${_RUN_WALL},${_RUN_RI},${_RUN_RO},${_og_pct_col},${_RUN_PCT:-OK},1,${_RUN_CPU_AVG:-0},${_RUN_CPU_PEAK:-0},${_RUN_RSS_PEAK:-0},${_RUN_GPU_AVG:-0},${_RUN_GPU_PEAK:-0},${_RUN_GPU_MEM_PEAK:-0},${SCENARIO}" >> "$CSV"
                write_status
            fi
        else
            _og_wall="${WALL_TIMES[${ds}:opengene:${threads}]:-N/A}"
            _og_ri="${OG_READS_IN[${ds}:${threads}]:-0}"
            _og_ro="${OG_READS_OUT[${ds}:${threads}]:-0}"
            printf "  [CSV] %-15s %-22s T=%-2s  " "opengene" "$ds" "$threads"
            echo "${_og_wall}s  reads_in=${_og_ri}  reads_out=${_og_ro}  (from CSV)"
            echo "${ds},${DS_TYPE[$ds]},opengene,${threads},csv,${_og_wall},${_og_ri},${_og_ro},baseline,CSV,0,0,0,0,0,0,0,${SCENARIO}" >> "$CSV"
            write_status
        fi

        # d0bromir CPU
        RUN_NUM=$(( RUN_NUM + 1 ))
        if [[ -n "${COMPLETED[${ds}:d0bromir_cpu:${threads}]:-}" ]]; then
            printf "  [%2d/%2d] %-15s %-22s T=%-2s  [resumed]\n" "$RUN_NUM" "$TOTAL_RUNS" "d0bromir_cpu" "$ds" "$threads"
            write_status
        else
            printf "  [%2d/%2d] %-15s %-22s T=%-2s  " "$RUN_NUM" "$TOTAL_RUNS" "d0bromir_cpu" "$ds" "$threads"
            run_with_retries "d0bromir_cpu" "$CPU_BIN" "$ds" "$threads" "1"
            if [[ "$is_validate_t" == "1" ]]; then
                validate_full_against_opengene "d0bromir_cpu" "$ds" "$threads" \
                    "$_RUN_JSON" "$_RUN_OUT1" "$_RUN_OUT2"
            else
                validate_against_opengene "d0bromir_cpu" "$ds" "$threads" "$_RUN_JSON"
            fi
            free_run_outputs
            echo "${_RUN_WALL}s  vs_opengene=${_RUN_PCT}  ${_VALID}  attempts=${_RUN_ATTEMPTS}"
            echo "${ds},${DS_TYPE[$ds]},d0bromir_cpu,${threads},1,${_RUN_WALL},${_RUN_RI},${_RUN_RO},${_RUN_PCT},${_VALID},${_RUN_ATTEMPTS},${_RUN_CPU_AVG:-0},${_RUN_CPU_PEAK:-0},${_RUN_RSS_PEAK:-0},${_RUN_GPU_AVG:-0},${_RUN_GPU_PEAK:-0},${_RUN_GPU_MEM_PEAK:-0},${SCENARIO}" >> "$CSV"
            write_status
        fi

        # d0bromir GPU
        RUN_NUM=$(( RUN_NUM + 1 ))
        if [[ -n "${COMPLETED[${ds}:d0bromir_gpu:${threads}]:-}" ]]; then
            printf "  [%2d/%2d] %-15s %-22s T=%-2s  [resumed]\n" "$RUN_NUM" "$TOTAL_RUNS" "d0bromir_gpu" "$ds" "$threads"
            write_status
        else
            printf "  [%2d/%2d] %-15s %-22s T=%-2s  " "$RUN_NUM" "$TOTAL_RUNS" "d0bromir_gpu" "$ds" "$threads"
            run_with_retries "d0bromir_gpu" "$GPU_BIN" "$ds" "$threads" "1"
            if [[ "$is_validate_t" == "1" ]]; then
                validate_full_against_opengene "d0bromir_gpu" "$ds" "$threads" \
                    "$_RUN_JSON" "$_RUN_OUT1" "$_RUN_OUT2"
            else
                validate_against_opengene "d0bromir_gpu" "$ds" "$threads" "$_RUN_JSON"
            fi
            free_run_outputs
            echo "${_RUN_WALL}s  vs_opengene=${_RUN_PCT}  ${_VALID}  attempts=${_RUN_ATTEMPTS}"
            echo "${ds},${DS_TYPE[$ds]},d0bromir_gpu,${threads},1,${_RUN_WALL},${_RUN_RI},${_RUN_RO},${_RUN_PCT},${_VALID},${_RUN_ATTEMPTS},${_RUN_CPU_AVG:-0},${_RUN_CPU_PEAK:-0},${_RUN_RSS_PEAK:-0},${_RUN_GPU_AVG:-0},${_RUN_GPU_PEAK:-0},${_RUN_GPU_MEM_PEAK:-0},${SCENARIO}" >> "$CSV"
            write_status
        fi
    done

    # End-of-dataset: free retained opengene reference JSON.
    if [[ -n "${OG_JSON[$ds]:-}" ]]; then
        rm -f "${OG_JSON[$ds]}"
        unset 'OG_JSON[$ds]'
    fi
done

kill "$TICKER_PID" 2>/dev/null || true
trap - EXIT

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " BENCHMARK COMPLETE — $(date)"
echo " CSV       : $CSV"
echo " Profile   : $PROFILE_LOG"
echo " Validation: $VALID_LOG"
echo " Status    : $STATUS_FILE"
echo "============================================================"
echo ""
printf "%-22s  %-4s  %-15s  %5s  %10s  %12s  %12s  %12s  %s\n" \
    "dataset" "type" "tool" "T" "wall_s" "reads_in" "reads_out" "vs_opengene" "valid"
printf "%s\n" "$(printf -- '-%.0s' {1..105})"
tail -n +2 "$CSV" | while IFS=, read -r ds tp tool thr rep wt ri ro pct vl _rest; do
    printf "%-22s  %-4s  %-15s  %5s  %10.1f  %12s  %12s  %12s  %s\n" \
        "$ds" "$tp" "$tool" "$thr" "$wt" "$ri" "$ro" "$pct" "$vl"
done
write_status
echo ""
echo "Done."
