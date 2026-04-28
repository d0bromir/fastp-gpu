#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Full three-tool benchmark with profiling + validation
#
# Tools   : opengene fastp 1.3.3
#           fastp-d0bromir CPU (fastp-cpu-profile)
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
# Validation: after every d0bromir run, reads_in / reads_out are compared
#   against the matching opengene baseline JSON.  FAIL aborts benchmark.
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
# Output goes to benchmark_results/<TS>/ as a self-contained run folder ready
# to be moved into the fastp-gpu_<v>/vs_opengene_<v>/<platform>/ layout.
RESULTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$RESULTS_ROOT/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

CSV="$RESULTS_DIR/full_benchmark_${TIMESTAMP}.csv"
PROFILE_LOG="$RESULTS_DIR/full_profiling_${TIMESTAMP}.log"
VALID_LOG="$RESULTS_DIR/full_validation_${TIMESTAMP}.log"
STATUS_FILE="$RESULTS_DIR/full_status_${TIMESTAMP}.txt"

OPENGENE="$PROJ_DIR/../fastp_opengene/fastp"
CPU_BIN="$PROJ_DIR/fastp-cpu-profile"
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

# Set to 1 to skip read-count validation against opengene baseline.
NO_VALIDATE=${NO_VALIDATE:-0}

# ---------------------------------------------------------------------------
# All available datasets
# ---------------------------------------------------------------------------
declare -A DS_TYPE DS_R1 DS_R2 DS_LABEL

DS_TYPE[Panel_SE_148M]="SE"
DS_R1[Panel_SE_148M]="/home/mpiuser/FASTQ/S1A_S1_L001_R1_001.fastq.gz"
DS_R2[Panel_SE_148M]=""
DS_LABEL[Panel_SE_148M]="Panel SE 148M (S1A_S1_L001_R1)"

DS_TYPE[Panel_PE_304M]="PE"
DS_R1[Panel_PE_304M]="/home/mpiuser/FASTQ/S1A_S1_L001_R1_001.fastq.gz"
DS_R2[Panel_PE_304M]="/home/mpiuser/FASTQ/S1A_S1_L001_R2_001.fastq.gz"
DS_LABEL[Panel_PE_304M]="Panel PE 304M (S1A_S1_L001 R1+R2)"

DS_TYPE[WGS_SE_6.3G]="SE"
DS_R1[WGS_SE_6.3G]="/home/mpiuser/FASTQ/WGS/ERR1044780_1.fastq.gz"
DS_R2[WGS_SE_6.3G]=""
DS_LABEL[WGS_SE_6.3G]="WGS SE 6.3G (ERR1044780_1)"

DS_TYPE[WGS_PE_12.8G]="PE"
DS_R1[WGS_PE_12.8G]="/home/mpiuser/FASTQ/WGS/ERR1044780_1.fastq.gz"
DS_R2[WGS_PE_12.8G]="/home/mpiuser/FASTQ/WGS/ERR1044780_2.fastq.gz"
DS_LABEL[WGS_PE_12.8G]="WGS PE 12.8G (ERR1044780 R1+R2)"

DS_TYPE[WGS_PE_18.2G]="PE"
DS_R1[WGS_PE_18.2G]="/home/mpiuser/FASTQ/WGS/ERR1044319_1.fastq.gz"
DS_R2[WGS_PE_18.2G]="/home/mpiuser/FASTQ/WGS/ERR1044319_2.fastq.gz"
DS_LABEL[WGS_PE_18.2G]="WGS PE 18.2G (ERR1044319 R1+R2)"

DS_TYPE[WGS_PE_40G]="PE"
DS_R1[WGS_PE_40G]="/home/mpiuser/FASTQ/WGS/DRR216653_1.fastq.gz"
DS_R2[WGS_PE_40G]="/home/mpiuser/FASTQ/WGS/DRR216653_2.fastq.gz"
DS_LABEL[WGS_PE_40G]="WGS PE 40G (DRR216653 R1+R2)"

# Public panel alternatives for the private TST-15 panel (same technology:
# MiSeq PE amplicon; PDAC 4-gene hotspot panel KRAS/TP53/CDKN2A/SMAD4)
# Download with: ./benchmark_results/_tools/download_bench_samples.sh panel
DS_TYPE[Panel_PDAC_PE_1M]="PE"
DS_R1[Panel_PDAC_PE_1M]="/home/mpiuser/FASTQ/Panel_public/DRR262998_1.fastq.gz"
DS_R2[Panel_PDAC_PE_1M]="/home/mpiuser/FASTQ/Panel_public/DRR262998_2.fastq.gz"
DS_LABEL[Panel_PDAC_PE_1M]="Panel PDAC PE 166MB (DRR262998, 1.0M reads, MiSeq 156bp)"

DS_TYPE[Panel_PDAC_PE_1.5M]="PE"
DS_R1[Panel_PDAC_PE_1.5M]="/home/mpiuser/FASTQ/Panel_public/DRR263018_1.fastq.gz"
DS_R2[Panel_PDAC_PE_1.5M]="/home/mpiuser/FASTQ/Panel_public/DRR263018_2.fastq.gz"
DS_LABEL[Panel_PDAC_PE_1.5M]="Panel PDAC PE 249MB (DRR263018, 1.5M reads, MiSeq 156bp)"

ALL_DATASETS=(Panel_SE_148M Panel_PE_304M Panel_PDAC_PE_1M Panel_PDAC_PE_1.5M WGS_SE_6.3G WGS_PE_12.8G WGS_PE_18.2G WGS_PE_40G)
ALL_THREADS=(1 2 4 8 16 32)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SELECTED_DATASETS=()
SELECTED_THREADS=()

usage() {
    echo "Usage: $0 [-d DATASET]... [-t THREADS]... [--list] [--use-og-csv [FILE]] [--no-validate]"
    echo ""
    echo "Options:"
    echo "  -d DATASET   Run only this dataset (repeatable)"
    echo "  -t THREADS   Run only this thread count (repeatable)"
    echo "  --list       List available dataset names and exit"
    echo "  --use-og-csv [FILE]  Skip opengene runs; load baseline from FILE"
    echo "               (default: canonical 20260417 dataset)"
    echo "  --no-validate        Skip read-count validation (useful when comparing"
    echo "               different fastp versions with known output differences)"
    echo "  -h, --help   Show this help"
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
        --no-validate)
            NO_VALIDATE=1
            shift
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

# Stores wall times for % calculation:  key = "ds:tool:threads"
declare -A WALL_TIMES
# Stores opengene reference JSON paths:  key = "ds:threads"
declare -A OG_JSON
# Stores opengene read counts for validation:  key = "ds:threads"
declare -A OG_READS_IN OG_READS_OUT

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
# CSV header
# ---------------------------------------------------------------------------
echo "dataset,type,tool,threads,rep,walltime_s,reads_in,reads_out,pct_vs_opengene,validation" > "$CSV"
printf "=== Full Benchmark Log: %s ===\n\n" "$(date)" > "$PROFILE_LOG"
printf "=== Validation Log: %s ===\n\n"    "$(date)" > "$VALID_LOG"

if [[ -n "$OG_CSV" ]]; then
    TOTAL_RUNS=$(( ${#DATASETS[@]} * 2 * ${#THREAD_COUNTS[@]} ))
else
    TOTAL_RUNS=$(( ${#DATASETS[@]} * 3 * ${#THREAD_COUNTS[@]} ))
fi
RUN_NUM=0

# ---------------------------------------------------------------------------
# write_status — builds the human-readable STATUS_FILE
# ---------------------------------------------------------------------------
write_status() {
    local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
    {
        printf "============================================================\n"
        printf " BENCHMARK STATUS — %s\n" "$ts"
        printf " Progress : %d / %d runs\n" "$RUN_NUM" "$TOTAL_RUNS"
        printf " Datasets : %s\n" "${DATASETS[*]}"
        printf " Threads  : %s\n" "${THREAD_COUNTS[*]}"
        printf " CSV      : %s\n" "$CSV"
        printf "============================================================\n\n"

        if [[ -s "$CSV" ]] && [[ $(wc -l < "$CSV") -gt 1 ]]; then
            printf "%-22s  %-4s  %-15s  %5s  %10s  %12s  %12s  %12s  %s\n" \
                "dataset" "type" "tool" "T" "wall_s" "reads_in" "reads_out" "vs_opengene" "valid"
            printf -- "%-22s  %-4s  %-15s  %5s  %10s  %12s  %12s  %12s  %s\n" \
                "----------------------" "----" "---------------" "-----" "----------" "------------" "------------" "------------" "------"
            tail -n +2 "$CSV" | while IFS=, read -r ds tp tool thr rep wt ri ro pct vl; do
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
    while true; do
        sleep 180
        write_status
    done
}
ticker_fn &
TICKER_PID=$!
trap "kill \$TICKER_PID 2>/dev/null; rm -f /tmp/fastp_bench_*; echo 'Benchmark interrupted.'" EXIT

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
        local wall ri ro
        if [[ "$has_rep" -eq 1 ]]; then
            # c5=rep, c6=walltime_s, c7=reads_in, rest starts with reads_out
            wall="$c6"; ri="$c7"; ro="${rest%%,*}"
        else
            # c5=walltime_s, c6=reads_in, c7=reads_out
            wall="$c5"; ri="$c6"; ro="$c7"
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
echo " Total runs: $TOTAL_RUNS"
echo " CSV       : $CSV"
echo " Profile   : $PROFILE_LOG"
echo " Validation: $VALID_LOG"
echo " Status    : $STATUS_FILE"
[[ -n "$OG_CSV" ]] && echo " OG CSV    : $OG_CSV  (opengene runs skipped)"
echo "============================================================"
echo ""
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
echo "All checks passed. Starting $TOTAL_RUNS runs..."
echo ""

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

    # Drop page cache before each run
    drop_cache

    local t_start t_end wall_s exit_code=0
    local t_start_sec; t_start_sec=$(date +%s)
    t_start=$(date +%s%N)

    # Launch fastp in background so the watchdog can monitor it
    "${cmd[@]}" > "$tmp_stderr" 2>&1 &
    local fastp_pid=$!

    # ---------------------------------------------------------------------------
    # Watchdog: kill if (a) wall-time hard limit exceeded, or
    #           (b) process CPU usage stays below HANG_CPU_PCT% for
    #               HANG_IDLE_SECS consecutive seconds after the startup grace
    #               period (HANG_GRACE_SECS).
    #
    # CPU% is measured as (delta jiffies / delta time) × 100 across all
    # threads of the process group, read from /proc/<pid>/stat field 14+15.
    # ---------------------------------------------------------------------------
    local hang_killed=0
    (
        set +e   # watchdog must not abort on non-zero sub-commands
        local grace=${HANG_GRACE_SECS:-20}
        local idle_limit=${HANG_IDLE_SECS:-30}
        local cpu_thresh=${HANG_CPU_PCT:-5}
        local hard_limit=${RUN_TIMEOUT:-1800}
        local hz; hz=$(getconf CLK_TCK 2>/dev/null || echo 100)

        local elapsed=0
        local idle_streak=0
        local prev_jiffies=0
        local prev_ts=0

        while kill -0 "$fastp_pid" 2>/dev/null; do
            sleep 2
            elapsed=$(( $(date +%s) - t_start_sec ))

            # Hard wall-time limit
            if [[ "$hard_limit" -gt 0 && "$elapsed" -ge "$hard_limit" ]]; then
                kill -TERM "$fastp_pid" 2>/dev/null
                sleep 5
                kill -9   "$fastp_pid" 2>/dev/null
                echo "HARD_TIMEOUT" > /tmp/fastp_bench_watchdog_$$
                break
            fi

            # CPU hang detection (skip during grace period)
            if [[ "$elapsed" -lt "$grace" ]]; then
                idle_streak=0
                prev_jiffies=0
                prev_ts=0
                continue
            fi

            # Sum utime+stime+cutime+cstime from /proc/<pid>/stat
            local stat_line
            stat_line=$(cat "/proc/${fastp_pid}/stat" 2>/dev/null) || break
            local cur_jiffies cur_ts
            cur_jiffies=$(awk '{print $14+$15+$16+$17}' <<< "$stat_line")
            cur_ts=$(date +%s%N)

            if [[ "$prev_ts" -gt 0 ]]; then
                local delta_j=$(( cur_jiffies - prev_jiffies ))
                local delta_ns=$(( cur_ts - prev_ts ))
                # cpu% = (delta_jiffies / hz) / (delta_ns/1e9) * 100
                local cpu_pct
                cpu_pct=$(awk "BEGIN{printf \"%d\", ($delta_j / $hz) / ($delta_ns / 1000000000) * 100}")
                if [[ "$cpu_pct" -lt "$cpu_thresh" ]]; then
                    idle_streak=$(( idle_streak + 2 ))
                else
                    idle_streak=0
                fi
                if [[ "$idle_streak" -ge "$idle_limit" ]]; then
                    kill -TERM "$fastp_pid" 2>/dev/null
                    sleep 5
                    kill -9   "$fastp_pid" 2>/dev/null
                    echo "HANG_DETECTED cpu=${cpu_pct}% idle=${idle_streak}s" > /tmp/fastp_bench_watchdog_$$
                    break
                fi
            fi
            prev_jiffies="$cur_jiffies"
            prev_ts="$cur_ts"
        done
    ) &
    local watchdog_pid=$!

    wait "$fastp_pid" || exit_code=$?
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

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
        return 0
    fi

    local ri ro
    ri=$(reads_in  "$tmp_json")
    ro=$(reads_out "$tmp_json")

    # % vs opengene
    local pct="baseline"
    if [[ "$tool_label" != "opengene" ]]; then
        local og_key="${ds}:opengene:${threads}"
        if [[ -n "${WALL_TIMES[$og_key]:-}" ]]; then
            pct=$(python3 -c "og=${WALL_TIMES[$og_key]}; me=$wall_s; print(f'{(me-og)/og*100:+.1f}%')")
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
        echo "--- Profiling output ---"
        cat "$tmp_stderr"
        echo "================================================================"
    } >> "$PROFILE_LOG"

    WALL_TIMES["${ds}:${tool_label}:${threads}"]="$wall_s"

    # Cleanup output files (keep json for validation)
    rm -f "$tmp_out1" "$tmp_stderr"
    [[ -n "$tmp_out2" ]] && rm -f "$tmp_out2"

    _RUN_WALL="$wall_s"
    _RUN_RI="$ri"
    _RUN_RO="$ro"
    _RUN_PCT="$pct"
    _RUN_JSON="$tmp_json"
}

# ---------------------------------------------------------------------------
# validate_against_opengene
# ---------------------------------------------------------------------------
validate_against_opengene() {
    local tool_label="$1" ds="$2" threads="$3" test_json="$4"

    # Skip validation for timed-out runs
    if [[ "${_RUN_PCT:-}" == "TIMEOUT" ]]; then
        _VALID="TIMEOUT"; return 0
    fi

    # Skip validation when --no-validate is set
    if [[ "${NO_VALIDATE:-0}" == "1" ]]; then
        _VALID="SKIP"; return 0
    fi

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
        echo "BENCHMARK ABORTED — validation failure" | tee -a "$VALID_LOG"
        _VALID="FAIL"
        kill "$TICKER_PID" 2>/dev/null
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=== Dataset: $ds (${DS_TYPE[$ds]}) — ${DS_LABEL[$ds]} ==="

    for threads in "${THREAD_COUNTS[@]}"; do
        echo "--- Threads: $threads ---"

        # opengene
        if [[ -z "$OG_CSV" ]]; then
            RUN_NUM=$(( RUN_NUM + 1 ))
            printf "  [%2d/%2d] %-15s %-22s T=%-2s  " "$RUN_NUM" "$TOTAL_RUNS" "opengene" "$ds" "$threads"
            run_one "opengene" "$OPENGENE" "$ds" "$threads" "1"
            # Only store baseline if opengene completed successfully
            if [[ "${_RUN_PCT:-}" != "TIMEOUT" ]]; then
                OG_READS_IN["${ds}:${threads}"]="$_RUN_RI"
                OG_READS_OUT["${ds}:${threads}"]="$_RUN_RO"
            fi
            rm -f "$_RUN_JSON"
            echo "${_RUN_WALL}s  reads_in=${_RUN_RI}  reads_out=${_RUN_RO}"
            echo "${ds},${DS_TYPE[$ds]},opengene,${threads},1,${_RUN_WALL},${_RUN_RI},${_RUN_RO},baseline,${_RUN_PCT:-OK}" >> "$CSV"
            write_status
        else
            _og_wall="${WALL_TIMES[${ds}:opengene:${threads}]:-N/A}"
            _og_ri="${OG_READS_IN[${ds}:${threads}]:-0}"
            _og_ro="${OG_READS_OUT[${ds}:${threads}]:-0}"
            printf "  [CSV] %-15s %-22s T=%-2s  " "opengene" "$ds" "$threads"
            echo "${_og_wall}s  reads_in=${_og_ri}  reads_out=${_og_ro}  (from CSV)"
            echo "${ds},${DS_TYPE[$ds]},opengene,${threads},csv,${_og_wall},${_og_ri},${_og_ro},baseline,CSV" >> "$CSV"
            write_status
        fi

        # d0bromir CPU
        RUN_NUM=$(( RUN_NUM + 1 ))
        printf "  [%2d/%2d] %-15s %-22s T=%-2s  " "$RUN_NUM" "$TOTAL_RUNS" "d0bromir_cpu" "$ds" "$threads"
        run_one "d0bromir_cpu" "$CPU_BIN" "$ds" "$threads" "1"
        validate_against_opengene "d0bromir_cpu" "$ds" "$threads" "$_RUN_JSON"
        rm -f "$_RUN_JSON"
        echo "${_RUN_WALL}s  vs_opengene=${_RUN_PCT}  ${_VALID}"
        echo "${ds},${DS_TYPE[$ds]},d0bromir_cpu,${threads},1,${_RUN_WALL},${_RUN_RI},${_RUN_RO},${_RUN_PCT},${_VALID}" >> "$CSV"
        write_status

        # d0bromir GPU
        RUN_NUM=$(( RUN_NUM + 1 ))
        printf "  [%2d/%2d] %-15s %-22s T=%-2s  " "$RUN_NUM" "$TOTAL_RUNS" "d0bromir_gpu" "$ds" "$threads"
        run_one "d0bromir_gpu" "$GPU_BIN" "$ds" "$threads" "1"
        validate_against_opengene "d0bromir_gpu" "$ds" "$threads" "$_RUN_JSON"
        rm -f "$_RUN_JSON"
        echo "${_RUN_WALL}s  vs_opengene=${_RUN_PCT}  ${_VALID}"
        echo "${ds},${DS_TYPE[$ds]},d0bromir_gpu,${threads},1,${_RUN_WALL},${_RUN_RI},${_RUN_RO},${_RUN_PCT},${_VALID}" >> "$CSV"
        write_status
    done
done

# Cleanup reference JSONs
for key in "${!OG_JSON[@]}"; do rm -f "${OG_JSON[$key]}" 2>/dev/null; done

kill "$TICKER_PID" 2>/dev/null
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
tail -n +2 "$CSV" | while IFS=, read -r ds tp tool thr rep wt ri ro pct vl; do
    printf "%-22s  %-4s  %-15s  %5s  %10.1f  %12s  %12s  %12s  %s\n" \
        "$ds" "$tp" "$tool" "$thr" "$wt" "$ri" "$ro" "$pct" "$vl"
done
write_status
echo ""
echo "Done."
