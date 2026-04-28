#!/usr/bin/env bash
# =============================================================================
# run_benchmark_wsl.sh — Full three-tool benchmark (WSL path: /mnt/c/bio/FASTQ/WGS/) with profiling + validation
#
# Tools   : opengene fastp 1.1.0
#           fastp-d0bromir CPU (fastp-cpu-profile)
#           fastp-d0bromir GPU (fastp)
#
# Datasets: Panel SE 148M  (S1A_S1_L001_R1)
#           WGS SE 6.3G    (ERR1044780 R1 only)
#           WGS PE 12.8G   (ERR1044780 R1+R2)
#           WGS PE 18.2G   (ERR1044319 R1+R2)
#           WGS PE 40G     (DRR216653 R1+R2)
#           WGS PE 18G     (ERR12107878 NovaSeq6000 2023, /tmp/wgs/)
#           WGS PE 34G     (DRR357122   HiSeqXTen   2023, /tmp/wgs/)
#           WGS PE 40G     (DRR635848   NovaSeqX    2025, /tmp/wgs/)
#
# Thread counts: 1 2 4 8 16  (default; override with -t)
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
#   Panel_SE_148M   WGS_SE_6.3G   WGS_PE_12.8G   WGS_PE_18.2G   WGS_PE_40G
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

# ---------------------------------------------------------------------------
# All available datasets
# ---------------------------------------------------------------------------
declare -A DS_TYPE DS_R1 DS_R2 DS_LABEL

DS_TYPE[Panel_SE_148M]="SE"
DS_R1[Panel_SE_148M]="/mnt/c/bio/FASTQ/S1A_S1_L001_R1_001.fastq.gz"
DS_R2[Panel_SE_148M]=""
DS_LABEL[Panel_SE_148M]="Panel SE 148M (S1A_S1_L001_R1)"

DS_TYPE[Panel_PE_304M]="PE"
DS_R1[Panel_PE_304M]="/mnt/c/bio/FASTQ/S1A_S1_L001_R1_001.fastq.gz"
DS_R2[Panel_PE_304M]="/mnt/c/bio/FASTQ/S1A_S1_L001_R2_001.fastq.gz"
DS_LABEL[Panel_PE_304M]="Panel PE 304M (S1A_S1_L001 R1+R2)"

DS_TYPE[WGS_SE_6.3G]="SE"
DS_R1[WGS_SE_6.3G]="/mnt/c/bio/FASTQ/WGS/ERR1044780_1.fastq.gz"
DS_R2[WGS_SE_6.3G]=""
DS_LABEL[WGS_SE_6.3G]="WGS SE 6.3G (ERR1044780_1)"

DS_TYPE[WGS_PE_12.8G]="PE"
DS_R1[WGS_PE_12.8G]="/mnt/c/bio/FASTQ/WGS/ERR1044780_1.fastq.gz"
DS_R2[WGS_PE_12.8G]="/mnt/c/bio/FASTQ/WGS/ERR1044780_2.fastq.gz"
DS_LABEL[WGS_PE_12.8G]="WGS PE 12.8G (ERR1044780 R1+R2)"

DS_TYPE[WGS_PE_18.2G]="PE"
DS_R1[WGS_PE_18.2G]="/mnt/c/bio/FASTQ/WGS/ERR1044319_1.fastq.gz"
DS_R2[WGS_PE_18.2G]="/mnt/c/bio/FASTQ/WGS/ERR1044319_2.fastq.gz"
DS_LABEL[WGS_PE_18.2G]="WGS PE 18.2G (ERR1044319 R1+R2)"

DS_TYPE[WGS_PE_40G]="PE"
DS_R1[WGS_PE_40G]="/mnt/c/bio/FASTQ/WGS/DRR216653_1.fastq.gz"
DS_R2[WGS_PE_40G]="/mnt/c/bio/FASTQ/WGS/DRR216653_2.fastq.gz"
DS_LABEL[WGS_PE_40G]="WGS PE 40G (DRR216653 R1+R2)"

# New ENA samples downloaded to /tmp/wgs/
DS_TYPE[WGS_PE_18G_ERR12107878]="PE"
DS_R1[WGS_PE_18G_ERR12107878]="/tmp/wgs/ERR12107878_1.fastq.gz"
DS_R2[WGS_PE_18G_ERR12107878]="/tmp/wgs/ERR12107878_2.fastq.gz"
DS_LABEL[WGS_PE_18G_ERR12107878]="WGS PE 18G (ERR12107878 NovaSeq6000 2023)"

DS_TYPE[WGS_PE_34G_DRR357122]="PE"
DS_R1[WGS_PE_34G_DRR357122]="/tmp/wgs/DRR357122_1.fastq.gz"
DS_R2[WGS_PE_34G_DRR357122]="/tmp/wgs/DRR357122_2.fastq.gz"
DS_LABEL[WGS_PE_34G_DRR357122]="WGS PE 34G (DRR357122 HiSeqXTen 2023)"

DS_TYPE[WGS_PE_40G_DRR635848]="PE"
DS_R1[WGS_PE_40G_DRR635848]="/tmp/wgs/DRR635848_1.fastq.gz"
DS_R2[WGS_PE_40G_DRR635848]="/tmp/wgs/DRR635848_2.fastq.gz"
DS_LABEL[WGS_PE_40G_DRR635848]="WGS PE 40G (DRR635848 NovaSeqX 2025)"

ALL_DATASETS=(Panel_SE_148M Panel_PE_304M WGS_SE_6.3G WGS_PE_12.8G WGS_PE_18.2G WGS_PE_40G WGS_PE_18G_ERR12107878 WGS_PE_34G_DRR357122 WGS_PE_40G_DRR635848)
ALL_THREADS=(1 2 4 8 16)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SELECTED_DATASETS=()
SELECTED_THREADS=()

usage() {
    echo "Usage: $0 [-d DATASET]... [-t THREADS]... [--list]"
    echo ""
    echo "Options:"
    echo "  -d DATASET   Run only this dataset (repeatable)"
    echo "  -t THREADS   Run only this thread count (repeatable)"
    echo "  --list       List available dataset names and exit"
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

TOTAL_RUNS=$(( ${#DATASETS[@]} * 3 * ${#THREAD_COUNTS[@]} ))
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
echo "============================================================"
echo ""
echo "Binary check:"
for pair in "opengene:$OPENGENE" "d0bromir_cpu:$CPU_BIN" "d0bromir_gpu:$GPU_BIN"; do
    lbl="${pair%%:*}"; pth="${pair##*:}"
    if [[ ! -x "$pth" ]]; then echo "  [MISSING] $lbl -> $pth"; exit 1; fi
    echo "  [OK] $lbl  $(ls -lh "$pth" | awk '{print $5,$6,$7,$8}')  $pth"
done
echo ""
echo "Dataset check:"
for ds in "${DATASETS[@]}"; do
    if [[ ! -f "${DS_R1[$ds]}" ]]; then
        echo "  [WARN] ${DS_R1[$ds]} not yet present — will attempt when reached"
    elif [[ "${DS_TYPE[$ds]}" == "PE" && ! -f "${DS_R2[$ds]}" ]]; then
        echo "  [WARN] ${DS_R2[$ds]} not yet present — will attempt when reached"
    else
        sz=$(du -sh "${DS_R1[$ds]}" | cut -f1)
        echo "  [OK] $ds  ${DS_TYPE[$ds]}  R1=$(basename "${DS_R1[$ds]}") ($sz)"
    fi
done
echo ""
echo "Starting $TOTAL_RUNS runs (files marked [WARN] will be attempted when reached)..."
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

    local t_start t_end wall_s
    t_start=$(date +%s%N)
    "${cmd[@]}" > "$tmp_stderr" 2>&1 || true
    t_end=$(date +%s%N)
    wall_s=$(awk "BEGIN{printf \"%.3f\",($t_end - $t_start)/1000000000}")

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
    local og_path="${OG_JSON[${ds}:${threads}]:-}"

    if [[ -z "$og_path" || ! -f "$og_path" ]]; then
        echo "[SKIP] $tool_label $ds T=$threads — no opengene baseline stored" | tee -a "$VALID_LOG"
        _VALID="SKIP"; return 0
    fi

    local og_ri og_ro te_ri te_ro
    og_ri=$(reads_in  "$og_path")
    og_ro=$(reads_out "$og_path")
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
        RUN_NUM=$(( RUN_NUM + 1 ))
        printf "  [%2d/%2d] %-15s %-22s T=%-2s  " "$RUN_NUM" "$TOTAL_RUNS" "opengene" "$ds" "$threads"
        run_one "opengene" "$OPENGENE" "$ds" "$threads" "1"
        # Save opengene JSON as reference
        og_ref=$(mktemp /tmp/fastp_og_ref_XXXXXX.json)
        cp "$_RUN_JSON" "$og_ref"
        OG_JSON["${ds}:${threads}"]="$og_ref"
        rm -f "$_RUN_JSON"
        echo "${_RUN_WALL}s  reads_in=${_RUN_RI}  reads_out=${_RUN_RO}"
        echo "${ds},${DS_TYPE[$ds]},opengene,${threads},1,${_RUN_WALL},${_RUN_RI},${_RUN_RO},baseline,OK" >> "$CSV"
        write_status

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
