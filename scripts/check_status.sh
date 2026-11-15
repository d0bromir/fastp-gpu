#!/usr/bin/env bash
# =============================================================================
# check_status.sh — Report on running download and benchmark jobs.
#
# Detects:
#   - download_bench_samples.sh runs (by process command line)
#   - run_benchmark.sh runs (by process command line) and reports CSV progress
#   - In-flight FASTQ downloads (largest growing file under FASTQ_DIR)
#   - Disk space, GPU utilization
#
# Usage:
#   ./scripts/check_status.sh           # one-shot snapshot
#   ./scripts/check_status.sh -w        # watch mode: refresh every 30s
#   ./scripts/check_status.sh -w -n 60  # watch mode, custom interval
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$PROJ_DIR/benchmark_results"
FASTQ_DIR="${FASTQ_DIR:-$HOME/FASTQ}"

WATCH=0
INTERVAL=30
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--watch) WATCH=1; shift ;;
        -n) shift; INTERVAL="$1"; shift ;;
        -h|--help)
            sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ----------------------------- helpers ---------------------------------------
hr() { printf '%.0s─' {1..78}; echo; }

human_size() {
    # bytes → human readable
    awk -v b="$1" 'BEGIN{
        s="BKMGTP"; i=1;
        while (b>=1024 && i<6) { b/=1024; i++ }
        printf "%.1f%s", b, substr(s,i,1)
    }'
}

human_dur() {
    # seconds → "Hh Mm Ss"
    local s="$1"
    printf "%dh %02dm %02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# Find PIDs for a script by matching its basename in /proc/*/cmdline
find_script_pids() {
    local script_basename="$1"
    pgrep -af "bash.*${script_basename}" 2>/dev/null \
        | awk '{print $1}' | sort -u
}

# ----------------------------- report sections -------------------------------
report_header() {
    hr
    echo " fastp benchmark + download status — $(date '+%Y-%m-%d %H:%M:%S')"
    hr
}

report_processes() {
    echo "[Processes]"
    local found=0
    for sh in run_benchmark.sh download_bench_samples.sh; do
        local pids
        pids=$(find_script_pids "$sh")
        if [[ -n "$pids" ]]; then
            for pid in $pids; do
                local etime stat cmd
                read -r etime stat <<< "$(ps -p "$pid" -o etime=,stat= 2>/dev/null | xargs)"
                cmd=$(ps -p "$pid" -o args= 2>/dev/null)
                printf "  [%s] PID=%s  ELAPSED=%s  STAT=%s\n" "$sh" "$pid" "$etime" "$stat"
                printf "         %s\n" "$cmd"
                found=1
            done
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "  (no benchmark or download processes running)"
    fi
    echo
}

report_benchmark() {
    echo "[Benchmark progress]"
    # Pick the newest full_benchmark_*.csv across all timestamp dirs.
    local csv
    csv=$(find "$RESULTS_ROOT" -maxdepth 3 -name 'full_benchmark_*.csv' \
            -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | awk '{print $2}')
    if [[ -z "$csv" || ! -f "$csv" ]]; then
        echo "  (no benchmark CSV found under $RESULTS_ROOT)"
        echo
        return
    fi
    local rows mtime age age_s
    rows=$(( $(wc -l < "$csv") - 1 ))
    mtime=$(stat -c %Y "$csv")
    age_s=$(( $(date +%s) - mtime ))
    age=$(human_dur "$age_s")
    printf "  CSV       : %s\n" "$csv"
    printf "  Rows      : %d   (last update: %s ago)\n" "$rows" "$age"

    # Pull last completed run
    local last
    last=$(tail -1 "$csv")
    printf "  Last row  : %s\n" "$last"

    # Tally pass/fail/timeout/skip in column 10 (validation)
    awk -F, 'NR>1 {c[$10]++} END {
        printf "  Validation: ";
        for (k in c) printf "%s=%d  ", k, c[k];
        print ""
    }' "$csv"

    # Per-tool counts
    awk -F, 'NR>1 {c[$3]++} END {
        printf "  By tool   : ";
        for (k in c) printf "%s=%d  ", k, c[k];
        print ""
    }' "$csv"

    # Status file (written by run_benchmark.sh ticker every 3 min)
    local status_file
    status_file="${csv%.csv}"
    status_file="$(dirname "$csv")/$(basename "$csv" .csv | sed 's/^full_benchmark/full_status/').txt"
    if [[ -f "$status_file" ]]; then
        printf "  Status    : %s\n" "$status_file"
        # Pull "Progress : X / Y runs"
        local prog
        prog=$(grep -m1 'Progress' "$status_file" 2>/dev/null | sed 's/^[[:space:]]*//')
        [[ -n "$prog" ]] && printf "  %s\n" "$prog"
    fi
    echo
}

report_download() {
    echo "[Download progress]"
    # Find most recent log written by download_bench_samples.sh.  The script
    # is normally launched with stdout redirected (e.g. /tmp/dl_samples.log).
    local log
    for cand in /tmp/dl_samples.log /tmp/download_bench.log; do
        [[ -f "$cand" ]] && log="$cand" && break
    done
    if [[ -n "${log:-}" ]]; then
        local mtime age_s
        mtime=$(stat -c %Y "$log")
        age_s=$(( $(date +%s) - mtime ))
        printf "  Log       : %s   (last update: %s ago)\n" "$log" "$(human_dur "$age_s")"
        # Count completed accessions ('>>>' lines) and SKIP lines
        local started skipped
        started=$(grep -c '^>>>' "$log" 2>/dev/null || echo 0)
        skipped=$(grep -c 'SKIP' "$log" 2>/dev/null || echo 0)
        printf "  Accessions: %d started, %d already-present\n" "$started" "$skipped"
        echo "  Tail:"
        tail -5 "$log" | sed 's/^/    /'
    fi

    # In-flight wget temp files (largest .fastq.gz currently growing)
    echo "  In-flight files (last 5 min mtime):"
    local now=$(date +%s)
    find "$FASTQ_DIR" -type f -name '*.fastq.gz*' -mmin -5 -printf '%T@ %s %p\n' 2>/dev/null \
        | sort -rn | head -5 \
        | while read -r ts size path; do
            printf "    %-12s  %s\n" "$(human_size "$size")" "$path"
        done
    echo
}

report_disk_gpu() {
    echo "[Disk]"
    df -h "$FASTQ_DIR" "$PROJ_DIR" 2>/dev/null \
        | awk 'NR==1 || /[0-9]/' \
        | awk '!seen[$1]++' \
        | sed 's/^/  /'
    echo
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[GPU]"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                   --format=csv,noheader 2>/dev/null \
            | sed 's/^/  /'
        # Active compute processes
        local procs
        procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                          --format=csv,noheader 2>/dev/null)
        if [[ -n "$procs" ]]; then
            echo "  Active compute processes:"
            echo "$procs" | sed 's/^/    /'
        fi
        echo
    fi
}

snapshot() {
    report_header
    report_processes
    report_benchmark
    report_download
    report_disk_gpu
}

if [[ "$WATCH" -eq 1 ]]; then
    while true; do
        clear
        snapshot
        echo "(refresh every ${INTERVAL}s — Ctrl-C to stop)"
        sleep "$INTERVAL"
    done
else
    snapshot
fi
