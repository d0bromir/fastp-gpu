#!/usr/bin/env bash
# equal_core_xeon.sh -- upstream fastp v1.3.3 and the fastp-gpu CPU build on the same
# number of cores, Xeon host (a2; no GPU, no sudo). JCB R3, AE major comment.
# Same protocol as equal_core_experiment.sh, except: cores 0..C-1 (node 0 for C <= 16,
# nodes 0-1 for C = 32) and page-cache eviction with posix_fadvise(DONTNEED).
# Binaries are those of the paper's Xeon table:
#   fastp_opengene/fastp       upstream v1.3.3
#   fastp_d0bromir/fastp-cpu   fastp-gpu CPU build, commit 8e87c0e
# Dataset ERR1044319 (19.4 GB PE). C = 8, 16, 32. Three repeats, shuffled.
set -uo pipefail
source "$HOME/bioauto_rev/watched_run.sh"
ROOT="${ROOT:-$HOME/bioauto_rev}"; REPS="${REPS:-3}"
OUT="${OUT:-$ROOT/results_equal_core}"; mkdir -p "$OUT"
CSV="$OUT/equal_core_xeon.csv"
declare -A BIN=( [upstream]="$ROOT/fastp_opengene/fastp" [cpu]="$ROOT/fastp_d0bromir/fastp-cpu" )
R1="$HOME/FASTQ/WGS/ERR1044319_1.fastq.gz"; R2="$HOME/FASTQ/WGS/ERR1044319_2.fastq.gz"
[[ -f "$CSV" ]] || echo "rep,cores,mode,wall_s,user_s,sys_s,rss_kb,exit,load1,reads_in,passed" > "$CSV"
evict() { python3 -c "
import os,sys
for p in sys.argv[1:]:
    fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" "$R1" "$R2"; }
for rep in $(seq 1 "$REPS"); do
  cells=(); for C in 8 16 32; do for m in upstream cpu; do cells+=("$C|$m"); done; done
  while IFS= read -r cell; do
    IFS='|' read -r C mode <<< "$cell"; bin=${BIN[$mode]}
    j=/dev/shm/ex_$$.json; o1=/dev/shm/ex_$$_R1.fq.gz; o2=/dev/shm/ex_$$_R2.fq.gz; tl=/dev/shm/ex_$$.time
    sync; evict; load1=$(cut -d' ' -f1 /proc/loadavg); : > "$tl"; rm -f "$j"
    WATCH_GLOB="$o1" watched_run "$bin" 300 -- taskset -c "0-$((C - 1))" /usr/bin/time -f "%e %U %S %M" -o "$tl" \
        timeout 7200 "$bin" -w "$C" -j "$j" -h /dev/null -i "$R1" -I "$R2" -o "$o1" -O "$o2" >/dev/null 2>&1
    ec=$WATCHED_EXIT
    read -r wall user sys rss < <(tail -1 "$tl")
    read -r rin pas < <(python3 -c "import json;d=json.load(open('$j'));print(d['summary']['before_filtering']['total_reads'],d['filtering_result']['passed_filter_reads'])" 2>/dev/null || echo "0 0")
    echo "$rep,$C,$mode,${wall:-0},${user:-0},${sys:-0},${rss:-0},$ec,$load1,$rin,$pas" | tee -a "$CSV"
    rm -f "$j" "$o1" "$o2" "$tl"
  done < <(printf '%s\n' "${cells[@]}" | shuf)
done
echo "EQUAL_CORE_XEON_DONE $(date -Is)" >> "$OUT/done.txt"
