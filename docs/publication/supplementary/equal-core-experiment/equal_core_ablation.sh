#!/usr/bin/env bash
# equal_core_ablation.sh -- ablation of the fastp-gpu CPU build on a fixed number of
# cores (JCB R3, AE major comment on "parallel output compression gives the largest
# gain"). Same protocol as equal_core_experiment.sh (taskset C cores, -w C).
# The full CPU build and upstream at the same C come from equal_core_experiment.sh.
#   one_compressor : FASTP_COMPRESSORS=1 (one compressor per output file, async decompression on)
#   no_async       : FASTP_NO_ASYNC_DECOMP=1 (full CPU build without async decompression)
# Dataset: DRR216653 (41.3 GB PE). C = 8, 32. Three repeats, shuffled.
set -uo pipefail
source "$HOME/bioauto_rev/watched_run.sh"
ROOT="${ROOT:-$HOME/bioauto_rev}"; REPS="${REPS:-3}"; CORE0="${CORE0:-64}"
OUT="${OUT:-$ROOT/results_equal_core}"; mkdir -p "$OUT"
CSV="$OUT/equal_core_ablation.csv"
CPU="$ROOT/fastp_final_gx/fastp-cpu"
CONFIGS=( "one_compressor|FASTP_COMPRESSORS=1" "no_async|FASTP_NO_ASYNC_DECOMP=1" )
[[ -f "$CSV" ]] || echo "rep,cores,config,wall_s,user_s,sys_s,rss_kb,exit,load1,reads_in,passed" > "$CSV"
R1="$HOME/FASTQ/WGS/DRR216653_1.fastq.gz"; R2="$HOME/FASTQ/WGS/DRR216653_2.fastq.gz"
for rep in $(seq 1 "$REPS"); do
  cells=(); for C in 8 32; do for c in "${CONFIGS[@]}"; do cells+=("$c|$C"); done; done
  while IFS= read -r cell; do
    IFS='|' read -r name envs C <<< "$cell"
    cores="$CORE0-$((CORE0 + C - 1))"
    j=/dev/shm/ea_$$.json; o1=/dev/shm/ea_$$_R1.fq.gz; o2=/dev/shm/ea_$$_R2.fq.gz; tl=/dev/shm/ea_$$.time
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    load1=$(cut -d' ' -f1 /proc/loadavg); : > "$tl"; rm -f "$j"
    WATCH_GLOB="$o1" watched_run "$CPU" 300 -- env $envs taskset -c "$cores" /usr/bin/time -f "%e %U %S %M" -o "$tl" \
        timeout 7200 "$CPU" -w "$C" -j "$j" -h /dev/null -i "$R1" -I "$R2" -o "$o1" -O "$o2" >/dev/null 2>&1
    ec=$WATCHED_EXIT
    read -r wall user sys rss < <(tail -1 "$tl")
    read -r rin pas < <(python3 -c "import json;d=json.load(open('$j'));print(d['summary']['before_filtering']['total_reads'],d['filtering_result']['passed_filter_reads'])" 2>/dev/null || echo "0 0")
    echo "$rep,$C,$name,${wall:-0},${user:-0},${sys:-0},${rss:-0},$ec,$load1,$rin,$pas" | tee -a "$CSV"
    rm -f "$j" "$o1" "$o2" "$tl"
  done < <(printf '%s\n' "${cells[@]}" | shuf)
done
echo "EQUAL_CORE_ABLATION_DONE $(date -Is)" >> "$OUT/done.txt"
