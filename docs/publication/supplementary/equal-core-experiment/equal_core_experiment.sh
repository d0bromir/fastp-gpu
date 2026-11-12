#!/usr/bin/env bash
# equal_core_experiment.sh -- upstream fastp v1.3.3, fastp-gpu CPU mode and GPU mode
# on the same number of cores (JCB R3, AE major comment).
#
# -w does not bound the thread count of any of the three binaries. Upstream adds
# reader threads to the -w workers; fastp-gpu adds reader, decompression, writer and
# compressor-pool threads. To give all three the same resources, every run is
# confined to C cores with taskset -c 64-(64+C-1) and gets -w C. All threads of the
# process share these C cores.
#
# Runs on galaxy from ~/bioauto_rev with the binaries of the paper:
#   fastp_opengene/fastp       upstream OpenGene fastp built from tag v1.3.3
#   fastp_final_gx/fastp-cpu   fastp-gpu CPU build, release .17 (commit 8e87c0e)
#   fastp_final_gx/fastp       fastp-gpu CUDA build, same commit (uses both A100s)
# Each run records wall time, user+sys CPU seconds and peak RSS (GNU time), the
# 1-min load average at start, reads in/passed, and the md5 of 14 JSON fields.
# A run whose output stops growing for 300 s is killed and recorded as HANG.
# Output to /dev/shm, deleted after each run. Page cache dropped before every run.
# Cell order is shuffled within each repeat.
set -uo pipefail
source "$HOME/bioauto_rev/watched_run.sh"

ROOT="${ROOT:-$HOME/bioauto_rev}"
REPS="${REPS:-3}"
CORE0="${CORE0:-64}"
OUT="${OUT:-$ROOT/results_equal_core}"; mkdir -p "$OUT"
CSV="$OUT/equal_core.csv"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"
declare -A BIN=( [upstream]="$ROOT/fastp_opengene/fastp" [cpu]="$ROOT/fastp_final_gx/fastp-cpu" [gpu]="$ROOT/fastp_final_gx/fastp" )

# dataset|layout
DATA=( "DRR216653|PE" "ERR1044319|PE" "ERR1044780|PE" "ERR1044780|SE" )
CORES=( 8 16 32 )

[[ -f "$CSV" ]] || echo "rep,dataset,layout,cores,mode,wall_s,user_s,sys_s,rss_kb,exit,load1,reads_in,passed,json14_md5" > "$CSV"

drop_cache() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }

json14() {
python3 - "$1" <<'PY'
import json, sys, hashlib
d = json.load(open(sys.argv[1]))
paths = """summary.before_filtering.total_reads summary.before_filtering.total_bases
summary.after_filtering.total_reads summary.after_filtering.total_bases
summary.after_filtering.q20_rate summary.after_filtering.q30_rate
filtering_result.passed_filter_reads filtering_result.low_quality_reads
filtering_result.too_many_N_reads filtering_result.adapter_dimer_reads
filtering_result.too_short_reads filtering_result.too_long_reads
adapter_cutting.adapter_trimmed_reads adapter_cutting.adapter_trimmed_bases""".split()
vals = []
for p in paths:
    x = d
    for k in p.split("."):
        x = x.get(k, {}) if isinstance(x, dict) else x
    vals.append(repr(x))
print(hashlib.md5("|".join(vals).encode()).hexdigest(),
      d["summary"]["before_filtering"]["total_reads"], d["filtering_result"]["passed_filter_reads"])
PY
}

for rep in $(seq 1 "$REPS"); do
  cells=()
  for d in "${DATA[@]}"; do for C in "${CORES[@]}"; do for m in upstream cpu gpu; do
    cells+=("$d|$C|$m"); done; done; done
  while IFS= read -r cell; do
    IFS='|' read -r ds layout C mode <<< "$cell"
    bin=${BIN[$mode]}
    cores="$CORE0-$((CORE0 + C - 1))"
    j=/dev/shm/ec_$$.json; o1=/dev/shm/ec_$$_R1.fq.gz; o2=/dev/shm/ec_$$_R2.fq.gz; tl=/dev/shm/ec_$$.time
    R1="$HOME/FASTQ/WGS/${ds}_1.fastq.gz"; R2="$HOME/FASTQ/WGS/${ds}_2.fastq.gz"
    if [[ $layout == PE ]]; then io=(-i "$R1" -I "$R2" -o "$o1" -O "$o2"); else io=(-i "$R1" -o "$o1"); fi
    drop_cache
    load1=$(cut -d' ' -f1 /proc/loadavg)
    : > "$tl"; rm -f "$j"
    WATCH_GLOB="$o1" watched_run "$bin" 300 -- taskset -c "$cores" /usr/bin/time -f "%e %U %S %M" -o "$tl" \
        timeout 7200 "$bin" -w "$C" -j "$j" -h /dev/null "${io[@]}" >/dev/null 2>&1
    ec=$WATCHED_EXIT
    read -r wall user sys rss < <(tail -1 "$tl"); wall=${wall:-0}; user=${user:-0}; sys=${sys:-0}; rss=${rss:-0}
    read -r md5 rin pas < <(json14 "$j" 2>/dev/null || echo "NA 0 0")
    echo "$rep,$ds,$layout,$C,$mode,$wall,$user,$sys,$rss,$ec,$load1,$rin,$pas,$md5" | tee -a "$CSV"
    rm -f "$j" "$o1" "$o2" "$tl"
  done < <(printf '%s\n' "${cells[@]}" | shuf)
done
echo "EQUAL_CORE_DONE $(date -Is)" >> "$OUT/done.txt"
