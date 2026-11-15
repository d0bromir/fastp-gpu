#!/usr/bin/env bash
# core_budget_experiment.sh -- CPU-only vs GPU mode under a fixed CPU-core allocation
# (BIOADV R2, Reviewer 3, comment 1).
#
# Scenario: a cluster job receives C CPU cores and one node with GPUs, as with
# SLURM --cpus-per-task=C. Every thread of the process (reader, decompressor,
# workers, compressors, writer, CUDA host threads) is confined to the same C cores
# with taskset, and fastp gets -w C. This is the setting in which moving the
# filtering and statistics work off the CPU can change wall time, because the
# process cannot spill onto idle cores of a 128-core host.
#
# Runs on galaxy from ~/bioauto_rev with the release .17 binaries (commit 8e87c0e):
#   fastp_final_gx/fastp-cpu   CPU-only build
#   fastp_final_gx/fastp       CUDA build (uses both A100s)
# Each run records wall time, user+sys CPU seconds and peak RSS (GNU time), the
# 1-min load average at start, reads in/passed, and the md5 of 14 JSON fields.
# Output goes to /dev/shm and is deleted after each run. Page cache is dropped
# before every run. Cell order is shuffled within each repeat.
set -uo pipefail
source "$HOME/bioauto_rev/watched_run.sh"

ROOT="${ROOT:-$HOME/bioauto_rev}"
REPS="${REPS:-3}"
CORE0="${CORE0:-64}"                   # first core of the allocation block
OUT="${OUT:-$ROOT/results_core_budget}"; mkdir -p "$OUT"
CSV="$OUT/core_budget.csv"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"
CPU="$ROOT/fastp_final_gx/fastp-cpu"
GPU="$ROOT/fastp_final_gx/fastp"

# dataset|layout|cores
CELLS_DEF=(
  "DRR216653|PE|8" "DRR216653|PE|16"
  "ERR1044319|PE|4" "ERR1044319|PE|8" "ERR1044319|PE|16"
  "ERR1044780|SE|4" "ERR1044780|SE|8" "ERR1044780|SE|16"
)

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
  for c in "${CELLS_DEF[@]}"; do cells+=("$c|cpu" "$c|gpu"); done
  while IFS= read -r cell; do
    IFS='|' read -r ds layout C mode <<< "$cell"
    bin=$CPU; [[ $mode == gpu ]] && bin=$GPU
    cores="$CORE0-$((CORE0 + C - 1))"
    j=/dev/shm/cb_$$.json; o1=/dev/shm/cb_$$_R1.fq.gz; o2=/dev/shm/cb_$$_R2.fq.gz; tl=/dev/shm/cb_$$.time
    R1="$HOME/FASTQ/WGS/${ds}_1.fastq.gz"; R2="$HOME/FASTQ/WGS/${ds}_2.fastq.gz"
    if [[ $layout == PE ]]; then io=(-i "$R1" -I "$R2" -o "$o1" -O "$o2"); else io=(-i "$R1" -o "$o1"); fi
    drop_cache
    load1=$(cut -d' ' -f1 /proc/loadavg)
    : > "$tl"
    WATCH_GLOB="$o1" watched_run "$bin" 300 -- taskset -c "$cores" /usr/bin/time -f "%e %U %S %M" -o "$tl" \
        timeout 7200 "$bin" -w "$C" -j "$j" -h /dev/null "${io[@]}" >/dev/null 2>&1
    ec=$WATCHED_EXIT
    read -r wall user sys rss < <(tail -1 "$tl"); wall=${wall:-0}; user=${user:-0}; sys=${sys:-0}; rss=${rss:-0}
    read -r md5 rin pas < <(json14 "$j" 2>/dev/null || echo "NA 0 0")
    echo "$rep,$ds,$layout,$C,$mode,$wall,$user,$sys,$rss,$ec,$load1,$rin,$pas,$md5" | tee -a "$CSV"
    rm -f "$j" "$o1" "$o2" "$tl"
  done < <(printf '%s\n' "${cells[@]}" | shuf)
done
echo "CORE_BUDGET_DONE $(date -Is)" >> "$OUT/done.txt"
