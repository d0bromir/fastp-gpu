#!/usr/bin/env bash
# Targeted 3-rep experiment: WGS_PE_40G, T=32, CPU vs GPU mode.
# Resolves the BIOADV review's "run once" vs "within noise" contradiction
# for the specific claim in the Application Notes paper (GPU mode at T=32
# on the largest dataset is "within noise" of CPU mode).
set -euo pipefail

REPO=~/fastp-gpu-appnotes-work/repo
CPU_BIN="$REPO/fastp-cpu"
GPU_BIN="$REPO/fastp"
R1=~/FASTQ/WGS/DRR216653_1.fastq.gz
R2=~/FASTQ/WGS/DRR216653_2.fastq.gz
OUT=~/fastp-gpu-appnotes-work/reps_results
mkdir -p "$OUT"
CSV="$OUT/reps_wgs_pe_40g_t32.csv"
echo "tool,rep,threads,wall_s,reads_in,reads_out" > "$CSV"

drop_cache() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true; }

run_one() {
    local label="$1" bin="$2" rep="$3"
    local tmp_json tmp_o1 tmp_o2
    tmp_json=$(mktemp /tmp/reps_XXXXXX.json)
    tmp_o1=$(mktemp /tmp/reps_XXXXXX_R1.fq.gz)
    tmp_o2=$(mktemp /tmp/reps_XXXXXX_R2.fq.gz)
    drop_cache
    local t0 t1 wall
    t0=$(date +%s.%N)
    "$bin" -w 32 -i "$R1" -I "$R2" -o "$tmp_o1" -O "$tmp_o2" -j "$tmp_json" -h /dev/null \
        > /tmp/reps_${label}_${rep}.stdout 2> /tmp/reps_${label}_${rep}.stderr
    t1=$(date +%s.%N)
    wall=$(echo "$t1 - $t0" | bc)
    local ri ro
    ri=$(python3 -c "import json;print(json.load(open('$tmp_json'))['summary']['before_filtering']['total_reads'])" 2>/dev/null || echo NA)
    ro=$(python3 -c "import json;print(json.load(open('$tmp_json'))['summary']['after_filtering']['total_reads'])" 2>/dev/null || echo NA)
    echo "$label,$rep,32,$wall,$ri,$ro" >> "$CSV"
    echo "[$label rep $rep] wall=${wall}s reads_in=$ri reads_out=$ro"
    cp "$tmp_json" "$OUT/${label}_rep${rep}.json"
    rm -f "$tmp_o1" "$tmp_o2" "$tmp_json"
}

for rep in 1 2 3; do
    run_one cpu "$CPU_BIN" "$rep"
    run_one gpu "$GPU_BIN" "$rep"
done

echo "=== Done. Results in $CSV ==="
cat "$CSV"
