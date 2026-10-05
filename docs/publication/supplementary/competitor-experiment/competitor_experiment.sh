#!/usr/bin/env bash
# Competitor comparison: fastp-gpu CPU-only vs RabbitQCPlus, same host (a2,
# x86-64 Xeon Gold 5218, 64 physical cores, no GPU -- the same host already
# used for this paper's x86 portability results), same datasets, default
# settings for both tools.
set -uo pipefail

REPO=~/appnotes-competitor-work/repo
FASTP_BIN="$REPO/fastp-cpu"
RABBIT_BIN=~/appnotes-competitor-work/RabbitQCPlus/RabbitQCPlus
WGS=~/FASTQ/WGS
OUT=~/appnotes-competitor-work/results
mkdir -p "$OUT"
CSV="$OUT/competitor_comparison.csv"
echo "dataset,tool,threads,wall_s,reads_in,reads_out" > "$CSV"

drop_cache() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true; }

declare -A R1 R2
R1[WGS_PE_18.2G]="$WGS/ERR1044319_1.fastq.gz"; R2[WGS_PE_18.2G]="$WGS/ERR1044319_2.fastq.gz"
R1[WGS_PE_40G]="$WGS/DRR216653_1.fastq.gz";    R2[WGS_PE_40G]="$WGS/DRR216653_2.fastq.gz"

run_fastp() {
    local ds="$1" threads="$2"
    local tmp_json tmp_o1 tmp_o2
    tmp_json=$(mktemp /tmp/comp_XXXXXX.json)
    tmp_o1=$(mktemp /tmp/comp_XXXXXX_R1.fq.gz)
    tmp_o2=$(mktemp /tmp/comp_XXXXXX_R2.fq.gz)
    drop_cache
    local t0 t1 wall
    t0=$(date +%s.%N)
    "$FASTP_BIN" -w "$threads" -i "${R1[$ds]}" -I "${R2[$ds]}" -o "$tmp_o1" -O "$tmp_o2" \
        -j "$tmp_json" -h /dev/null > "$OUT/fastp_${ds}_${threads}.stdout" 2>&1
    t1=$(date +%s.%N)
    wall=$(echo "$t1 - $t0" | bc)
    local ri ro
    ri=$(python3 -c "import json;print(json.load(open('$tmp_json'))['summary']['before_filtering']['total_reads'])" 2>/dev/null || echo NA)
    ro=$(python3 -c "import json;print(json.load(open('$tmp_json'))['summary']['after_filtering']['total_reads'])" 2>/dev/null || echo NA)
    echo "$ds,fastp_cpu,$threads,$wall,$ri,$ro" >> "$CSV"
    echo "[fastp $ds T=$threads] wall=${wall}s reads_in=$ri reads_out=$ro"
    cp "$tmp_json" "$OUT/fastp_${ds}_T${threads}.json"
    rm -f "$tmp_o1" "$tmp_o2" "$tmp_json"
}

run_rabbit() {
    local ds="$1" threads="$2"
    local tmp_o1 tmp_o2 workdir
    workdir=$(mktemp -d /tmp/comp_rqc_XXXXXX)
    tmp_o1="$workdir/o1.fq.gz"
    tmp_o2="$workdir/o2.fq.gz"
    drop_cache
    local t0 t1 wall
    t0=$(date +%s.%N)
    (cd "$workdir" && "$RABBIT_BIN" -w "$threads" -i "${R1[$ds]}" -I "${R2[$ds]}" -o "$tmp_o1" -O "$tmp_o2" \
        --overWrite > "$OUT/rabbit_${ds}_${threads}.stdout" 2> "$OUT/rabbit_${ds}_${threads}.stderr")
    t1=$(date +%s.%N)
    wall=$(echo "$t1 - $t0" | bc)
    local ri="NA" ro="NA"
    local jf
    jf=$(ls "$workdir"/*RabbitQCPlus.json 2>/dev/null | head -1)
    if [[ -n "$jf" ]]; then
        ri=$(python3 -c "
import json
d=json.load(open('$jf'))
r1=d.get('summary',{}).get('before_filtering',{}).get('read1',{}).get('total_reads',0)
r2=d.get('summary',{}).get('before_filtering',{}).get('read2',{}).get('total_reads',0)
print(r1+r2 if (r1 or r2) else 'NA')
" 2>/dev/null || echo NA)
        ro=$(python3 -c "
import json
d=json.load(open('$jf'))
print(d.get('filtering_result',{}).get('passed_filter_reads','NA'))
" 2>/dev/null || echo NA)
        cp "$jf" "$OUT/rabbit_${ds}_T${threads}.json"
    fi
    echo "$ds,rabbitqcplus,$threads,$wall,$ri,$ro" >> "$CSV"
    echo "[rabbit $ds T=$threads] wall=${wall}s reads_in=$ri reads_out=$ro"
    rm -rf "$workdir"
}

for ds in WGS_PE_18.2G WGS_PE_40G; do
    for t in 8 32; do
        run_fastp "$ds" "$t"
        run_rabbit "$ds" "$t"
    done
done

echo "=== Done ==="
cat "$CSV"
