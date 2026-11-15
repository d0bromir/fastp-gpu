#!/usr/bin/env bash
# RTX 3050 laptop re-measurement: CPU build vs GPU build on the panel inputs.
# Usage: rtx3050_sweep.sh <bin_dir> <panel_dir> <out_csv>
set -u
BIN=$1; P=$2; OUT=$3
echo "rep,layout,threads,mode,wall_s,md5_r1,md5_r2" > "$OUT"
cells=()
for L in SE PE; do for T in 1 2 4 8; do for M in cpu gpu; do cells+=("$L,$T,$M"); done; done; done
for rep in 1 2 3; do
  for c in $(printf '%s\n' "${cells[@]}" | shuf); do
    IFS=, read -r L T M <<< "$c"
    rm -f /dev/shm/rtx_o1.fq.gz /dev/shm/rtx_o2.fq.gz
    if [ "$L" = SE ]; then
      args=(-i "$P/R1.fastq.gz" -o /dev/shm/rtx_o1.fq.gz)
    else
      args=(-i "$P/R1.fastq.gz" -I "$P/R2.fastq.gz" -o /dev/shm/rtx_o1.fq.gz -O /dev/shm/rtx_o2.fq.gz)
    fi
    t0=$(date +%s.%N)
    "$BIN/fastp-$M-x86" -w "$T" "${args[@]}" -j /dev/shm/rtx.json -h /dev/shm/rtx.html >/dev/null 2>&1
    t1=$(date +%s.%N)
    m1=$(zcat /dev/shm/rtx_o1.fq.gz | md5sum | cut -c1-32)
    m2=-; [ "$L" = PE ] && m2=$(zcat /dev/shm/rtx_o2.fq.gz | md5sum | cut -c1-32)
    echo "$rep,$L,$T,$M,$(echo "$t1 - $t0" | bc),$m1,$m2" >> "$OUT"
  done
done
