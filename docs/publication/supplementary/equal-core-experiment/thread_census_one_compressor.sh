#!/usr/bin/env bash
# thread_census_one_compressor.sh -- thread census of the CPU build with both ablation switches
# set (FASTP_COMPRESSORS=1 FASTP_NO_ASYNC_DECOMP=1), i.e. the "switches off" ablation row.
# much CPU each thread uses (JCB R3, AE major comment).
#
# For each binary and -w value, run once without taskset on the 13.6 GB paired-end
# dataset (ERR1044780) and sample /proc/<pid>/task/*/stat every 0.5 s. For every
# thread id keep the last seen user+system time. Report: threads created, threads
# that used >= 5% of the wall time on a CPU ("busy"), and process CPU seconds / wall
# seconds (mean number of busy cores). Output to /dev/shm, page cache dropped first.
set -uo pipefail
ROOT="${ROOT:-$HOME/bioauto_rev}"
OUT="${OUT:-$ROOT/results_equal_core}"; mkdir -p "$OUT"
CSV="$OUT/thread_census_one_compressor.csv"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"
declare -A BIN=( [upstream]="$ROOT/fastp_opengene/fastp" [cpu]="$ROOT/fastp_final_gx/fastp-cpu" [gpu]="$ROOT/fastp_final_gx/fastp" )
R1="$HOME/FASTQ/WGS/ERR1044780_1.fastq.gz"; R2="$HOME/FASTQ/WGS/ERR1044780_2.fastq.gz"
echo "mode,w,wall_s,cpu_s,mean_cores,threads_created,threads_busy5pct,per_thread_cpu_s_sorted" > "$CSV"
for w in 8 32; do for mode in cpu; do
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  o1=/dev/shm/tc_R1.fq.gz; o2=/dev/shm/tc_R2.fq.gz
  start=$(date +%s.%N)
  FASTP_COMPRESSORS=1 FASTP_NO_ASYNC_DECOMP=1 "${BIN[$mode]}" -w "$w" -j /dev/shm/tc.json -h /dev/null -i "$R1" -I "$R2" -o "$o1" -O "$o2" >/dev/null 2>&1 &
  pid=$!
  python3 - "$pid" > /dev/shm/tc_threads.txt <<'PY'
import os, sys, time
pid = sys.argv[1]; hz = os.sysconf('SC_CLK_TCK'); seen = {}
while os.path.exists(f"/proc/{pid}/stat"):
    try:
        st = open(f"/proc/{pid}/stat").read()
        if st.split(") ")[1].split()[0] == "Z": break
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                f = open(f"/proc/{pid}/task/{tid}/stat").read()
                rest = f.rsplit(") ", 1)[1].split()
                seen[tid] = (int(rest[11]) + int(rest[12])) / hz
            except OSError: pass
    except (OSError, IndexError): break
    time.sleep(0.5)
for tid, c in seen.items(): print(tid, c)
PY
  wait "$pid"; end=$(date +%s.%N)
  wall=$(python3 -c "print(round($end-$start,1))")
  python3 - "$mode" "$w" "$wall" >> "$CSV" <<'PY'
import sys
mode, w, wall = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = sorted((float(l.split()[1]) for l in open("/dev/shm/tc_threads.txt")), reverse=True)
tot = sum(c)
print(f"{mode},{w},{wall},{tot:.0f},{tot/wall:.1f},{len(c)},{sum(1 for x in c if x >= 0.05*wall)},\"{' '.join(f'{x:.0f}' for x in c)}\"")
PY
  tail -1 "$CSV"
  rm -f "$o1" "$o2" /dev/shm/tc.json /dev/shm/tc_threads.txt
done; done
