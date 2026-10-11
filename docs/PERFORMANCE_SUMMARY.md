# Performance Summary

**Last Updated:** 2026-05 (full benchmark vs upstream OpenGene fastp 1.3.3,
198 runs across 8 datasets × 6 thread counts, see CSV path below)
**fastp Version:** 1.3.3-d0bromir (rebased on upstream 1.3.3)
**Hardware:** ARM Neoverse N1 (128 cores) + 2× NVIDIA A100-SXM4-80GB
**CUDA:** 12.6 / nvCOMP 13

## Source of truth

This file gives a high-level orientation only. The **canonical** numbers used
by the paper live in:

- CSV: `benchmark_results/fastp-gpu_v1.3.3-d0bromir/vs_opengene_v1.3.3/galaxy_arm_a100/20260503_193146/full_benchmark_20260503_193146.csv`
- Paper: `docs/publication/GPU_ACCELERATED_FASTP_PAPER.pdf` (and the App Notes / TUS / ST_SIM concept variants in the same folder)
- Figures: `docs/publication/figures/`

Reproduce on your hardware with `scripts/build_all.sh && scripts/run_benchmark.sh`.

## Headline result

Geometric-mean wall-clock speedup of `fastp-d0bromir` (GPU build) over
upstream `fastp 1.3.3` at the per-dataset best thread count: **3.00×**.

The dominant mechanism is parallel BGZF output compression, which removes
the single-threaded `gzip` write bottleneck that pins the upstream baseline
at T≥4 on PE workloads. The GPU per-read statistics kernel contributes an
additional 8–14% wall-clock reduction on stdout / non-compressed-output
workloads where compression is not the bottleneck.

See the paper for the full per-dataset breakdown, the CPU-envelope caveat
(`fastp-cpu` and `fastp-gpu` track each other within ±6% on gzip-output
workloads where compression dominates), and the seven upstream-baseline
crash sites we worked around at high thread counts on aarch64.

---

## Historical notes (v5 / v6 / v7, kept for context)

The sections below describe earlier benchmark snapshots and the
optimisations that produced them. The numbers are *not* the current
canonical results — those live in the CSV referenced above — but the
explanations of *why* each optimisation works are still accurate.

## Key Finding (v6 — parallel output compression)

v6 adds a pigz-style parallel compression pipeline in `WriterThread`: up to 4 `libdeflate` compressor threads produce `gzip` blocks concurrently; a single writer thread flushes them in sequence-number order. This eliminates the single-threaded gzip bottleneck that previously limited throughput at T≥4.

With gzip output, **parallel compression becomes the new bottleneck at T≥4**, masking the GPU acceleration advantage. CPU and GPU achieve near-identical wall times, with GPU running 2–6% slower than CPU due to CUDA context initialisation overhead (~3 s on dual-A100). At T=16 on WGS-PE-18.2G, CPU reaches **100.9 s** and GPU **105.0 s**.

For benchmarks without gzip output (stdout or raw FASTQ), the v5 GPU advantage (8–14% at 8T) remains unchanged.

## Key Finding (v5 — speculative-stats optimisation, previous)

The speculative post-filter statistics optimisation (`unstatRead`) eliminates redundant per-base histogram computation for ~95% of reads. Combined with the GPU batch filtering pipeline, fastp-GPU achieves **8–18% speedup over CPU-only fastp** on WGS-scale files at 4–32 threads. On the largest test file (WGS-6G), GPU mode runs in **88.2 s at 8 threads vs 100.3 s CPU-only** — a 12.1% improvement.

---

## v7 Benchmark — Full Three-Tool Comparison (2026-04-16)

**Datasets:** Panel SE 148M (S1A, 1.38M reads) · WGS SE 6.3G (ERR1044780, 80.3M reads) · WGS PE 12.8G (ERR1044780 R1+R2, 160.6M reads) · WGS PE 18.2G (ERR1044319, 234.2M reads)
**Output:** `.fq.gz` (pigz-style parallel compression, 4 libdeflate workers)
**Validation:** 60/60 PASS — `filtering_result.passed_filter_reads` identical between d0bromir and opengene for all runs
**Source:** `full_benchmark_20260416_064709.csv` — all three tools benchmarked in the same invocation

### Panel SE 148M (S1A_S1_L001_R1, 1.38M reads, SE)

| Tool | 1T | 2T | 4T | 8T | 16T |
|---|---|---|---|---|---|
| opengene | 7.6 s | 7.5 s | 7.5 s | 7.6 s | 7.6 s |
| d0bromir_cpu | 7.6 s | 4.6 s | **3.8 s** | 3.9 s | 4.0 s |
| d0bromir_gpu | 11.6 s | 8.6 s | 7.2 s | 7.4 s | 7.3 s |
| CPU vs opengene | +0.8% | **−38.7%** | **−49.8%** | **−48.3%** | **−47.6%** |
| GPU vs opengene | +52.4% | +14.3% | −4.9% | −2.7% | −3.6% |

Panel file is too small to amortise GPU context init (~3 s). GPU is slower at T=1–2; near-equal at T≥4. CPU mode achieves ~2× speedup at T=4 through parallel compression.

### WGS-SE-6.3G (ERR1044780_1, 80.3M reads, SE)

| Tool | 1T | 2T | 4T | 8T | 16T |
|---|---|---|---|---|---|
| opengene | 293.7 s | 293.6 s | 292.5 s | 293.0 s | 292.8 s |
| d0bromir_cpu | 293.5 s | 151.8 s | **82.3 s** | 82.4 s | 82.6 s |
| d0bromir_gpu | 295.6 s | 156.6 s | 86.8 s | 86.9 s | 86.8 s |
| CPU vs opengene | −0.1% | **−48.3%** | **−71.9%** | **−71.9%** | **−71.8%** |
| GPU vs opengene | +0.7% | **−46.7%** | **−70.3%** | **−70.3%** | **−70.4%** |

opengene is flat ~293 s at all T — bottlenecked on single-threaded gzip output. d0bromir_cpu drops to 82.3 s at T=4 via parallel compression (**3.56×** speedup over opengene).

### WGS-PE-12.8G (ERR1044780 R1+R2, 160.6M reads, PE)

| Tool | 1T | 2T | 4T | 8T | 16T |
|---|---|---|---|---|---|
| opengene | 710.9 s | 378.2 s | 268.2 s | 269.0 s | 269.3 s |
| d0bromir_cpu | 563.1 s | 286.6 s | 146.7 s | 76.5 s | **68.8 s** |
| d0bromir_gpu | 569.6 s | 291.8 s | 149.8 s | 79.5 s | 72.9 s |
| CPU vs opengene | −20.8% | −24.2% | **−45.3%** | **−71.6%** | **−74.5%** |
| GPU vs opengene | −19.9% | −22.8% | **−44.1%** | **−70.5%** | **−72.9%** |

### WGS-PE-18.2G (ERR1044319, 234.2M reads, PE)

| Tool | 1T | 2T | 4T | 8T | 16T |
|---|---|---|---|---|---|
| opengene | 978.8 s | 519.1 s | 397.3 s | 397.5 s | 398.4 s |
| d0bromir_cpu | 743.9 s | 379.8 s | 192.4 s | 102.0 s | **100.9 s** |
| d0bromir_gpu | 749.9 s | 382.9 s | 196.9 s | 104.4 s | 105.0 s |
| CPU vs opengene | −24.0% | −26.8% | **−51.6%** | **−74.3%** | **−74.7%** |
| GPU vs opengene | −23.4% | −26.2% | **−50.4%** | **−73.7%** | **−73.6%** |

CPU peaks at T=16 (100.9 s). At T=16, d0bromir_cpu is **3.95×** faster than opengene (398.4 s) due to parallel compression.

### Observation: parallel compression shifts the bottleneck

At T≥4 the 4 libdeflate compressor threads are saturated; adding more processor threads does not reduce wall time. The GPU's per-read quality/stats pipeline (the target of GPU acceleration) is no longer the dominant cost. GPU mode adds ~3 s of CUDA initialisation overhead at all thread counts, making it 2–6% slower than CPU for gzip output. For stdout/uncompressed output the v5 speedup (8–14% at T≥8) remains.

---

## v5 Benchmark Tables (2026-03-31, stdout output)

| Tool | Panel 148 MB | WGS 6 GB | WGS 8.5 GB | WGS 9.5 GB |
|---|---|---|---|---|
| fastqc (cpu) | 15.3 s | 434.4 s | 634.5 s | 703.5 s |
| falco (cpu) | 6.6 s | 250.4 s | 367.1 s | 412.0 s |
| fastp_opengene (cpu) | **3.8 s** | 96.8 s | 144.5 s | 157.8 s |
| fastp_d0bromir (cpu) | **3.8 s** | 100.3 s | 145.6 s | 160.3 s |
| fastp_d0bromir (gpu) | 7.8 s | **88.1 s** | **134.4 s** | **147.2 s** |

## Table 2 — GPU vs CPU thread-scaling (all four datasets)

### WGS-6G (ERR1044906, 78.7M reads, 6.0 GB compressed)

| Tool | 1T | 2T | 4T | 8T | 16T | 32T |
|---|---|---|---|---|---|---|
| fastp_opengene (cpu) | 585.7 | 302.0 | 152.8 | 96.8 | 98.1 | 104.0 |
| fastp_d0bromir (cpu) | 576.7 | 306.2 | 155.1 | 100.3 | 99.6 | 103.2 |
| fastp_d0bromir (gpu) | 617.4 | 315.4 | 163.7 | **88.2** | **88.1** | **89.5** |
| GPU vs CPU-d0bromir | +7.1% | +3.0% | +5.5% | **−12.1%** | **−11.5%** | **−13.3%** |
| GPU vs fastp_opengene | +5.4% | +4.5% | +7.1% | **−8.9%** | **−10.1%** | **−14.0%** |

### WGS-8.5G (ERR1044900, 106.4M reads, 100 bp, 8.5 GB compressed)

| Tool | 1T | 2T | 4T | 8T | 16T | 32T |
|---|---|---|---|---|---|---|
| fastp_opengene (cpu) | 259.5 | 153.0 | 144.5 | 147.5 | 149.9 | 155.2 |
| fastp_d0bromir (cpu) | 251.8 | 157.7 | 148.5 | 145.6 | 149.7 | 156.8 |
| fastp_d0bromir (gpu) | 299.1 | 167.8 | **136.2** | **134.4** | **134.9** | **141.2** |
| GPU vs CPU-d0bromir | +18.8% | +6.4% | **−8.3%** | **−7.6%** | **−9.8%** | **−10.0%** |

### WGS-9.5G (ERR1044320, 120.0M reads, 100 bp, 9.5 GB compressed)

| Tool | 1T | 2T | 4T | 8T | 16T | 32T |
|---|---|---|---|---|---|---|
| fastp_opengene (cpu) | 292.6 | 165.4 | 157.8 | 160.0 | 164.6 | 169.9 |
| fastp_d0bromir (cpu) | 282.2 | 171.5 | 162.9 | 160.3 | 165.0 | 172.0 |
| fastp_d0bromir (gpu) | 333.4 | 184.9 | **147.7** | **147.2** | **147.9** | **154.2** |
| GPU vs CPU-d0bromir | +18.1% | +7.8% | **−9.3%** | **−8.2%** | **−10.3%** | **−10.3%** |

### Panel 148 MB (S1A, 1.38M reads, 151 bp)

| Tool | 1T | 2T | 4T | 8T | 16T | 32T |
|---|---|---|---|---|---|---|
| fastp_opengene (cpu) | 6.0 | 3.9 | 3.8 | 3.8 | 3.9 | 4.0 |
| fastp_d0bromir (cpu) | 5.9 | 3.9 | 3.8 | 3.9 | 3.9 | 4.0 |
| fastp_d0bromir (gpu) | 12.1 | 9.6 | 8.4 | 7.8 | 7.8 | 8.0 |

Panel file is too small to amortise GPU initialisation overhead (~3 s for dual-A100).

## Table 3 — GPU mode vs CPU-forced mode (same GPU binary, WGS-6G)

| Threads | CPU-forced | GPU mode | GPU overhead |
|---|---|---|---|
| 4T | 162.2 s | 163.4 s | +0.8% |
| 8T | 87.2 s | 88.2 s | +1.2% |
| 16T | 86.1 s | 87.9 s | +2.1% |

The speculative-stats optimisation benefits both GPU and CPU-forced paths equally (it is a CPU-side change). GPU kernel dispatch overhead accounts for only 1–2% of wall time.

---

## Architecture Details

### Slot Pool Configuration

| Constant | Value | Source |
|---|---|---|
| `NUM_SLOTS` | 8 | `cuda_stats_wrapper.h` |
| `SLOT_MAX_READS` | 16384 | `cuda_stats_wrapper.cpp` |
| `SLOT_BUF_BYTES` | 16 MB | `cuda_stats_wrapper.cpp` |
| `PACK_SIZE` | 8192 | `src/common.h` |
| `BLOCK_SIZE` | 256 | `cuda_stats.cu` |
| `READS_PER_BLOCK` | 8 | `cuda_stats.cu` |
| Total GPU device memory | ~256 MB | 8 slots × 2 × 16 MB |

### Speculative Post-Filter Statistics (v5 optimisation)

Original pipeline called `Stats::statRead()` twice per passing read: once pre-filter, once post-filter. Since ~95% of reads pass, this doubled the dominant CPU cost.

The optimised pipeline:
1. **First pass**: Call `postStats->statRead(r)` speculatively for all trimmed reads
2. **Second pass**: For the ~5% that fail filtering, call `postStats->unstatRead(r)` to reverse the histogram updates

This reduces `statRead()` calls on the hot path from $2N$ to $N + 0.05N$, saving ~45% of per-base histogram computation time.

### I/O Buffer Tuning

| Buffer | Original fastp | fastp-GPU |
|---|---|---|
| `FQ_BUF_SIZE` | 8 MB | 16 MB |
| `PACK_SIZE` | 256 | 8192 |
| `PACK_IN_MEM_LIMIT` | 256 | 64 |

---

## Build Instructions

```bash
# GPU-accelerated
cd /home/mpiuser/tools/src/fastp_d0bromir
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 -j$(nproc)

# CPU-only
make clean && make -j$(nproc)
```

## Recommendation

For WGS-scale gzip FASTQ on this hardware:

**With gzip output (v6, parallel compression):**
- **CPU mode** (`-w 8`): 75–101 s on WGS PE files — faster than GPU due to no CUDA init overhead
- **GPU mode** (`-w 8`): 79–105 s — ~4% slower than CPU; GPU advantage masked by compression bottleneck
- Parallel compression caps throughput at 4 libdeflate workers; T>4 gives minimal benefit for SE data

**Without gzip output / stdout (v5 behaviour):**
- **GPU mode** (`-w 8` or higher): 88–147 s on WGS files, **8–14% faster** than CPU-only fastp
- **CPU-only fastp:** `-w 8` achieves 97–160 s on the same files
- **Small files (< 1 GB):** CPU-only is faster; GPU initialisation overhead dominates

