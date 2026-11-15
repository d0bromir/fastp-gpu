# Performance Summary

**Last updated:** 2026-10 (final build, commit `8e87c0e`, release `v1.3.3-fastp-d0bromir.17`)
**Baseline:** upstream OpenGene fastp v1.3.3, built from the upstream tag
**Hardware:** ARM Neoverse N1 (128 cores, 246 GiB) with two NVIDIA A100 80 GB PCIe cards;
a second host (Xeon, no GPU) is used for one dataset.
**Output:** gzip FASTQ files. Input page cache is evicted before every run.

This file gives the headline numbers. The data, the protocol, and the tables of the
manuscripts are in `docs/publication/supplementary/bioautomation-revision/`
(`SUMMARY.md` and `raw/`). Older benchmark snapshots are not kept in this file,
because they were measured with earlier builds.

Speedup is the wall time of upstream fastp v1.3.3 divided by the wall time of
fastp-gpu, at the same `-w` on the same dataset.

## Where the speedup comes from

Parallel output compression gives most of the gain. Upstream compresses the output
of each file on one thread. fastp-gpu runs a pool of compressor threads.

Ablation on the paired-end 41.3 GB dataset (mean of three runs, wall time):

| Configuration | T=8 | T=32 |
|---|---|---|
| fastp v1.3.3 | 693.5 s | 680.0 s |
| fastp-gpu, all switches off | 1082.4 s | 1084.3 s |
| + asynchronous decompression | 1082.7 s | 1084.5 s |
| + parallel compression (full CPU build) | 447.3 s | 245.3 s |
| full CPU build without asynchronous decompression | 450.0 s | 341.4 s |
| full GPU build | 387.3 s | 243.8 s |

- Parallel compression is 2.4x at 8 threads and 4.4x at 32 threads.
- Asynchronous decompression has no effect at 8 threads (0.6%). At 32 threads,
  removing it makes the run 39.2% longer.
- Inflating one input file takes 204.5 s on one thread (41.3 GB data set, one mate).
  The CPU build reaches 241 s from 16 threads on, which is close to this bound.

## End-to-end wall time (ARM host, mean of 3 to 5 runs)

| Dataset | T | fastp v1.3.3 | CPU build | GPU build | Speedup (CPU / GPU) |
|---|---|---|---|---|---|
| PE 13.6 GB | 8 | 132.9 s | 71.3 s | 70.5 s | 1.86x / 1.88x |
| PE 13.6 GB | 16 | 109.6 s (n=2) | 70.4 s | 70.9 s | 1.56x / 1.55x |
| PE 13.6 GB | 32 | 115.2 s | 71.1 s | 71.5 s | 1.62x / 1.61x |
| PE 19.4 GB | 8 | 190.9 s | 98.8 s | 99.1 s | 1.93x / 1.93x |
| PE 19.4 GB | 16 | 161.0 s (n=1) | 99.2 s | 99.6 s | not reported |
| PE 19.4 GB | 32 | 168.1 s | 99.6 s | 100.1 s | 1.69x / 1.68x |
| PE 41.3 GB | 8 | 691.1 s | 447.8 s | 389.3 s | 1.54x / 1.78x |
| PE 41.3 GB | 16 | 407.7 s (n=1) | 241.0 s | 241.9 s | not reported |
| PE 41.3 GB | 32 | 696.4 s | 243.7 s | 244.9 s | 2.86x / 2.84x |
| SE 6.7 GB | 8 | hung in 3 of 3 runs | 79.3 s | 79.7 s | not reported |
| SE 6.7 GB | 16 | 195.4 s | 79.3 s | 79.6 s | 2.46x / 2.45x |
| SE 6.7 GB | 32 | 184.1 s | 79.2 s | 79.6 s | 2.33x / 2.31x |

Upstream fastp v1.3.3 hung or was killed in several runs. Cells with fewer than three
finished upstream runs have no speedup entry. Hung runs are excluded from all means.
The Xeon host (paired-end 19.4 GB, three runs per cell) gives a CPU-build speedup of
1.75x at 8 threads, 1.44x at 16 threads, and 1.71x at 32 threads.

## GPU build versus CPU build

The GPU build is faster than the CPU build in two of twelve cells, both at 8 threads:

- 41.3 GB: 389.3 s against 447.8 s (13.1% less).
- 13.6 GB: 70.5 s against 71.3 s (1.0% less, Welch p = 0.003, three runs each).

In the other ten cells the GPU build is not faster. It is 0.3% to 0.6% slower.
Inflation and output compression stay on the CPU in both modes. The GPU path is
optional and needs an NVIDIA device.

Peak device memory is 1,360 MiB in every GPU cell.

## CPU time and memory (41.3 GB, ARM host)

| Build | CPU s at T=8 | CPU s at T=32 |
|---|---|---|
| fastp v1.3.3 | 6358 | 8542 |
| CPU build | 6587 (+3.6%) | 7087 (-17.0%) |
| GPU build | 6081 (-4.4%) | 6606 (-22.7%) |

Peak resident set: CPU build 2.1 GiB (T=8) and 3.2 GiB (T=32), GPU build 2.8 and
3.8 GiB, upstream 1.2 and 1.8 GiB.

## Not measured with the final build

- Thread counts below 8.
- Panel, RNA-seq, and bisulfite datasets. Single runs of earlier builds are not reported.

## Reproduce

```bash
scripts/build_all.sh
scripts/run_benchmark.sh
```
