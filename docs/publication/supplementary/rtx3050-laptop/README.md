# RTX 3050 laptop: CPU build vs GPU build on the panel inputs

Purpose: check the Application Notes claim about a consumer GPU. The earlier
claim ("GPU mode 4-47% slower in every configuration") had no archived raw
data, so it was re-measured on 2026-11-15.

## Setup

- Host: laptop, NVIDIA GeForce RTX 3050 Laptop GPU (compute capability 8.6,
  4 GiB), driver 610.62, 8 logical CPUs, WSL2, Ubuntu 24.04, GCC 13.3,
  CUDA 12.0 (nvcc).
- Source: `src/` and `Makefile` identical to release v1.3.3-fastp-d0bromir.17
  (commit 8e87c0e; `git diff 8e87c0e` over these paths is empty). Built in a
  clean worktree: `WITH_CUDA=1 make fastp` (GPU build), then `make clean`,
  `rm obj/*.o`, `make fastp` (CPU build).
- Input: panel sample S1A_S1_L001 (the paper's Panel SE / Panel PE data),
  gzip level 6. Decompressed R1 MD5 d9f0e6201afa9d099e16ef6856d81770 equals the
  original FASTQ. 1,381,299 reads pass the filter, as in the figure data.
  The gzip files here are 118/128 MB; the paper's 148/156 MB files used another
  compression level.
- Runs: SE and PE, `-w` 1, 2, 4, 8, CPU and GPU build, three repeats,
  shuffled order within each repeat, output to /dev/shm. The page cache was
  not dropped (no root on this host); inputs were read once before the sweep.
- GPU engagement: `gpu_util_smoke_test.txt` samples `nvidia-smi` during one
  PE run at `-w 4` (up to 25% utilisation, 333 MiB).

## Results (mean of 3 runs, seconds; delta = GPU relative to CPU)

| layout | -w | CPU | GPU | delta | Welch t |
|---|---|---|---|---|---|
| SE | 1 | 8.58 | 8.86 | +3.3% | 0.33 |
| SE | 2 | 5.66 | 6.74 | +19.0% | 1.55 |
| SE | 4 | 4.52 | 5.48 | +21.4% | 2.46 |
| SE | 8 | 4.92 | 5.45 | +10.8% | 1.55 |
| PE | 1 | 13.87 | 13.25 | -4.5% | -0.56 |
| PE | 2 | 10.94 | 11.08 | +1.2% | 0.06 |
| PE | 4 | 9.25 | 9.91 | +7.2% | 0.23 |
| PE | 8 | 8.75 | 8.74 | -0.1% | -0.01 |

No difference is significant at the 5% level (|t| < 2.8 for about 4 degrees
of freedom). Run-to-run spread on this host reaches 40% (PE). Decompressed
output was identical in all 48 runs: one MD5 per mate for each layout,
in both builds.

Conclusion for the paper: on this GPU, GPU mode was not faster than CPU mode,
and its output was bit-identical. The old "4-47% slower in every
configuration" is not reproduced and is withdrawn.

## Unit tests

`fastp test` with the GPU build on this laptop: ALL PASSED, including the
CUDA tests (`unittest_output.log`).

## Files

- `unittest_output.log` - output of `fastp test` (GPU build).

- `rtx3050_sweep.sh` - driver: `rtx3050_sweep.sh <bin_dir> <panel_dir> <out_csv>`.
- `rtx3050.csv` - one row per run (wall time, MD5 of decompressed outputs).
- `gpu_util_smoke_test.txt` - nvidia-smi samples (utilisation, memory).
