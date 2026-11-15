# Core-budget experiment: CPU mode vs GPU mode on a fixed number of cores

Reviewer ask (BIOADV R2, Reviewer 3, comment 1): show a realistic setting in which
GPU offload gives a reproducible end-to-end or resource-efficiency benefit over the
CPU-only build.

## Setting

A cluster job usually receives a fixed number of cores. Each run was confined to C
cores with `taskset -c 64-(64+C-1)` and fastp-gpu got `-w C`. All threads of the
process (reader, decompressor, workers, compressors, writer, CUDA host threads)
share these C cores.

- Host: galaxy (Ampere Altra Max, 128 cores; 2x NVIDIA A100 80 GB PCIe). The host is
  shared; the 1-min load average at the start of each run is in column `load1`.
- Binaries: release v1.3.3-fastp-d0bromir.17 (commit 8e87c0e), `fastp-cpu` (CPU build)
  and `fastp` (CUDA build, uses both A100s).
- Datasets: WGS SE 6.3 GB (ERR1044780_1), WGS PE 18.1 GB (ERR1044319), WGS PE
  38.5 GB (DRR216653). Sizes as in the paper.
- C = 4, 8, 16 (38.5 GB: 8, 16). Three repeats per cell and mode; cell order
  shuffled within each repeat; page cache dropped before every run; output to
  /dev/shm.
- Run date: 2026-10-31. All 48 runs exited 0. Within each dataset the md5 of the
  14 checked JSON report fields is identical in every run of both modes.

## Results (mean +- SD, n = 3; delta = GPU relative to CPU mode)

| data | C | CPU wall s | GPU wall s | delta wall | CPU s (user+sys) CPU / GPU | delta CPU s |
|---|---|---|---|---|---|---|
| SE 6.3 GB | 4 | 149.2 +- 0.8 | 140.0 +- 1.8 | -6.2% | 553 / 515 | -6.9% |
| SE 6.3 GB | 8 | 98.0 +- 1.3 | 92.2 +- 2.3 | -6.0% | 571 / 533 | -6.6% |
| SE 6.3 GB | 16 | 80.0 +- 0.5 | 80.7 +- 0.3 | +0.8% | 605 / 562 | -7.0% |
| PE 18.1 GB | 4 | 453.3 +- 3.7 | 422.1 +- 1.6 | -6.9% | 1800 / 1674 | -7.0% |
| PE 18.1 GB | 8 | 226.7 +- 0.4 | 212.5 +- 0.5 | -6.3% | 1781 / 1663 | -6.6% |
| PE 18.1 GB | 16 | 133.2 +- 3.3 | 127.2 +- 2.0 | -4.6% | 1825 / 1707 | -6.5% |
| PE 38.5 GB | 8 | 811.1 +- 0.6 | 747.5 +- 2.4 | -7.8% | 6428 / 5931 | -7.7% |
| PE 38.5 GB | 16 | 427.3 +- 4.9 | 412.5 +- 2.6 | -3.5% | 6511 / 6029 | -7.4% |

GPU mode used 0.6-0.8 GB more peak RSS (maximum per dataset) (column `rss_kb`).

## Files

- `core_budget_experiment.sh` - the driver (runs on galaxy from ~/bioauto_rev; needs
  `watched_run.sh` from scripts/local).
- `raw/core_budget.csv` - one row per run.
- `raw/done.txt` - completion stamp.
