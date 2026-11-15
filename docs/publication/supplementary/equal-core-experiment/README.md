# Equal-core experiment: upstream fastp, fastp-gpu CPU mode and GPU mode on the same cores

Associate Editor's request (JCB-2026-0222.R2, major comment): `-w` does not bound
the thread count, so a comparison at equal `-w` does not give the tools equal
resources. These runs compare the tools on equal cores. They also count the
threads of each binary.

## Setting

- ARM host: galaxy (Ampere Altra Max, 128 cores, 2x A100 80 GB PCIe), shared with
  other users. The 1-min load average at the start of each run is in column `load1`
  (2.1 to 30.3).
- Binaries: upstream OpenGene fastp v1.3.3 (`fastp_opengene/fastp`). fastp-gpu
  release .17, commit 8e87c0e (`fastp_final_gx/fastp-cpu` and `fastp_final_gx/fastp`).
  These are the binaries of the paper's Table 4 sweep.
- Each run is confined with `taskset -c 64-(63+C)` and gets `-w C`. All threads of
  the process share the C cores. C = 8, 16, 32. Four WGS datasets. Three repeats per
  cell. The cell order is shuffled within each repeat, the page cache is dropped
  before every run, and output goes to /dev/shm. A watchdog kills a run whose output
  stops growing for 300 s (`exit` = HANG).
- Run date: 2026-11-11/12.

## Files

| file | content |
|---|---|
| `equal_core_experiment.sh` | driver, upstream/CPU/GPU on C cores (raw/equal_core.csv) |
| `equal_core_ablation.sh` | CPU build with FASTP_COMPRESSORS=1 or FASTP_NO_ASYNC_DECOMP=1 on C = 8, 32 cores, 41.3 GB dataset (raw/equal_core_ablation.csv) |
| `thread_census.sh` | threads and per-thread CPU time at -w 8 and 32, no core limit, 13.6 GB PE (raw/thread_census.csv) |
| `thread_census_one_compressor.sh` | same, CPU build with both ablation switches set (raw/thread_census_one_compressor.csv) |
| `thread_census_two_compressors.sh` | control: FASTP_COMPRESSORS=2 with async decompression off, -w 8 (raw/thread_census_two_compressors.csv) |
| `equal_core_xeon.sh` | Xeon host (a2): upstream and CPU build on cores 0..C-1, 19.4 GB PE (raw/equal_core_xeon.csv) |
| `summarize_equal_core.py` | means, SDs, CPU seconds, Welch tests (no scipy); output in SUMMARY.txt |

Dataset labels: ERR1044780 SE = WGS SE 6.7 GB, ERR1044780 PE = WGS PE 13.6 GB,
ERR1044319 = WGS PE 19.4 GB, DRR216653 = WGS PE 41.3 GB.

## Results in short (ARM host)

- Upstream hung in 9 of 36 runs. The CPU and GPU builds finished all 72.
  Within each dataset, the md5 of the 14 checked JSON fields is the same in all 99
  finished runs.
- CPU build against upstream, in the 8 cells with >= 2 finished upstream runs:
  19.5-56.1% less wall time, 3.8-31.3% fewer CPU seconds.
- GPU build against CPU build: 3.2-7.9% less wall time in 8 of 12 cells
  (Welch p <= 0.016). The SE file on 32 cores is 0.4% slower (p = 0.031).
- Ablation on equal cores (41.3 GB): one compressor per file 1181.1 s (8 cores) and
  1085.3 s (32 cores), against 809.0 s and 288.0 s for the full CPU build.
- Thread census: at -w 8 upstream creates 11 threads (9.4 busy cores), the CPU build
  25 (17.8), the GPU build 30 (17.0). In the ablation configuration two compressor
  threads run for 99% of the wall time. With FASTP_COMPRESSORS=2 the wall time falls
  from 257.8 s to 130.5 s.

## Results in short (Xeon host, a2)

- Upstream v1.3.3 and the CPU build (`fastp_opengene/fastp`, `fastp_d0bromir/fastp-cpu`,
  the binaries of the paper's Xeon table) on cores 0..C-1 with -w C, 19.4 GB PE,
  three repeats. Cores 0-15 are NUMA node 0, 16-31 node 1. Page cache evicted with
  posix_fadvise (no sudo). Run date: 2026-11-12.
- 8 cores: 330.6 s against 215.4 s (-34.8% wall, -27.5% CPU seconds).
- 32 cores: 280.2 s against 139.2 s (-50.3% wall, -43.7% CPU seconds).
- 16 cores: upstream hung in 2 of 3 runs; no speedup. The CPU build finished all 9
  runs with the full read count (234,238,034).
