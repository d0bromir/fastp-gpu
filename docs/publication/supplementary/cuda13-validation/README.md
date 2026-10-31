# CUDA 13.1 build/correctness validation

Answers Reviewer 1's question ("do the authors foresee that this code can
be maintained with coming CUDA versions, e.g. CUDA 13") empirically rather
than speculatively.

**Date:** 2026-08-29
**Host:** `galaxy` (ARM Neoverse N1, dual NVIDIA A100 80GB PCIe)
**Toolkit:** CUDA 13.1.115 (`nvcc_version.txt`), driver 595.71.05
**Repo state:** fresh clone of `d0bromir/fastp` at commit `ed94dd0`, built in
an isolated worktree (`~/fastp-gpu-appnotes-work/repo` on `galaxy`), no
source changes required for the CUDA-13 toolkit -- `Makefile` already
auto-detects `/usr/local/cuda-13.1` ahead of the default `cuda-12.6`.

## What was run

1. `WITH_CUDA=1 make -j32 fastp` against CUDA 13.1 -- built clean, no
   warnings or errors beyond normal.
2. `./fastp test` (`unittest_output.log`) -- full unit test suite,
   including the CUDA-specific tests (`cuda_compute_read_stats`,
   `CudaStatsWrapper::processBatch{,StatsOnly,FilterAndStats}`,
   `gpu_trim_head_tail`, `gpu_trim_poly_g`, `gpu_trim_quality`,
   `cuda_fastq_parse_device`, multi-GPU load spread across both A100s):
   **ALL PASSED**.
3. Real-data correctness: ran the CUDA-13 GPU build (`gpu.json`) and a
   separately-built CPU-only binary from the identical source (`cpuonly.json`,
   `WITH_CUDA` unset) on `DRR262998` (Panel_public, paired-end, ~1M read
   pairs). Decompressed FASTQ output `md5sum` was **identical** between the
   two builds on both mates:
   - R1: `cd5ff534d23247465663882409bbf674` (both builds)
   - R2: `27d9f17213954ddf4c5a90c0c7c33de1` (both builds)

`cpu.json` is a duplicate GPU-mode run kept for reference (both GPUs engage
automatically once `WITH_CUDA=1`; there is no separate CPU flag on the same
binary -- see the paper's own description of the CPU-only vs. GPU-mode
build split).

## Conclusion

No source changes were needed for CUDA 13.1: the kernel code uses only
long-stable CUDA primitives (warp shuffles, `atomicAdd`, `cudaMemcpyAsync`,
CUDA streams/events, pinned host memory) with no deprecated-API usage
between 12.6 and 13.1. This is reflected in the manuscript's Vulkan/CUDA-13
discussion.
