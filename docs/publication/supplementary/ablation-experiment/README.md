# 3-way ablation: batching vs. compute (BIOADV Reviewer 3, Comment 4)

Reviewer ask: "an ablation comparing stock fastp, the batched pipeline
with a CPU kernel, and the batched pipeline with the CUDA kernel would
help validate the reported gain."

All three numbers below are on the same dataset, WGS SE 6.3 GB
(ERR1044780, 80,314,764 reads), single worker thread (`-w 1`), OS page
cache cleared before each run, measured on `galaxy` (ARM Neoverse N1,
dual A100).

## 1. Stock (unbatched CPU filter decision)

`fastp-cpu-prof` (`make WITH_CUDA=0 PROFILING=1`), run once at `-w 1`.
This is the traditional per-read `passFilter()` call — no packing, no
per-cycle histogram/4-mer computation (those are handled separately by
the reversible-update `statRead`/`unstatRead` scheme already described
in the paper).

Result: `filter calls (incl. GPU): 52808.4 ms` = **52.81 s**
(`stock_unbatched_cpu_T1_profiling.log`).

## 2. Batched pipeline, CPU-computed

Standalone tool `ablation_batch_cpu.cpp` (built via `build_ablation.sh`
against a clean `WITH_CUDA=0` object build). Packs reads into
contiguous host buffers exactly as `cuda_stats_wrapper.cpp`'s real pack
loop does, then computes the *same* math as
`filter_and_stats_warp_kernel` (N-base/low-qual/quality scan, threshold
filter, per-cycle histogram + 4-mer for passing reads) sequentially on
CPU instead of dispatching to the GPU kernel. Single-threaded by
design, to isolate per-batch pack+compute cost.

Result (`batched_cpu_result.json`): pack = **10.55 s**, compute =
**75.09 s**, total **85.63 s**.

This tool is deliberately standalone, not wired into the shipped
pipeline — it carries zero risk to the production binary.

## 3. Batched pipeline, CUDA-computed

Real production GPU build (`fastp`, `WITH_CUDA=1 PROFILING=1`), run
once at `-w 1` on the same dataset. `mNumGPUs > 0` on this host, so all
post-filter batches were GPU-dispatched (pack + H2D + kernel + D2H, the
same profiling counters used elsewhere in the paper for kernel-level
timing).

Result: `filter calls (incl. GPU): 12332.8 ms` = **12.33 s**
(`batched_cuda_T1_profiling.log`).

## Interpretation

| Arm | Time | What it includes |
|---|---|---|
| Stock (unbatched CPU) | 52.81 s | filter decision only |
| Batched, CPU-computed | 85.63 s | pack + filter decision + full per-cycle stats |
| Batched, CUDA-computed | 12.33 s | pack + H2D + kernel (filter + full per-cycle stats) + D2H |

Packing itself is a small fraction of the batched-CPU cost (10.55 s of
85.63 s, ~12%). The per-read statistics math dominates, and that is
exactly what the GPU accelerates: the same math costs 75.09 s on a
single CPU thread and 12.33 s end-to-end (including both transfers) on
the GPU — about 7x less. This confirms the reported gain comes from
parallel execution of the per-read math, not from the batched memory
layout by itself. It also shows that naively porting the batched
kernel math to CPU would be a regression versus the CPU-only build's
existing reversible-update scheme (85.63 s vs. 52.81 s for filtering
alone) — which is why the CPU-only build does not use this batched
approach at all.

## Reproduce

```
bash build_ablation.sh          # builds experiments/ablation_batch_cpu
                                 # against a clean WITH_CUDA=0 object set
./ablation_batch_cpu <in1.fastq.gz> 1 8192
```

For arms 1 and 3, build with `PROFILING=1` (`WITH_CUDA=0` and `=1`
respectively) and read the `filter calls (incl. GPU)` line of the
profiling summary printed to stderr.
