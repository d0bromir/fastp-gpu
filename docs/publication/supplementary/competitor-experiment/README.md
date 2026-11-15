# Competitor comparison: fastp-gpu vs. RabbitQCPlus (BIOADV review, Comment 3)

Reviewer ask: compare against another GPU/accelerated FASTQ QC tool,
not only against upstream fastp.

## Tool chosen: RabbitQCPlus

[RabbitQCPlus](https://github.com/RabbitBio/RabbitQCPlus) is an
accelerated FASTQ QC/filtering tool from the same problem space as
fastp. It was the most direct point of comparison we found (see
`gpu-fastq-qc-competitor-landscape.md` in the parent directory for the
broader survey).

## Portability finding (why x86, not ARM)

RabbitQCPlus does not build on aarch64: its bundled `pugz` component
unconditionally includes `<pmmintrin.h>` (x86 SSE3 intrinsics) with no
ARM/NEON code path in the Makefile. This is a genuine limitation of
the tool, not a configuration issue on our end. We therefore ran the
comparison on `a2` (Intel Xeon Gold 5218, 64 physical cores, x86-64,
no GPU) — the same host already used elsewhere in this paper for the
x86 portability results — rather than on the ARM/A100 host used for
the headline numbers.

RabbitQCPlus also required building zlib 1.3.1 from source into a
user-space prefix (`~/appnotes-competitor-work/local`), since the
system's `zlib1g-dev` package was not installed and passwordless sudo
was not available on this host. This is a host packaging gap, not a
tool limitation.

## Methodology

Both tools run at default settings, same input files, same host,
`-w 8` and `-w 32`, OS page cache cleared before each run. Datasets:
WGS PE 18.2 GB (ERR1044319) and WGS PE 40 GB (DRR216653).

`fastp_cpu` here is `fastp-gpu` v1.3.3-d0bromir, CPU-only build
(`make WITH_CUDA=0`, **without** `PROFILING=1`) — the same build
configuration used for every other CPU-mode number in this paper.

### Build-configuration error, caught and corrected

The first pass (`competitor_comparison.csv`, timestamps ~19:5x) used a
`fastp-cpu` binary on this host that turned out to have been built
with `PROFILING=1` left on from earlier work — its timing
instrumentation (per-call `chrono::now()` and atomic counters on every
worker thread) adds real overhead and is not representative of the
binary users actually run. That run's first data point
(WGS_PE_18.2G, T=8) was also caught live as contaminated by an
orphaned `RabbitQCPlus` smoke-test process still consuming CPU in the
background. Both issues are visible in the raw logs
(`raw_results/competitor_comparison.csv`,
`raw_results/rerun_...`) and are kept for the record, but are
**not** the numbers used in the paper.

All four `fastp_cpu` measurements were redone with a freshly built,
non-instrumented binary (`fastp-cpu-clean`, `make WITH_CUDA=0`) after
confirming no other process was running on the host
(`raw_results/competitor_comparison_clean.csv`). RabbitQCPlus's own
four measurements were unaffected by either issue (RabbitQCPlus has
no such profiling switch) and are reused as-is from the first pass.

## Results

| dataset | threads | fastp-gpu (CPU, clean) | RabbitQCPlus | faster |
|---|---|---|---|---|
| WGS_PE_18.2G | 8  | 163.87 s | 234.10 s | fastp-gpu, 1.43x |
| WGS_PE_18.2G | 32 | 112.96 s | 98.18 s  | RabbitQCPlus, 1.15x |
| WGS_PE_40G   | 8  | 726.64 s | 1057.47 s | fastp-gpu, 1.46x |
| WGS_PE_40G   | 32 | 413.42 s | 331.58 s | RabbitQCPlus, 1.25x |

fastp-gpu wins at low thread count (T=8) on both datasets; RabbitQCPlus
wins at high thread count (T=32) on both datasets, consistent with
better thread scaling on this host at high core counts. Neither tool
dominates the other across the tested range.

Source: `raw_results/competitor_comparison_clean.csv` (fastp) and the
`rabbitqcplus` rows of `raw_results/competitor_comparison.csv`.

## Reproduce

```
bash competitor_experiment.sh
```
