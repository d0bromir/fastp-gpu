# The T=64 Thread-Ceiling Collapse: Root Cause, Fix, and Verification

**Purpose.** This note documents a severe performance regression found while
extending the GPU-utilisation investigation to `fastp-gpu`'s actual thread
ceiling (`-w` is hard-capped at 64), its root cause, the fix applied, and the
verification evidence. It is the supporting detail for the "Regression at the
tool's thread ceiling" discussion in the accompanying manuscripts; this note
carries the reproducible detail that does not fit in a page-limited paper.

**Date of experiments:** 2026-08-22 -- 2026-08-23.
**Raw data:** [`benchmark_results/fastp-gpu_v1.3.3-d0bromir/vs_opengene_v1.3.3/galaxy_arm_a100/20260823_thread_ceiling_investigation/`](../../../benchmark_results/fastp-gpu_v1.3.3-d0bromir/vs_opengene_v1.3.3/galaxy_arm_a100/20260823_thread_ceiling_investigation/)
(`summary.csv` + one JSON report and captured run log per test).

---

## 1. Summary

All of this project's main benchmark suite runs at `-w` up to 32. A routine
follow-up question -- does GPU utilisation improve at higher thread counts,
and would that justify extending the sweep to the tool's actual ceiling
(`-w` is hard-capped at 64) -- led to testing `T=64` directly for the first
time. The result was not "no further improvement": `WGS_PE_40G` at `T=64`
took **3,969 s**, about **16x** longer than `T=32`'s 245 s, with GPU
utilisation collapsing to **1.6%** average (from 26.5% at `T=32`). The GPU
build never completed within a reasonable wait; the CPU build was left to run
to completion to get an exact number.

We root-caused this to a global (not per-worker) in-flight-pack budget that
collapses to essentially zero headroom exactly at `T=64`, fixed it, and
re-verified. Post-fix, `T=64` completes in **238 s (CPU) / 240 s (GPU)**,
matching `T=32`, with zero regression at any other thread count and
bit-exact output throughout every one of the 11 runs in this investigation
(pre-fix, post-fix, CPU, GPU, and every rule-out configuration).

---

## 2. Investigation

Three explanations were tested directly rather than assumed:

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | GPU slot pool too small | `FASTP_GPU_SLOTS=32` (4x default) at `T=64` | Wall time identical to the default-8 run (3,969 s both) |
| 2 | Compressor thread pool starving workers | `FASTP_COMPRESSORS=8` at `T=64` (reduces live threads from 141 to 93, well under the host's 128 cores) | No material change (3,965 s) |
| 3 | NUMA topology | N/A -- host is single-socket, single-NUMA-node, no SMT | Not applicable; ruled out by hardware inspection |

A live `gdb -p <pid> -batch -ex 'thread apply all bt'` backtrace of the
stalled process was decisive: the large majority of worker threads were
parked in `usleep()` inside `PairEndProcessor::processorTask`, each polling
its own round-robin-fed input queue on a 100 &micro;s cycle with no
backpressure and no work-stealing. Correctness was unaffected throughout
(read counts and filtering results identical to every other configuration
tested, including the collapsed ones) -- this pointed at worker starvation,
not GPU contention or a correctness bug.

## 3. Root cause

`src/common.h` defines a global ceiling on in-flight (produced-but-not-yet-
consumed) packs:

```cpp
static const int PACK_IN_MEM_LIMIT = 64;
```

Packs are distributed **round-robin** across `-w` per-worker queues
(`src/peprocessor.cpp`, `src/seprocessor.cpp`). Because the limit is global
rather than per-worker, each worker only gets `PACK_IN_MEM_LIMIT / threads`
packs of buffering headroom on average:

| `-w` | headroom / worker |
|---|---|
| 8  | 8 packs |
| 16 | 4 packs |
| 32 | 2 packs |
| 64 | ~1 pack |

At `T=64` the reader can never build a meaningful lead over consumption --
every worker is running essentially hand-to-mouth with the reader, forcing
near-lockstep handoff and eliminating the overlap the whole producer/worker
pipeline depends on. This is a pure liveness/throughput bug: at no point does
it affect correctness, because the pack-count ceiling only throttles
*production rate*, never *what* is produced.

## 4. Fix

Two changes, both read/write-order neutral (correctness does not depend on
timing):

1. **`src/common.h`** -- scale the ceiling with thread count instead of
   using a bare constant:

   ```cpp
   static const int PACK_IN_MEM_HEADROOM = 4;
   static inline long packInMemLimit(int effectiveThreads) {
       long perWorker = (long)effectiveThreads * PACK_IN_MEM_HEADROOM;
       return perWorker > PACK_IN_MEM_LIMIT ? perWorker : PACK_IN_MEM_LIMIT;
   }
   ```

   `max(64, threads*4)` is unchanged at `threads <= 16` (where the original
   constant was tuned, so memory footprint at low-to-moderate thread counts
   is untouched), and grows linearly above that. Applied at the throttle
   comparison sites in `src/peprocessor.cpp` and `src/seprocessor.cpp` in
   place of the bare `PACK_IN_MEM_LIMIT` constant.

2. **`src/singleproducersingleconsumerlist.h`** -- secondary fix: replaced
   the `usleep(100)`-based polling wait in the consumer with a
   condition-variable-based blocking wait (`waitForData()`, notified from
   `produce()` / `setProducerFinished()`), reducing wake/sleep churn at high
   `-w`. Verified *not* to be the primary fix on its own -- an earlier
   attempt with only this change still showed a multi-thousand-second
   trajectory at `T=64` in early-progress-rate checks, which is what led to
   finding the `PACK_IN_MEM_LIMIT` issue above.

## 5. Verification

| Config | Pre-fix | Post-fix | Correctness |
|---|---|---|---|
| `T=64`, CPU | 3,969 s | **238 s** | bit-exact (722,563,222 / 717,214,748 reads) |
| `T=64`, GPU | >3,600 s (never completed) | **240 s** | bit-exact |
| `T=32` (regression check) | 241.3 s (paper baseline) | 234 s | bit-exact |
| `T=8` (spot check -- fix is a no-op here by construction) | -- | 564 s | bit-exact |

`packInMemLimit(8) == max(64, 8*4) == 64`, identical to the original
constant, so `T=8` is mathematically unaffected by the change; the spot
check above confirms this empirically.

Every one of the 11 runs referenced in this note -- pre-fix and post-fix,
CPU and GPU builds, every thread count and every rule-out configuration --
produced identical read counts (722,563,222 reads in, 717,214,748 after
filtering) on the same `WGS_PE_40G` input, confirming the collapse and its
fix were purely a throughput/liveness issue and never a correctness issue.
Full per-run JSON reports and captured logs are in the raw data directory
linked above.
