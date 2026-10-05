# Upstream commits intentionally not applied to this fork

This file lists commits from `OpenGene/fastp` that were inspected during a
rebase but deliberately **not** cherry-picked. Each entry must record the
upstream SHA, the rationale, and (if possible) a regression test that proves
the fork's behavior remains correct without it.

## v1.2.2 → v1.3.3 rebase

### `5bb7473` — "update producer/consumer list to fix possible hang"

- File: `src/singleproducersingleconsumerlist.h`
- Date: 2026-04-24
- Status: **rejected**
- Reason: this fork's writer pipeline is significantly different from
  upstream (custom `writerthread.cpp`, GPU-side staging in
  `cuda_stats*`/`gds_pipeline.*`, and a concurrent slot pool). Upstream's
  `canBeConsumed()` change makes the queue report a single-item tail as
  consumable before any publication barrier; in our pipeline this has been
  observed to interact with the GPU writer thread and cause hangs / data
  races under high `--thread` counts.
- The atomic load/store hygiene part of the patch (acquire/release on
  `producerFinished` / `consumerFinished` / `nextItemReady`) is *useful* and
  may be backported as a separate, narrower commit if profiling shows torn
  reads on aarch64. The semantic change to `canBeConsumed()` must remain
  out.
- Regression test: run `scripts/run_benchmark.sh` (the canonical
  benchmark + validation driver) with `--thread 32` (or higher) on a
  multi-GB BGZF input. Both the read-count and deep (FASTQ md5 + JSON
  biology fields) validation built into the script must pass and the run
  must complete without hangs; this is the early-warning signal that the
  upstream change has not silently leaked back in.

## How to add an entry

1. `git show <sha>` — read the patch.
2. Decide whether the change is biologically meaningful (apply it) or
   purely a threading/IO refactor (likely conflicts with the fork's
   pipeline — reject and document here).
3. Add a section above with SHA, file(s), reason, and the test that
   demonstrates the fork stays correct.
